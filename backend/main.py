import configparser
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.algorithms import Algorithm, AlgorithmManager
from backend.cameras import (
    AbstractCamera,
    CameraManager,
    CameraType,
    OpenCVCamera,
    VideoCaptureOpenError,
    VideoCaptureReadError,
)
from backend.custom_types import ModelName
from backend.upload_endpoints import router as upload_router

# from backend.vision import MediaPipeHolistics
from backend.vision import ASLFingerspelling


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes persistent camera instances (e.g., for indices 0 and 4)
    and the AlgorithmManager.
    """
    app.config = configparser.ConfigParser()
    success = app.config.read("./config.cfg")
    if not success:
        raise FileNotFoundError("Could not find config file!")

    # Open and store persistent camera instances (example: indices 0 and 4)
    cameras = []
    for idx in [0, 4]:
        try:
            cam = OpenCVCamera(address=str(idx), type=CameraType.RGB)
            cameras.append(cam)
        except VideoCaptureOpenError as err:
            print(f"Could not open camera {idx}: {err}")
    app.camera_manager = CameraManager(cameras=cameras)
    app.algorithm_manager = AlgorithmManager([ASLFingerspelling()])
    yield
    # On shutdown: release all cameras
    for cam in cameras:
        if hasattr(cam, "_capture") and cam._capture is not None:
            cam._capture.release()
            print(f"Camera {idx} released on shutdown.")


app = FastAPI(lifespan=lifespan)

# Include the upload router
app.include_router(upload_router)

# Mount static files: serves the uploaded videos
BASE_DIR = Path(
    __file__
).parent.parent  # main.py is in "backend", so one level up is the project root
UPLOAD_DIRECTORY = str(BASE_DIR / "VideoFiles")
os.makedirs(
    UPLOAD_DIRECTORY, exist_ok=True
)  # Stelle sicher, dass das Verzeichnis existiert

app.mount("/static", StaticFiles(directory=UPLOAD_DIRECTORY), name="static")

# CORS configuration for the frontend (e.g., http://127.0.0.1:3000 and http://localhost:3000)
origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/cameras/")
def list_cameras():
    """
    Returns the cameras stored in the CameraManager.
    """
    cameras = []
    for i in range(7):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cameras.append({"id": i, "name": f"Camera {i}"})
            cap.release()
    return cameras


@app.get("/video/")
async def video_feed(
    request: Request,
    camera_type: int,  # Expects an index (e.g., 0 or 4)
    model_name: ModelName,
):
    """
    Provides a video stream from the persistent camera based on the index and an optional algorithm.
    """
    try:
        camera = request.app.camera_manager.get_camera_by_index(camera_type)
    except KeyError as err:
        raise HTTPException(status_code=500, detail=str(err))

    algorithm = None
    if model_name is not ModelName.NONE:
        try:
            algorithm = request.app.algorithm_manager.get_algorithm_by_name(
                name=model_name
            )
        except Exception as e:
            print("Algorithm error:", e)
            algorithm = None

    return StreamingResponse(
        content=get_view(camera, algorithm),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


def get_view(camera: AbstractCamera, algorithm: Algorithm = None):
    """
    Generator that continuously provides frames from the persistent camera, optionally applies the algorithm,
    and returns the frames encoded as JPEG.
    Since the camera is persistent, we do not release it with every frame.
    """
    try:
        while True:
            try:
                frame = camera.get_frame()
                if algorithm is not None:
                    frame = algorithm(frame)
                    print("After algorithm - frame shape:", frame.shape)
                ret, buf = cv2.imencode(".jpg", frame)
                if not ret:
                    print("Failed to encode frame")
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            except VideoCaptureOpenError as e:
                print("VideoCaptureOpenError, frame skipped:", e)
                continue
            except VideoCaptureReadError as e:
                print("VideoCaptureReadError, frame skipped:", e)
                continue
            except Exception as e:
                print("Error in get_view:", e)
                continue
    except GeneratorExit:
        print("Stream closed.")


@app.get("/uploaded_video/")
async def uploaded_video_feed(request: Request, video_url: str, model_name: ModelName):
    """Verarbeitet ein hochgeladenes Video mit dem gewählten Algorithmus"""

    # If no Algrotihm was selected, return Original-Video
    if model_name is ModelName.NONE:
        return {"status": "original", "video_url": video_url}

    # Get Video Path
    video_path = None
    if "/static/" in video_url:
        filename = video_url.split("/static/")[-1]
        video_path = os.path.join(UPLOAD_DIRECTORY, filename)

    if not video_path or not os.path.exists(video_path):
        return {"status": "error", "error": "Video not found"}

    # Load Algorithm
    try:
        algorithm = request.app.algorithm_manager.get_algorithm_by_name(name=model_name)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "error": "Could not open video"}

    # Video-Properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Capture processed Frames for GIF-Creation
    frames = []
    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampling_interval = max(1, total_frames // 120)

    duration = 1000 // int(fps) * sampling_interval

    try:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Just take every n Frame (based on sampling_interval)
            if frame_index % sampling_interval == 0:
                # Process frames
                processed_frame = algorithm(frame)

                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image and keep the original Sizes of video
                pil_image = Image.fromarray(rgb_frame)

                # Add to frame array
                frames.append(pil_image)
                processed_frames += 1

            frame_index += 1

    finally:
        cap.release()
    if processed_frames == 0:
        return {"status": "error", "error": "No frames processed"}

    # Save GIF-File
    gif_filename = f"processed_{uuid.uuid4()}.gif"
    gif_path = os.path.join(UPLOAD_DIRECTORY, gif_filename)

    # Save GIF with adjusted Length
    try:
        # Create optimized GIF-Animation
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=[duration * 3] + [duration] * (len(frames) - 1),
            loop=0,  # 0 = endless Loop
        )

        # Return GIF-URL
        return {
            "status": "completed",
            "video_url": f"/static/{gif_filename}",  # use /static/ for GIFs
            "frames_processed": processed_frames,
            "is_gif": True,
        }

    except Exception as e:
        return {"status": "error", "error": f"Failed to create GIF: {str(e)}"}
