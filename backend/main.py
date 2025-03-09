import configparser
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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


@app.get("/video_file/{filename}")
async def get_video_file(filename: str):
    """Serve video files with correct content type"""
    video_path = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(path=video_path, media_type="video/mp4", filename=filename)


@app.get("/uploaded_video/")
async def uploaded_video_feed(request: Request, video_url: str, model_name: ModelName):
    """Verarbeitet ein hochgeladenes Video mit dem gewählten Algorithmus"""
    import uuid

    # Wenn kein Algorithmus gewählt wurde, Original-Video zurückgeben
    if model_name is ModelName.NONE:
        return {"status": "original", "video_url": video_url}

    # Video-Pfad ermitteln
    video_path = None
    if "/static/" in video_url:
        filename = video_url.split("/static/")[-1]
        video_path = os.path.join(UPLOAD_DIRECTORY, filename)

    if not video_path or not os.path.exists(video_path):
        return {"status": "error", "error": "Video not found"}

    # Algorithmus laden
    try:
        algorithm = request.app.algorithm_manager.get_algorithm_by_name(name=model_name)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Video öffnen
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "error": "Could not open video"}

    # Video-Eigenschaften
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Ausgabedatei vorbereiten
    output_filename = f"processed_{uuid.uuid4()}.mp4"
    output_path = os.path.join(UPLOAD_DIRECTORY, output_filename)

    # VideoWriter mit XVID codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        return {"status": "error", "error": "Could not create output video"}

    # Frames verarbeiten
    processed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = algorithm(frame)
            out.write(processed_frame)
            processed_frames += 1
    finally:
        cap.release()
        out.release()

    # Prüfen ob Video erstellt wurde
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return {"status": "error", "error": "Failed to create video"}

    # Berechtigungen setzen
    try:
        os.chmod(output_path, 0o644)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")

    return {
        "status": "completed",
        "video_url": f"/video_file/{output_filename}",
        "frames_processed": processed_frames,
    }
