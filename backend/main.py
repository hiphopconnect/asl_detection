import configparser
import time
from contextlib import asynccontextmanager

import cv2
from backend.algorithms import Algorithm, AlgorithmManager, AlgorithmType
from backend.cameras import (
    AbstractCamera,
    CameraManager,
    CameraType,
    OpenCVCamera,
    VideoCaptureOpenError,
    VideoCaptureReadError,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from backend.custom_types import ModelName
from backend.vision import MediaPipeHolistics


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

    # Open and store persistent camera instances"
    cameras = []
    for idx in [0, 4]:
        try:
            cam = OpenCVCamera(address=str(idx), type=CameraType.RGB)
            cameras.append(cam)
        except VideoCaptureOpenError as err:
            print(f"Could not open camera {idx}: {err}")
    app.camera_manager = CameraManager(cameras=cameras)
    app.algorithm_manager = AlgorithmManager([MediaPipeHolistics()])
    yield
    # On shutdown: release all cameras
    for cam in cameras:
        if hasattr(cam, "_capture") and cam._capture is not None:
            cam._capture.release()
            print(f"Camera {cam.index} released on shutdown.")


app = FastAPI(lifespan=lifespan)

# CORS configuration for the frontend (e.g., http://127.0.0.1:3000)
origins = ["http://127.0.0.1:3000"]
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
    # Iteriere über die persistente Kamera-Liste (Dictionary mit Index als Schlüssel)
    for idx, cam in app.camera_manager._cameras.items():
        cameras.append({"id": idx, "name": f"Camera {idx}"})
    return cameras


@app.get("/video/")
async def video_feed(
        request: Request,
        camera_type: int,  # Erwartet einen Index (z. B. 0 oder 4)
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
            algorithm = request.app.algorithm_manager.get_algorithm_by_name(name=model_name)
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
                print("Frame shape:", frame.shape, "Frame dtype:", frame.dtype)
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
