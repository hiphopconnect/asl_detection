import configparser
from contextlib import asynccontextmanager

import cv2
from algorithms import Algorithm, AlgorithmManager, AlgorithmType
from cameras import (
    AbstractCamera,
    CameraManager,
    CameraType,
    OpenCVCamera,
    VideoCaptureOpenError,
    VideoCaptureReadError,
)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from vision import ObjectDetection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of a FastAPI application.
    This context manager is responsible for setting up resources when the FastAPI application starts
    and for cleaning up resources before the application shuts down.

    Args:
        app (FastAPI): The instance of the FastAPI application.
    """

    app.config = configparser.ConfigParser()
    success = app.config.read("../config.cfg")
    if not success:
        raise FileNotFoundError("Could not find config file!")

    cameras = []
    try:
        cameras.append(
            OpenCVCamera(
                type=CameraType.RGB, address=int(app.config.get("camera.RGB", "ID"))
            )
        )
        # NOTE: add more cameras here
    except VideoCaptureOpenError as err:
        print("Could not open camera stream: ", err)

    app.camera_manager = CameraManager(cameras=cameras)
    app.algorithm_manager = AlgorithmManager([ObjectDetection()])

    yield

    # NOTE: add cleanup code here if necessary


# create app and apply lifespan
app = FastAPI(lifespan=lifespan)
# List of allowed origins. In this case, the IP running the frontend app
origins = ["http://127.0.0.1:3000"]
# add middleware to allow cors communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Simple endpoint that provides a debug message.
    """
    return {"message": "Hello World!"}


@app.get("/video/")
async def video_feed(
    request: Request,
    camera_type: CameraType,
    algorithm_type: AlgorithmType,
):
    """
    Endpoint to provide a video feed from a specified camera type, with an optional algorithm applied.

    Args:
        request (Request): The request object, which allows access to the application's state.
        camera_type (CameraType): The type of camera for the video feed.
        algorithm_type (AlgorithmType): The type of algorithm to apply to the video feed.

    Returns:
        StreamingResponse: A live video feed, processed by the specified algorithm (if any),
            delivered as a multipart stream of JPEG images.
    """

    if not camera_type:
        raise TypeError("No camera type was specified!")

    camera = request.app.camera_manager.get_camera_by_type(type=camera_type)

    # retrieve algorith if requested
    algorithm = None
    if algorithm_type is not AlgorithmType.NONE:
        algorithm = request.app.algorithm_manager.get_algorithm_by_type(
            type=algorithm_type
        )

    return StreamingResponse(
        content=get_view(camera, algorithm),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# generator for webcam video chunks
def get_view(camera: AbstractCamera, algorithm: Algorithm = None):
    """
    Generator to continuously capture frames from the camera and optionally apply an algorithm.

    Args:
        camera (Camera): The camera instance used to capture video frames.
        algorithm (Algorithm, optional): The algorithm instance used to process video frames.

    Yields:
        bytes: The binary string of the encoded frame, formatted for multipart response.
    """

    while True:
        try:
            frame = camera.get_frame()

            # apply algorithm if specified by overwriting frame
            if algorithm is not None:
                frame = algorithm(frame)

            binary_string = cv2.imencode(".png", frame)[1].tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + binary_string + b"\r\n"
            )
        except VideoCaptureOpenError:
            print("VideoCaptureOpenError, frame skipped.")
            continue
        except VideoCaptureReadError:
            print("VideoCaptureReadError, frame skipped.")
            continue
