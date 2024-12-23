from abc import ABC, abstractmethod
from threading import Lock, Thread
from typing import Any

import cv2
from custom_types import CameraType
import numpy as np


class VideoCaptureOpenError(Exception):
    """
    Exception raised when a video capture device fails to open.
    """

    pass


class VideoCaptureReadError(Exception):
    """
    Exception raised when a frame cannot be read from the video capture device.
    """

    pass


class AbstractCamera(ABC):
    """
    Abstract camera class for handling a camera.
    Defines abstract methods, all cameras need.

    Attributes:
        type (CameraType): The type of camera specified by an instance of CameraType enum.
    """

    type: CameraType

    @abstractmethod
    def __init__(self, type: CameraType) -> None:
        """
        Initialize a camera instance with the given address and camera type.

        Args:
            type (CameraType): The type of camera specified by an instance of CameraType enum.
        """
        self.type = type

    @abstractmethod
    def get_frame(self) -> np.ndarray[Any]:
        """
        Capture and return a single frame from the camera stream.

        Returns:
            A single frame captured from the camera stream.
        """
        pass

    @abstractmethod
    def __del__(self) -> None:
        """
        Release the camera resource on deletion of the Camera object instance.
        """
        pass


class OpenCVCamera(AbstractCamera):
    """
    Camera class for handling video stream capture from a given source via OpenCV.
    """

    def __init__(self, address: str, type: CameraType) -> None:
        """
        Initialize a camera instance with the given address and camera type.

        Args:
            address (str): The address for the camera stream (URL or device identifier).
            type (CameraType): The type of camera specified by an instance of CameraType enum.

        Raises:
            VideoCaptureOpenError: If the video stream could not be opened with the provided address.
        """
        super().__init__(type)
        self._capture = cv2.VideoCapture(address)
        if not self._capture.isOpened():
            raise VideoCaptureOpenError(
                f"{self.__class__}'s camera stream could not be opened under: {address}"
            )

    def get_frame(self):
        """
        Capture and return a single frame from the camera stream.

        Returns:
            A single frame captured from the camera stream.
            It has the shape (width x height x 3) because it is a RGB image.

        Raises:
            VideoCaptureReadError: If the frame could not be read from the capture.
            VideoCaptureOpenError: If the camera stream is closed when attempting to read a frame.
        """

        if self._capture.isOpened():
            ret, frame = self._capture.read()
            if ret:
                return frame
            raise VideoCaptureReadError(
                f"Could not read frame from {self.__class__} capture!"
            )
        raise VideoCaptureOpenError(f"{self.__class__}'s camera stream is closed!")

    def __del__(self) -> None:
        """
        Release the camera resource on deletion of the Camera object instance.
        """

        self._capture.release()


class CameraManager:
    """
    A manager class to handle multiple camera instances based on their types.

    This class provides methods to add, remove, and retrieve camera instances.

    Attributes:
        _cam (dict[CameraType, Camera]): A private dictionary mapping camera types to camera instances.
    """

    _cam: dict[CameraType, AbstractCamera] = {}

    def __init__(self, cameras: list[AbstractCamera] = []) -> None:
        """
        Initialize the CameraManager with an optional list of camera instances.

        Args:
            cameras (list[Camera], optional): A list of Camera objects to be managed. Defaults to an empty list.
        """

        for cam in cameras:
            self._cam[cam.type] = cam

    def add_camera(self, camera: AbstractCamera) -> None:
        """
        Adds a camera instance to the manager.
        If a camera with the same type already exists, it will be replaced by the new one.

        Args:
            camera (Camera): The Camera object to be added to the manager.
        """
        self._cam[camera.type] = camera

    def remove_camera_by_type(self, type: CameraType) -> None:
        """
        Remove a camera instance from the manager by its type.

        Args:
            type (CameraType): The type of the camera to be removed.
        """

        if type in self._cam:
            self._cam.pop(type)

    def get_camera_by_type(self, type: CameraType) -> AbstractCamera:
        """
        Retrieve a camera instance from the manager by its type.

        Args:
            type (CameraType): The type of the camera to retrieve.

        Returns:
            Camera: The camera instance corresponding to the specified type.

        Raises:
            KeyError: If a camera with the specified type does not exist in the manager.
        """

        if type in self._cam:
            return self._cam[type]
        return TypeError(f"{type} isn't a valid item of this Manager!")
