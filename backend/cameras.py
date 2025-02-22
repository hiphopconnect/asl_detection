from abc import ABC, abstractmethod
import cv2
from backend.custom_types import CameraType
import numpy as np
from typing import Any

class VideoCaptureOpenError(Exception):
    """Exception raised when a video capture device fails to open."""
    pass

class VideoCaptureReadError(Exception):
    """Exception raised when a frame cannot be read from the video capture device."""
    pass

class AbstractCamera(ABC):
    """
    Abstract camera class for handling a camera.
    Defines abstract methods, die alle Kameras implementieren müssen.

    Attributes:
        type (CameraType): Der Kameratyp.
    """
    type: CameraType

    @abstractmethod
    def __init__(self, type: CameraType) -> None:
        self.type = type

    @abstractmethod
    def get_frame(self) -> np.ndarray[Any]:
        """Liefert einen einzelnen Frame als numpy-Array."""
        pass

    @abstractmethod
    def __del__(self) -> None:
        """Gibt die Kameraressource frei."""
        pass

class OpenCVCamera(AbstractCamera):
    """
    Camera class that opens the video stream using OpenCV.
Additionally, it stores the used index (if the address is numeric).
    """

    def __init__(self, address: str, type: CameraType) -> None:
        super().__init__(type)
        try:
            self.index = int(address)  # If address is numeric
        except ValueError:
            self.index = None

        if self.index is not None:
            self._capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        else:
            self._capture = cv2.VideoCapture(address)

        if not self._capture.isOpened():
            raise VideoCaptureOpenError(
                f"{self.__class__.__name__} could not be opened at {address}."
            )

    def get_frame(self):
        if self._capture.isOpened():
            ret, frame = self._capture.read()
            if ret:
                return frame
            raise VideoCaptureReadError(
                f"Frame could not be read from {self.__class__.__name__}!"
            )
        raise VideoCaptureOpenError(f"{self.__class__.__name__} is closed!")

    def __del__(self) -> None:
        # With persistent management, we want to control resource release through the manager.
        if hasattr(self, '_capture') and self._capture is not None:
            self._capture.release()


class CameraManager:
    """
    Manages persistent camera instances, indexed by camera index.
    """

    def __init__(self, cameras: list[AbstractCamera] = []) -> None:
        self._cameras: dict[int, AbstractCamera] = {}
        for cam in cameras:
            if cam.index is not None:
                self._cameras[cam.index] = cam

    def add_camera(self, camera: AbstractCamera) -> None:
        if camera.index is not None:
            self._cameras[camera.index] = camera

    def remove_camera_by_index(self, index: int) -> None:
        if index in self._cameras:
            self._cameras.pop(index)

    def get_camera_by_index(self, index: int) -> AbstractCamera:
        if index in self._cameras:
            return self._cameras[index]
        raise KeyError(f"Camera with index {index} does not exist.")
