import numpy as np
import cv2
import mediapipe as mp
from backend.algorithms import Algorithm
from backend.custom_types import AlgorithmType

class MediaPipeHolistics(Algorithm):
    """
    Implementation of a MediaPipe Holistics algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type = AlgorithmType.POSEDETECTION
        self.name = ModelName.MediaPipeHolistics
        self.model = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles


    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # INSERT ALGORITHM HERE (draw inference results on frame)
        # Example color flip
        return np.flip(frame, axis=2)


class ObjectDetection(Algorithm):
    """
    Implementation of an object detection algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type = AlgorithmType.DETECTION

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # INSERT ALGORITHM HERE (draw inference results on frame)
        # Example color flip
        return np.flip(frame, axis=2)
