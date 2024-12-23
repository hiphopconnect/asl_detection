import numpy as np
from backend.algorithms import Algorithm
from backend.custom_types import AlgorithmType


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
