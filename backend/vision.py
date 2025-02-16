import numpy as np
import cv2
import mediapipe as mp
from backend.algorithms import Algorithm
from backend.custom_types import AlgorithmType, ModelName

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
        with self.model(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5) as holistics:

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistics.process(frame)

            frame.flags.writable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                frame.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    self.model.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
            
            if results.pose_landmarks:
                frame.draw_landmarks(
                    frame,
                    results.pose_landsmarks,
                    self.model.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.model.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.model.HAND_CONNECTIONS)
            
        return frame


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
