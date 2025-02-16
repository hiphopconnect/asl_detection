import numpy as np
import cv2
import mediapipe as mp
from backend.algorithms import Algorithm
from backend.custom_types import AlgorithmType, ModelName

class MediaPipeHolistics(Algorithm):
    """
    Attributes:
        type (AlgorithmType): The type of the algorithm, set to POSEDETECTION.
        name (ModelName): The name of the model, set to MEDIAPIPE_HOLISTICS.
        model: The MediaPipe Holistic model.
        mp_drawing: MediaPipe drawing utilities.
        mp_drawing_styles: MediaPipe drawing styles.
    Methods:
        __call__(frame: np.ndarray) -> np.ndarray:
            Processes the input frame to detect and draw face, pose, and hand landmarks.
            Args:
                frame (np.ndarray): The input image frame in BGR format.
            Returns:
                np.ndarray: The processed image frame with landmarks drawn.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type = AlgorithmType.POSEDETECTION
        self.name = ModelName.MEDIAPIPE_HOLISTICS
        self.model = mp.solutions.holistic.Holistic()
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

