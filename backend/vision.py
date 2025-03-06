from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from backend.algorithms import Algorithm
from backend.custom_types import AlgorithmType, ModelName


class HandSignNet(nn.Module):
    def __init__(self, num_classes=24):
        super(HandSignNet, self).__init__()

        # Feature Extraction Blocks
        self.features = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
        )

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ASLFingerspelling(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.type = AlgorithmType.DETECTION
        self.name = ModelName.ASLFINGERSPELLING
        # MediaPipe Hands initialisieren (wie beim Training)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self.model_path = "/workspaces/asl_detection/machine_learning/models/asl_fingerspelling/best_model.pth"

        # Modell laden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Nutze Device: {self.device}")

        self.model = HandSignNet().to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        # Buchstaben-Mapping (alle Buchstaben außer j und z)
        self.letters = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
        ]

        # Puffer für stabilere Vorhersagen
        self.prediction_buffer = []
        self.buffer_size = 5
        self.last_prediction = ""

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Zeichne Handpunkte
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=4
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )

        # Mache Vorhersage wenn Hand erkannt wurde (exakt wie im Training)
        if results.multi_hand_landmarks:
            # Features extrahieren
            hand_keypoints = np.zeros(21 * 3)

            if results.multi_hand_landmarks:
                # Wenn mehrere Hände erkannt wurden, finde die richtige Hand
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Die Hand-Klassifikation ist aus Sicht der Kamera
                    handedness = (
                        results.multi_handedness[hand_idx].classification[0].label
                    )
                    if (
                        handedness == "Right"
                    ):  # Wir suchen die rechte Hand aus Sicht der Kamera
                        hand_keypoints = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        ).flatten()
                        break
                # Falls keine rechte Hand gefunden wurde, nimm die erste erkannte Hand
                if np.all(hand_keypoints == 0) and results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_keypoints = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    ).flatten()

            landmarks = hand_keypoints

            # Modellvorhersage
            with torch.no_grad():
                landmarks_tensor = (
                    torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
                )
                outputs = self.model(landmarks_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)

                # Hole Buchstaben und Konfidenz
                predicted_letter = self.letters[prediction.item()]
                confidence_value = confidence.item()

                # Vorhersage zum Puffer hinzufügen
                self.prediction_buffer.append(predicted_letter)

                # Puffer-Größe begrenzen
                if len(self.prediction_buffer) > self.buffer_size:
                    self.prediction_buffer.pop(0)

                # Häufigste Vorhersage auswählen
                if self.prediction_buffer:
                    most_common = Counter(self.prediction_buffer).most_common(1)
                    self.last_prediction = most_common[0][0]
                    frequency = most_common[0][1] / len(self.prediction_buffer)

                    # Zeige Vorhersage an
                    if (
                        frequency > 0.6
                    ):  # Nur anzeigen wenn mehr als 60% der Vorhersagen übereinstimmen
                        cv2.rectangle(frame, (0, 0), (200, 100), (245, 117, 16), -1)
                        cv2.putText(
                            frame,
                            self.last_prediction.upper(),
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Konf: {confidence_value:.2f}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
        return frame
