import time
from collections import Counter, deque

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
        self.buffer_size = 10  # Erhöht für stabilere Erkennung
        self.last_prediction = ""

        # Queue für erkannte Buchstaben
        self.letter_queue = deque(maxlen=10)  # Maximal 7 Buchstaben
        self.stable_frames = 0  # Zähler für stabile Frames
        self.required_stable_frames = 10  # Anzahl der Frames für stabile Erkennung
        self.last_added_letter = None

        # Timer für Queue-Reset
        self.last_queue_update = time.time()
        self.queue_timeout = 7

    def add_to_queue(self, letter):
        """Fügt einen Buchstaben zur Queue hinzu, wenn er stabil erkannt wurde."""
        if letter != self.last_added_letter:
            self.stable_frames = 1
            self.last_added_letter = letter
        else:
            self.stable_frames += 1

        if self.stable_frames >= self.required_stable_frames:
            if len(self.letter_queue) == 0 or letter != self.letter_queue[-1]:
                self.letter_queue.append(letter)
                self.stable_frames = 0
                self.last_queue_update = time.time()  # Aktualisiere Timer

    def check_queue_timeout(self):
        """Überprüft, ob die Queue zurückgesetzt werden soll."""
        current_time = time.time()
        if (
            len(self.letter_queue) > 0
            and current_time - self.last_queue_update >= self.queue_timeout
        ):
            self.letter_queue.clear()
            self.last_queue_update = current_time

    def draw_letter_queue(self, frame):
        """Zeichnet die Buchstaben-Queue auf dem Frame."""
        if not self.letter_queue:
            return frame

        # Erstelle String aus Queue mit Bindestrichen
        queue_text = "-".join(letter.upper() for letter in self.letter_queue)

        # Berechne Position und Größe
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        color = (255, 255, 255)

        # Hole Textgröße für Zentrierung
        text_size = cv2.getTextSize(queue_text, font, font_scale, thickness)[0]

        # Berechne Position (zentriert, unten)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 30

        # Zeichne halbtransparenten Hintergrund
        overlay = frame.copy()
        bg_padding = 20
        cv2.rectangle(
            overlay,
            (text_x - bg_padding, text_y - text_size[1] - bg_padding),
            (text_x + text_size[0] + bg_padding, text_y + bg_padding),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Zeichne Text
        cv2.putText(
            frame, queue_text, (text_x, text_y), font, font_scale, color, thickness
        )

        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # Überprüfe Queue-Timeout
        self.check_queue_timeout()

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

        # Mache Vorhersage wenn Hand erkannt wurde
        if results.multi_hand_landmarks:
            # Features extrahieren
            hand_keypoints = np.zeros(21 * 3)

            if results.multi_hand_landmarks:
                # Wenn mehrere Hände erkannt wurden, finde die richtige Hand
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
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
                    current_prediction = most_common[0][0]
                    frequency = most_common[0][1] / len(self.prediction_buffer)

                    # Wenn die Vorhersage stabil ist
                    if frequency > 0.6 and confidence_value > 0.7:
                        self.last_prediction = current_prediction
                        # Füge zur Queue hinzu
                        self.add_to_queue(current_prediction)

                        # Zeige aktuelle Vorhersage an
                        cv2.rectangle(frame, (0, 0), (200, 100), (245, 117, 16), -1)
                        cv2.putText(
                            frame,
                            current_prediction.upper(),
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

        # Zeichne die Buchstaben-Queue
        frame = self.draw_letter_queue(frame)
        return frame
