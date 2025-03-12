import json
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from backend.algorithms import Algorithm
from backend.custom_types import ModelName


# Base Class for Fingerspelling Model, just like in training
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


# Base Class for WordDetection Model, just like in training
class SignLanguageModel(nn.Module):
    def __init__(
        self, num_classes, input_size=126, hidden_size=64, num_layers=1, dropout=0.3
    ):
        super(SignLanguageModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,  # Anpassbar aus JSON
            batch_first=True,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Tanh())

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm wird wieder verwendet
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Classification
        out = self.fc(context)
        return out


class ASLFingerspelling(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = ModelName.ASLFINGERSPELLING
        # Initialize MediaPipe Hands (as done during training)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self.model_path = "/workspaces/asl_detection/machine_learning/models/asl_fingerspelling/best_model.pth"

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HandSignNet().to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        # Letter mapping (all letters except j and z)
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

        # Buffer for more stable predictions
        self.prediction_buffer = []
        self.buffer_size = 10  # Increased for more stable recognition
        self.last_prediction = ""

        # Queue for recognized letters
        self.letter_queue = deque(maxlen=10)  # Max 7 letters
        self.stable_frames = 0  # Counter for stable frames
        self.required_stable_frames = 10  # Number of frames for stable recognition
        self.last_added_letter = None

        # Timer for queue reset
        self.last_queue_update = time.time()
        self.queue_timeout = 7

    def add_to_queue(self, letter):
        """Adds a letter to the queue if it was stably recognized."""
        if letter != self.last_added_letter:
            self.stable_frames = 1
            self.last_added_letter = letter
        else:
            self.stable_frames += 1

        if self.stable_frames >= self.required_stable_frames:
            if len(self.letter_queue) == 0 or letter != self.letter_queue[-1]:
                self.letter_queue.append(letter)
                self.stable_frames = 0
                self.last_queue_update = time.time()  # Update timer

    def check_queue_timeout(self):
        """Checks if the queue should be reset."""
        current_time = time.time()
        if (
            len(self.letter_queue) > 0
            and current_time - self.last_queue_update >= self.queue_timeout
        ):
            self.letter_queue.clear()
            self.last_queue_update = current_time

    def draw_letter_queue(self, frame):
        """Draws the letter queue on the frame."""
        if not self.letter_queue:
            return frame

        # Create string from queue with dashes
        queue_text = "-".join(letter.upper() for letter in self.letter_queue)

        # Calculate position and size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        color = (255, 255, 255)

        # Get text size for centering
        text_size = cv2.getTextSize(queue_text, font, font_scale, thickness)[0]

        # Calculate position (centered, bottom)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 30

        # Draw semi-transparent background
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

        # Draw text
        cv2.putText(
            frame, queue_text, (text_x, text_y), font, font_scale, color, thickness
        )

        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # Check queue timeout
        self.check_queue_timeout()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Draw hand landmarks
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

        # Make prediction if hand is detected
        if results.multi_hand_landmarks:
            # Extract features
            hand_keypoints = np.zeros(21 * 3)

            if results.multi_hand_landmarks:
                # If multiple hands are detected, find the correct hand
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = (
                        results.multi_handedness[hand_idx].classification[0].label
                    )
                    if (
                        handedness == "Right"
                    ):  # We select the right hand from the camera's perspective
                        hand_keypoints = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        ).flatten()
                        break
                # If no right hand is found, take the first detected hand
                if np.all(hand_keypoints == 0) and results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_keypoints = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    ).flatten()

            landmarks = hand_keypoints

            # Model prediction
            with torch.no_grad():
                landmarks_tensor = (
                    torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
                )
                outputs = self.model(landmarks_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)

                # Get letter and confidence
                predicted_letter = self.letters[prediction.item()]
                confidence_value = confidence.item()

                # Add prediction to buffer
                self.prediction_buffer.append(predicted_letter)

                # Limit buffer size
                if len(self.prediction_buffer) > self.buffer_size:
                    self.prediction_buffer.pop(0)

                # Choose the most frequent prediction
                if self.prediction_buffer:
                    most_common = Counter(self.prediction_buffer).most_common(1)
                    current_prediction = most_common[0][0]
                    frequency = most_common[0][1] / len(self.prediction_buffer)

                    # If prediction is stable
                    if frequency > 0.6 and confidence_value > 0.7:
                        self.last_prediction = current_prediction
                        # Add to queue
                        self.add_to_queue(current_prediction)

                        # Display current prediction
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
                            f"Conf: {confidence_value:.2f}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )

        # Draw the letter queue
        frame = self.draw_letter_queue(frame)
        return frame


class ASLWordDetection(Algorithm):
    def __init__(self) -> None:
        super().__init__()

        def read_actions_from_json(
            json_path="/workspaces/asl_detection/machine_learning/datasets/asl_word_detection/actions_list.json",
        ):
            """Liest die Actions aus der JSON-Datei."""
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    actions = data.get("actions", [])
                    if actions:
                        return np.array(actions)
            except FileNotFoundError:
                print(f"Keine actions_list.json gefunden unter {json_path}")
            except json.JSONDecodeError:
                print(f"Fehler beim Lesen der JSON-Datei {json_path}")
            except Exception as e:
                print(f"Unerwarteter Fehler beim Lesen der JSON: {e}")
            return None

        self.name = ModelName.ASLWORDDETECTION
        # Initialize MediaPipe Hands (as done during training)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.model_path = "/workspaces/asl_detection/machine_learning/models/asl_word_detection/best_model.pth"

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        json_file = "/workspaces/asl_detection/machine_learning/datasets/asl_word_detection/actions_list.json"
        with open(json_file, "r") as f:
            data = json.load(f)
            model_info = data.get("model_info", {})
            hidden_size = model_info.get("hidden_size", 64)
            num_layers = model_info.get("num_layers", 2)

        self.actions = read_actions_from_json()
        input_size = 126
        self.model = SignLanguageModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=len(self.actions),
            num_layers=num_layers,
        )

        self.sequence = []

        # Load Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "/workspaces/asl_detection/machine_learning/models/asl_word_detection/best_model.pth"
        checkpoint = torch.load(model_path, map_location=device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.holistic.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:  # Show but don't use
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )

        # Extract ONLY hand keypoints for prediction
        lh = np.zeros(21 * 3)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                lh[i * 3] = lm.x
                lh[i * 3 + 1] = lm.y
                lh[i * 3 + 2] = lm.z

        rh = np.zeros(21 * 3)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                rh[i * 3] = lm.x
                rh[i * 3 + 1] = lm.y
                rh[i * 3 + 2] = lm.z

        keypoints = np.concatenate([lh, rh])  # Only hands!
        self.sequence.append(keypoints)

        sequence = self.sequence[-102:]  # Keep all frames just like in training
        top_actions = []

        if len(sequence) == 102:
            # Make prediction
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0)
                outputs = self.model(sequence_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

            action_probs = [
                (action, prob.item()) for action, prob in zip(self.actions, probs[0])
            ]
            action_probs.sort(key=lambda x: x[1], reverse=True)

            top_k = 5  # Number of the most possible predicted word
            top_actions = action_probs[:top_k]

        cv2.putText(
            frame,
            "TOP 5 PREDICTIONS:",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for i, (action, prob_value) in enumerate(top_actions):
            # Farbcodierung nach Konfidenz
            if prob_value > 0.3:  # Grüner Schwellenwert
                color = (0, 255, 0)  # Grün für hohe Konfidenz
            elif prob_value > 0.1:  # Gelber Schwellenwert
                color = (0, 255, 255)  # Gelb für mittlere Konfidenz
            else:
                color = (0, 0, 255)  # Rot für niedrige Konfidenz

            # Prozentbalken für visuelle Darstellung
            bar_width = int(prob_value * 200)  # Max 200 Pixel breit
            cv2.rectangle(
                frame, (140, 40 + i * 25), (140 + bar_width, 55 + i * 25), color, -1
            )

            # rank and word
            cv2.putText(
                frame,
                f"{i+1}. {action}:",
                (10, 50 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Confidence-value
            cv2.putText(
                frame,
                f"{prob_value:.2f}",
                (145 + bar_width, 50 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
