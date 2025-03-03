import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# MediaPipe Holistics initialisieren
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Pfad zum Dataset
DATASET_PATH = "machine_learning/datasets/asl_now/Dataset"
# Pfad zum Speichern der extrahierten Keypoints
OUTPUT_PATH = "machine_learning/datasets/keypoints"

# Stelle sicher, dass der Ausgabeordner existiert
os.makedirs(OUTPUT_PATH, exist_ok=True)

def extract_keypoints(results):
    """
    Extrahiert alle relevanten Keypoints von MediaPipe Holistics
    """
    # Pose (33 Keypoints mit x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Linke Hand (21 Keypoints mit x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Rechte Hand (21 Keypoints mit x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Gesicht (468 Keypoints mit x, y, z) - optional, könnte viel Speicherplatz benötigen
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Wir verwenden keine Gesichtskeypoints, da diese für Gebärdensprache weniger relevant sind
    # und die Datenmenge erheblich erhöhen würden
    return np.concatenate([pose, lh, rh])

def process_image(image_path, holistic):
    """
    Verarbeitet ein einzelnes Bild und extrahiert Keypoints
    """
    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {image_path}")
        return None
    
    # Farbraum für MediaPipe konvertieren
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Keypoints mit MediaPipe Holistics extrahieren
    results = holistic.process(image_rgb)
    
    # Extrahiere alle relevanten Keypoints
    keypoints = extract_keypoints(results)
    
    return keypoints

def main():
    # Alle Buchstabenordner im Dataset (a-y ohne j)
    alphabet = [chr(ord('a') + i) for i in range(25) if chr(ord('a') + i) != 'j']
    
    # Dataframe für die extrahierten Keypoints
    all_keypoints = []
    
    # MediaPipe Holistics mit hoher Erkennungsgenauigkeit initialisieren
    with mp_holistic.Holistic(
            static_image_mode=True,  # Bilder werden einzeln verarbeitet
            model_complexity=2,     # Höchste Genauigkeit (0, 1 oder 2)
            enable_segmentation=False,
            refine_face_landmarks=False) as holistic:
        
        # Über alle Buchstabenordner iterieren
        for letter in alphabet:
            letter_dir = os.path.join(DATASET_PATH, letter)
            if not os.path.isdir(letter_dir):
                print(f"Ordner für Buchstabe {letter} nicht gefunden: {letter_dir}")
                continue
            
            print(f"Verarbeite Buchstabe: {letter}")
            
            # Alle PNG-Dateien im Ordner finden
            image_files = [f for f in os.listdir(letter_dir) if f.endswith('.png')]
            
            # Über alle Bilder im Ordner iterieren mit Fortschrittsbalken
            for image_file in tqdm(image_files, desc=f"Buchstabe {letter}"):
                image_path = os.path.join(letter_dir, image_file)
                
                # Keypoints aus dem Bild extrahieren
                keypoints = process_image(image_path, holistic)
                
                if keypoints is not None:
                    # Keypoints mit Label und Dateinamen speichern
                    keypoint_data = {
                        'letter': letter,
                        'filename': image_file,
                        'keypoints': keypoints
                    }
                    all_keypoints.append(keypoint_data)
    
    print(f"Insgesamt {len(all_keypoints)} Bilder verarbeitet.")
    
    # Speichern der extrahierten Keypoints
    keypoints_df = pd.DataFrame(all_keypoints)
    
    # CSV-Datei mit Metadaten speichern (ohne die großen Keypoint-Arrays)
    metadata_df = keypoints_df[['letter', 'filename']].copy()
    metadata_df.to_csv(os.path.join(OUTPUT_PATH, 'metadata.csv'), index=False)
    
    # Numpy-Datei mit allen Keypoints speichern
    keypoints_array = np.array([data['keypoints'] for data in all_keypoints])
    labels = np.array([ord(data['letter']) - ord('a') for data in all_keypoints])
    
    np.savez(os.path.join(OUTPUT_PATH, 'asl_keypoints.npz'),
             keypoints=keypoints_array,
             labels=labels)
    
    print(f"Keypoints wurden gespeichert unter: {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 