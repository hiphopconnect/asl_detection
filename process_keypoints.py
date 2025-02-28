import json
import numpy as np

def load_keypoint_data(file_path):
    """
    Lädt die Keypoint-Daten aus einer JSON-Datei.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_frames(data):
    """
    Verarbeitet die Keypoint-Daten frame für frame.
    Konvertiert die Daten in das Format (N, T, V, C), wobei:
    N = Batch size (1 in diesem Fall)
    T = Anzahl der Frames
    V = Anzahl der Keypoints
    C = Dimensionen pro Keypoint (x, y, z, confidence für pose)
    """
    num_frames = len(data['keypoints'])
    
    # Initialisiere Arrays für Pose und Face
    pose_data = []
    face_data = []
    
    for frame in data['keypoints']:
        # Extrahiere die relevanten Keypoints
        pose = np.array(frame['pose'])  # Shape: (V, 4) - x, y, z, confidence
        face = np.array(frame['face'])  # Shape: (V, 3) - x, y, z
        
        pose_data.append(pose)
        face_data.append(face)
    
    # Konvertiere zu numpy arrays
    pose_data = np.array(pose_data)  # Shape: (T, V, 4)
    face_data = np.array(face_data)  # Shape: (T, V, 3)
    
    # Füge Batch-Dimension hinzu
    pose_data = np.expand_dims(pose_data, axis=0)  # Shape: (1, T, V, 4)
    face_data = np.expand_dims(face_data, axis=0)  # Shape: (1, T, V, 3)
    
    return {
        'pose': pose_data,
        'face': face_data
    }

def main():
    # Laden der Daten
    data = load_keypoint_data('message.txt')
    
    # Grundlegende Informationen ausgeben
    print(f"Gloss: {data['gloss']}")
    print(f"Video ID: {data['video_id']}")
    print(f"Anzahl Frames: {len(data['keypoints'])}")
    
    # Verarbeite die Frames
    processed_data = process_frames(data)
    
    # Gebe die Shapes der verarbeiteten Daten aus
    print("\nVerarbeitete Daten Shapes:")
    print(f"Pose Shape: {processed_data['pose'].shape}")
    print(f"Face Shape: {processed_data['face'].shape}")
    
    # Beispiel für die ersten Pose-Keypoints des ersten Frames
    print("\nErste 3 Pose-Keypoints (Frame 0):")
    print(processed_data['pose'][0, 0, :3])
    
if __name__ == "__main__":
    main() 