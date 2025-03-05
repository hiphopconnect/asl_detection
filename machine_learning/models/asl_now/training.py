import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import random
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, precision_score, recall_score
import itertools

# Konstanten
RANDOM_SEED = 42
BATCH_SIZE = 128  # Vom Benutzer geändert
EPOCHS = 100  # Erhöht für mehr Trainingszeit
LEARNING_RATE = 0.0007  # Vom Benutzer geändert
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/home/geiger/asl_detection/machine_learning/models/asl_now/best_model.pth'  # Pfad zum bestehenden Modell
EARLY_STOPPING_PATIENCE = 30  # Geduld erhöht, um längeres Training zu ermöglichen
# Problemklassen mit spezifischen Augmentationsfaktoren
PROBLEM_CLASSES = {
    'k': 4.0,    # Extremes Problem, wird nie richtig erkannt (F1-Score 0.000)
    'y': 3.0,    # Hohe Precision aber sehr niedriger Recall, wird oft als 'i' erkannt
    'p': 2.5,    # Wird oft mit 'q' verwechselt
    's': 2.5,    # Wird oft mit 'n' verwechselt
    'r': 2.0,    # Niedrige Precision, wird mit 'u' verwechselt
    'q': 2.0,    # Gegenpart zu 'p', ebenfalls verstärken
    'i': 1.8,    # Wird mit 'y' verwechselt
    'u': 1.8     # Wird mit 'k' und 'r' verwechselt
}

# Dataset-Klasse
class HandSignDataset(Dataset):
    def __init__(self, keypoints, labels, augment=False):
        self.keypoints = torch.FloatTensor(keypoints)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        keypoints = self.keypoints[idx]
        
        # Einfache Datenaugmentation (nur beim Training)
        if self.augment and random.random() > 0.5:
            # Leichtes zufälliges Rauschen hinzufügen (max 1.5%)
            noise = torch.randn_like(keypoints) * 0.015
            keypoints = keypoints + noise
            
        return keypoints, self.labels[idx]

# Modell-Definition
class HandSignNet(nn.Module):
    def __init__(self, num_classes=24):
        super(HandSignNet, self).__init__()
        
        # Feature Extraction Blocks
        self.features = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.35),  # Leicht erhöht von 0.3
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.35),  # Leicht erhöht von 0.3
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.35)  # Leicht erhöht von 0.3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(cm, classes, normalize=False, title='Konfusionsmatrix', cmap=plt.cm.Blues, figsize=(10, 8)):
    """
    Zeichnet eine Konfusionsmatrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    
    # Speichere die Konfusionsmatrix
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Analysiere die Konfusionsmatrix
    worst_classes_idx, top_confusion_pairs = analyze_confusion_matrix(cm, classes)
    
    return worst_classes_idx, top_confusion_pairs

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def get_class_weights(y_train, alphabet):
    """
    Berechnet Klassengewichte für unbalancierte Datasets.
    Problemklassen werden zusätzlich stärker gewichtet.
    
    Args:
        y_train: Label-Array
        alphabet: Liste der Klassennamen
        
    Returns:
        Dictionary mit Klassengewichten
    """
    # Berechne Grundgewichte basierend auf Klassenverteilung
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    
    # Erstelle Dictionary mit Klassenindizes als Schlüssel
    class_weights = {i: w for i, w in zip(unique_classes, weights)}
    
    # Dictionary zur Zuordnung von Buchstaben zu Indizes
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    # Problemspezifische zusätzliche Gewichtungsfaktoren
    problem_weights = {
        'k': 3.0,  # Extremes Problem
        'y': 2.5,  # Sehr niedriger Recall
        'p': 2.0,  # Wird oft mit 'q' verwechselt
        's': 2.0,  # Wird oft mit 'n' verwechselt
        'r': 1.8,  # Precision-Problem
        'q': 1.5,  # Gegenpart zu 'p'
        'i': 1.3,  # Wird mit 'y' verwechselt
        'u': 1.3   # Wird mit 'k' und 'r' verwechselt
    }
    
    # Verstärke Gewichte für Problemklassen basierend auf spezifischen Faktoren
    for char, extra_factor in problem_weights.items():
        if char in char_to_idx:
            idx = char_to_idx[char]
            if idx in class_weights:
                old_weight = class_weights[idx]
                class_weights[idx] *= extra_factor
                print(f"Zusätzliches Gewicht für Problemklasse '{char}': {old_weight:.2f} -> {class_weights[idx]:.2f} (Faktor {extra_factor})")
    
    return class_weights

def analyze_confusion_matrix(cm, classes):
    """
    Analysiert die Konfusionsmatrix, um häufige Probleme zu identifizieren
    """
    print("\n=== Analyse der Konfusionsmatrix ===")
    
    # Normalisierte Konfusionsmatrix für Vergleiche
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Berechne Precision, Recall und F1-Score pro Klasse
    precision = np.zeros(len(classes))
    recall = np.zeros(len(classes))
    f1_score = np.zeros(len(classes))
    
    for i in range(len(classes)):
        # True Positives: Diagonalelement
        tp = cm[i, i]
        # False Positives: Summe der Spalte minus True Positives
        fp = np.sum(cm[:, i]) - tp
        # False Negatives: Summe der Zeile minus True Positives
        fn = np.sum(cm[i, :]) - tp
        
        # Precision: TP / (TP + FP)
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Recall: TP / (TP + FN)
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Sortiere nach F1-Score (aufsteigend)
    sorted_indices = np.argsort(f1_score)
    worst_classes_idx = sorted_indices[:5]  # Die 5 schlechtesten Klassen
    
    print("\nDie 5 am schlechtesten erkannten Buchstaben (nach F1-Score):")
    print("Buchstabe | Precision | Recall  | F1-Score | Hauptprobleme")
    print("-" * 75)
    
    for idx in worst_classes_idx:
        class_name = classes[idx]
        # Finde die häufigsten Verwechslungen für diese Klasse
        confusion_row = [(classes[j], cm[idx, j]) for j in range(len(classes)) if j != idx and cm[idx, j] > 0]
        confusion_row.sort(key=lambda x: x[1], reverse=True)
        
        # Nehme die Top 2 Verwechslungen, wenn verfügbar
        top_confusions = confusion_row[:min(2, len(confusion_row))]
        confusion_text = ", ".join([f"verwechselt mit '{c}' ({n} mal)" for c, n in top_confusions])
        
        # Etwas genauere Diagnose
        diagnosis = ""
        if recall[idx] < 0.3:
            if precision[idx] > 0.7:
                diagnosis = "Extrem niedrige Erkennungsrate, aber wenn erkannt, dann korrekt"
            else:
                diagnosis = "Wird kaum erkannt und häufig verwechselt"
        elif precision[idx] < 0.3:
            diagnosis = "Sehr niedrige Genauigkeit, erzeugt viele Fehlalarme"
        
        print(f"{class_name.ljust(8)} | {precision[idx]:.3f}   | {recall[idx]:.3f}  | {f1_score[idx]:.3f}   | {confusion_text}")
        if diagnosis:
            print(f"DIAGNOSE: {diagnosis}")
    
    # Finde häufig verwechselte Buchstabenpaare
    print("\nHäufig verwechselte Buchstabenpaare:")
    
    confusion_pairs = []
    for i in range(len(classes)):
        row_sum = np.sum(cm[i, :])
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                # Normalisierter Wert: wie oft wurde i als j falsch klassifiziert
                norm_value = cm[i, j] / row_sum if row_sum > 0 else 0
                confusion_pairs.append((classes[i], classes[j], cm[i, j], norm_value))
    
    # Sortiere nach absoluter Anzahl der Verwechslungen
    top_confusion_pairs = sorted(confusion_pairs, key=lambda x: x[2], reverse=True)[:10]
    
    print("\nBuchstabe | Verwechselt mit | Anzahl | % der Klasse")
    print("-" * 50)
    for true_class, pred_class, count, norm_value in top_confusion_pairs:
        print(f"{true_class.ljust(8)} | {pred_class.ljust(14)} | {count:5d} | {norm_value*100:5.1f}%")
    
    return worst_classes_idx, top_confusion_pairs

def augment_minority_classes(X_train, y_train, alphabet, augmentation_factor=1.5, problem_classes=None):
    """
    Erhöht die Anzahl der Samples für unterrepräsentierte Klassen durch Datenaugmentation.
    
    Args:
        X_train: Feature-Array
        y_train: Label-Array
        alphabet: Liste der Klassennamen
        augmentation_factor: Genereller Faktor für alle Klassen
        problem_classes: Dict mit {Buchstabe: spez. Faktor} für Problemklassen
    
    Returns:
        Erweiterte X_train und y_train Arrays
    """
    # Konvertiere zu NumPy-Arrays, falls es Torch-Tensoren sind
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    
    # Zähle die Samples pro Klasse
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print("Klassenverteilung vor der Augmentation:")
    for i, label in enumerate(unique_labels):
        if label < len(alphabet):
            class_name = alphabet[label]
            print(f"Klasse {class_name}: {counts[i]} Samples")
    
    # Erstelle ein Dictionary zur Zuordnung von Buchstaben zu Indizes
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    # Bestimme die Zielanzahl von Samples pro Klasse basierend auf Problemklassen
    max_samples = max(counts)
    target_samples = {}
    
    for label, count in zip(unique_labels, counts):
        if label >= len(alphabet):
            continue
            
        class_name = alphabet[label]
        # Standard-Augmentationsfaktor
        factor = augmentation_factor
        
        # Erhöhter Faktor für Problemklassen
        if problem_classes and class_name in problem_classes:
            factor = problem_classes[class_name]
            print(f"Verwende erhöhten Augmentationsfaktor {factor} für Problemklasse '{class_name}'")
        
        # Zielanzahl: Erhöhe um den Faktor, aber maximal bis zur größten Klassenanzahl
        target_samples[label] = min(int(count * factor), max_samples)
    
    augmented_X = list(X_train)
    augmented_y = list(y_train)
    
    for label in unique_labels:
        if label >= len(alphabet):
            continue
            
        # Indizes der aktuellen Klasse finden
        indices = np.where(y_train == label)[0]
        current_count = len(indices)
        needed_extra = target_samples[label] - current_count
        
        if needed_extra <= 0:
            continue
        
        class_name = alphabet[label]
        print(f"Augmentiere Klasse {class_name} um {needed_extra} zusätzliche Samples")
        
        # Zufällige Samples auswählen (mit Zurücklegen, wenn nötig)
        selected_indices = np.random.choice(indices, size=needed_extra, replace=True)
        
        # Stärkere Augmentation für Problemklassen
        is_problem_class = problem_classes and class_name in problem_classes
        
        # Augmentation durchführen
        for idx in selected_indices:
            new_sample = X_train[idx].copy()
            
            # Basisrauschen für alle Klassen
            noise_level = 0.02
            
            # Stärkeres Rauschen für Problemklassen
            if is_problem_class:
                noise_level = 0.03  # 50% mehr Rauschen
                
            # 1. Zufälliges Rauschen
            noise = np.random.randn(*new_sample.shape) * noise_level
            new_sample += noise
            
            # 2. Spezifische Augmentation für bestimmte Buchstaben
            if class_name == 'k':
                # 'k' hat ein extremes Erkennungsproblem - deutlich verstärkte Augmentation
                # Verstärke Zeigefinger und kleinen Finger (typisch für 'k')
                for i in range(12, 24):  # Zeigefinger
                    if i % 3 == 0:  # x, y Koordinaten (nicht z)
                        new_sample[i] *= random.uniform(1.10, 1.25)  # Stärkere Verstärkung
                
                for i in range(48, 60):  # Kleiner Finger
                    if i % 3 == 0:  # x, y Koordinaten
                        new_sample[i] *= random.uniform(1.10, 1.25)
                        
                # Extra Variationen für 'k' hinzufügen (mit höherer Wahrscheinlichkeit)
                if random.random() > 0.3:  # Erhöhte Wahrscheinlichkeit für spezielle Augmentation
                    # Stärkere Winkelung zwischen Fingern
                    angle_adjust = random.uniform(0.08, 0.15)  # Stärkere Anpassung
                    # Mittelfinger deutlicher nach innen bewegen
                    for i in range(24, 36):  # Mittelfinger
                        if i % 3 == 0:  # x-Koordinaten
                            new_sample[i] -= angle_adjust
                    # Zeigefinger stärker nach außen
                    for i in range(12, 24):  # Zeigefinger
                        if i % 3 == 0:  # x-Koordinaten
                            new_sample[i] += angle_adjust * 0.7
                    
                    # Ringfinger leicht anpassen für bessere Unterscheidung zu 'r' und 'u'
                    for i in range(36, 48):  # Ringfinger
                        if i % 3 == 0:  # x-Koordinaten
                            new_sample[i] += random.uniform(0.05, 0.12)
            
            elif class_name == 'y':
                # 'y' wird oft als 'i' erkannt - verstärke die Unterschiede
                # Verstärke den Daumen (Hauptunterschied zu 'i')
                for i in range(0, 12):  # Daumen
                    new_sample[i] *= random.uniform(1.08, 1.20)
                
                # Handgelenksrotation anpassen
                rotation = random.uniform(0.05, 0.10)
                for i in range(0, 63, 3):  # Alle x-Koordinaten
                    new_sample[i] += rotation
            
            elif class_name == 'r':
                # 'r' hat ein Problem mit der Precision - mehr Variation, um Überlappungen zu reduzieren
                # Verändere die Kreuzung von Zeige- und Mittelfinger
                for i in range(15, 30):  # Angenommene Indizes für Finger-Kreuzung
                    new_sample[i] += random.uniform(-0.04, 0.06)  # Größere Variation
                
                # Zusätzliche Unterscheidung zu 'k' und 'u'
                if random.random() > 0.5:
                    # Verbessere die charakteristische Finger-Kreuzung
                    for i in range(12, 36):  # Zeige- und Mittelfinger
                        if i % 3 == 1:  # y-Koordinaten
                            new_sample[i] += random.uniform(-0.05, 0.05)
            
            elif class_name == 's':
                # 's' wird oft mit 'n' verwechselt
                # Verstärke Daumen (Hauptunterschied zu 'n')
                for i in range(0, 12):  # Daumen
                    new_sample[i] *= random.uniform(1.05, 1.15)
                
                # Charakteristische Faustform verstärken
                for i in range(12, 63):  # Alle Finger außer Daumen
                    if i % 3 == 0:  # x-Koordinaten
                        new_sample[i] *= random.uniform(0.92, 0.98)  # Leicht zusammenziehen
            
            elif class_name == 'p' or class_name == 'q':
                # 'p' und 'q' können ähnlich aussehen - verstärke die Unterschiede
                # Handgelenksrotation stärker anpassen
                wrist_adj = random.uniform(-0.07, 0.07)
                for i in range(0, 9):  # Handgelenkbereich
                    new_sample[i] += wrist_adj
                
                # Spezifische Anpassung je nach Buchstabe
                if class_name == 'p':
                    # Verstärke die charakteristischen Merkmale von 'p'
                    for i in range(36, 48):  # Ringfinger
                        new_sample[i] *= random.uniform(1.05, 1.15)
                else:  # q
                    # Verstärke die charakteristischen Merkmale von 'q'
                    for i in range(12, 24):  # Zeigefinger
                        new_sample[i] *= random.uniform(1.05, 1.15)
            
            elif class_name == 'i':
                # 'i' wird mit 'y' verwechselt - verstärke die Unterschiede
                # Bei 'i' sollte der kleine Finger gestreckt sein
                for i in range(48, 60):  # Kleiner Finger
                    if i % 3 == 1:  # y-Koordinaten
                        new_sample[i] *= random.uniform(1.05, 1.15)
            
            augmented_X.append(new_sample)
            augmented_y.append(label)
    
    # Konvertiere zurück zu Arrays
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    # Zeige die neue Verteilung
    unique_labels, counts = np.unique(augmented_y, return_counts=True)
    print("Klassenverteilung nach der Augmentation:")
    for i, label in enumerate(unique_labels):
        if label < len(alphabet):
            class_name = alphabet[label]
            print(f"Klasse {class_name}: {counts[i]} Samples")
    
    return augmented_X, augmented_y

def plot_both_confusion_matrices(y_true, y_pred, classes, figsize=(18, 16)):
    """
    Erstellt eine einzelne, gut lesbare Konfusionsmatrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Wir verwenden Seaborn für eine besser lesbare Darstellung
    plt.figure(figsize=figsize)
    
    # Absolute Werte in der Hauptmatrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Erstelle eine schöne Heatmap mit Seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=classes, yticklabels=classes,
               linewidths=.5, square=True)
    
    plt.title('Konfusionsmatrix (absolute Werte)', fontsize=14)
    plt.ylabel('Tatsächliche Klasse', fontsize=12)
    plt.xlabel('Vorhergesagte Klasse', fontsize=12)
    
    # Diagonale Elemente hervorheben (korrekte Vorhersagen)
    for i in range(len(classes)):
        # Markiere die Diagonale mit einem farbigen Rand
        plt.plot([i-.5, i+.5], [i-.5, i-.5], '-', color='green', linewidth=2)
        plt.plot([i-.5, i+.5], [i+.5, i+.5], '-', color='green', linewidth=2)
        plt.plot([i-.5, i-.5], [i-.5, i+.5], '-', color='green', linewidth=2)
        plt.plot([i+.5, i+.5], [i-.5, i+.5], '-', color='green', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    # Analysiere die Konfusionsmatrix
    worst_classes_idx, top_confusion_pairs = analyze_confusion_matrix(cm, classes)
    
    return cm, worst_classes_idx, top_confusion_pairs

def main(load_model=True):
    # Setze Seeds für Reproduzierbarkeit
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Alphabet-Definition
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    
    # Lade Daten für Training
    print("Lade Trainingsdaten...")
    train_data = np.load('/home/geiger/asl_detection/machine_learning/datasets/asl_now/Mix_Keypoints/asl_keypoints.npz')
    X_train = train_data['keypoints']
    y_train = train_data['labels']
    
    # Lade separate Daten für Validierung
    print("Lade separate Validierungsdaten...")
    try:
        val_data = np.load('/home/geiger/asl_detection/machine_learning/datasets/asl_now/Keypoints_1/asl_keypoints.npz')
        X_val = val_data['keypoints']
        y_val = val_data['labels']
        print(f"Validierungsdatensatz erfolgreich geladen: {len(X_val)} Samples")
    except Exception as e:
        print(f"Fehler beim Laden der separaten Validierungsdaten: {e}")
        print("Verwende stattdessen einen Teil der Trainingsdaten für die Validierung...")
        # Fallback: Wenn die separate Datei nicht geladen werden kann, verwende train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
        )
    
    # Analysiere Klassenverteilung
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    val_unique, val_counts = np.unique(y_val, return_counts=True)
    
    print("\nKlassenverteilung im Trainingsdatensatz:")
    for idx, count in zip(train_unique, train_counts):
        if idx < len(alphabet):
            print(f"Klasse {alphabet[idx]}: {count} Samples")
    
    print("\nKlassenverteilung im Validierungsdatensatz:")
    for idx, count in zip(val_unique, val_counts):
        if idx < len(alphabet):
            print(f"Klasse {alphabet[idx]}: {count} Samples")
    
    # Balanciere unterrepräsentierte Klassen durch spezifische Augmentation
    # Kommentiere die folgende Zeile aus, wenn du keine zusätzliche Klassenbalancierung willst
    X_train, y_train = augment_minority_classes(X_train, y_train, alphabet, augmentation_factor=1.5, problem_classes=PROBLEM_CLASSES)
    
    # Berechne Klassengewichte für unbalanciertes Dataset
    class_weights = get_class_weights(y_train, alphabet)
    print("Verwende Klassengewichte für besseren Umgang mit Problemklassen")
    
    print(f"Trainingsdaten: {len(X_train)} Samples, Validierungsdaten: {len(X_val)} Samples")
    
    # Erstelle DataLoader
    train_dataset = HandSignDataset(X_train, y_train, augment=True)  # Augmentation aktiviert
    val_dataset = HandSignDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialisiere Modell und lade vortrainiertes Modell, falls gewünscht
    print(f"Initialisiere Modell auf {DEVICE}...")
    model = HandSignNet(num_classes=len(alphabet)).to(DEVICE)
    
    # Kostenfunktion mit Klassengewichten
    if class_weights:
        # Konvertiere Klassengewichte zu Tensor
        weights = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(len(alphabet))])
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        print("Verwende gewichtete Verlustfunktion")
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Leichte L2-Regularisierung
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    initial_val_acc = 0
    if load_model and os.path.exists(MODEL_PATH):
        print(f"Lade vortrainiertes Modell von {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Vortrainiertes Modell erfolgreich geladen!")
        
        # Führe eine Validierung mit dem geladenen Modell durch
        print("Validiere geladenes Modell...")
        initial_val_loss, initial_val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        print(f"Initiale Validierungs-Genauigkeit: {initial_val_acc:.2f}%")
    else:
        if load_model:
            print(f"Kein vortrainiertes Modell gefunden unter {MODEL_PATH}. Starte mit neuem Modell.")
        else:
            print("Training mit neuem Modell gestartet.")
    
    best_val_acc = initial_val_acc
    
    # Early Stopping Variable
    no_improve_epochs = 0
    
    # Training
    print("Starte Training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        # Learning Rate Anpassung
        scheduler.step(val_loss)
        
        # Speichere Metriken
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Speichere bestes Modell
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Neues bestes Modell gespeichert mit Accuracy: {val_acc:.2f}%")
            no_improve_epochs = 0  # Reset Counter
        else:
            no_improve_epochs += 1
            print(f"Keine Verbesserung seit {no_improve_epochs} Epochen")
        
        # Ausgabe
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Early Stopping
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early Stopping nach {epoch+1} Epochen ohne Verbesserung.")
            break
    
    # Lade bestes Modell für finale Evaluation
    if os.path.exists(MODEL_PATH):
        print("Lade bestes Modell für finale Evaluation...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _, final_acc, final_preds, final_labels = validate(model, val_loader, criterion, DEVICE)
        
        # Plotte Ergebnisse
        plot_training_history(train_losses, val_losses, train_accs, val_accs)
        
        # Erstelle und analysiere Konfusionsmatrix
        cm, worst_classes_idx, top_confusion_pairs = plot_both_confusion_matrices(final_labels, final_preds, alphabet)
        
        # Speichere detaillierte Metriken
        precision, recall, f1, support = precision_recall_fscore_support(final_labels, final_preds)
        
        # Speichere Bericht in CSV-Datei
        with open('classification_report.csv', 'w') as f:
            f.write('Buchstabe,Precision,Recall,F1-Score,Support\n')
            for i in range(len(alphabet)):
                f.write(f"{alphabet[i]},{precision[i]:.3f},{recall[i]:.3f},{f1[i]:.3f},{support[i]}\n")
        
        print("\nKlassifikationsbericht wurde in 'classification_report.csv' gespeichert.")
        print("Konfusionsmatrizen wurden in 'confusion_matrix.png' gespeichert.")
        
        # Vorschlag für aktualisierte PROBLEM_CLASSES
        print("\nVorschlag für aktualisierte PROBLEM_CLASSES-Dictionary basierend auf den Ergebnissen:")
        print("PROBLEM_CLASSES = {")
        for idx in worst_classes_idx:
            class_name = alphabet[idx]
            factor = 4.0 if f1[idx] < 0.2 else 3.0 if f1[idx] < 0.5 else 2.5 if f1[idx] < 0.8 else 2.0 if f1[idx] < 0.9 else 1.8
            print(f"    '{class_name}': {factor:.1f},  # F1-Score: {f1[idx]:.3f}")
        print("}")
    else:
        print("Kein gespeichertes Modell gefunden. Überspringe finale Evaluation.")
        # Verwende die Ergebnisse der letzten Epoche für die Plots
        final_preds = val_preds
        final_labels = val_labels
        plot_training_history(train_losses, val_losses, train_accs, val_accs)
        if len(final_labels) > 0:  # Nur wenn wir Validierungsdaten haben
            # Erstelle und analysiere Konfusionsmatrix
            cm, worst_classes_idx, top_confusion_pairs = plot_both_confusion_matrices(final_labels, final_preds, alphabet)
            
            # Speichere detaillierte Metriken
            precision, recall, f1, support = precision_recall_fscore_support(final_labels, final_preds)
            
            # Speichere Bericht in CSV-Datei
            with open('classification_report.csv', 'w') as f:
                f.write('Buchstabe,Precision,Recall,F1-Score,Support\n')
                for i in range(len(alphabet)):
                    f.write(f"{alphabet[i]},{precision[i]:.3f},{recall[i]:.3f},{f1[i]:.3f},{support[i]}\n")
            
            print("\nKlassifikationsbericht wurde in 'classification_report.csv' gespeichert.")
            print("Konfusionsmatrizen wurden in 'confusion_matrix.png' gespeichert.")
    
    print(f"\nBeste Validierungs-Accuracy: {best_val_acc:.2f}%")
    if best_val_acc > initial_val_acc:
        print(f"Verbesserung gegenüber initialem Modell: +{best_val_acc - initial_val_acc:.2f}%")

if __name__ == "__main__":
    # Prüfe, ob ein Kommandozeilenargument übergeben wurde
    load_model = False  # Standard: Lade vorhandenes Modell nicht
    
    main(load_model) 