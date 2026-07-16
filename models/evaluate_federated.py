import os

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import build_model
from dataloader import get_dataloaders


# ==========================================================
# Configuration
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_global_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 6

BATCH_SIZE = 32


CLASS_NAMES = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumothorax",
    "Pneumonia"
]


def main():

    # ==========================================================
    # Load Test Data
    # ==========================================================

    _, _, test_loader = get_dataloaders(

        train_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "train.csv"
        ),

        valid_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "valid.csv"
        ),

        test_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "test.csv"
        ),

        image_dir=os.path.join(
            BASE_DIR,
            "data",
            "images"
        ),

        batch_size=BATCH_SIZE,
        num_workers=0        # <-- IMPORTANT on Windows
    )

    # ==========================================================
    # Load Model
    # ==========================================================

    model = build_model(NUM_CLASSES)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    # ==========================================================
    # Evaluation
    # ==========================================================

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    print("\nEvaluating Model...\n")

    with torch.no_grad():

        progress_bar = tqdm(
            test_loader,
            desc="Testing"
        )

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    # ==========================================================
    # Metrics
    # ==========================================================

    test_loss = running_loss / len(test_loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # ==========================================================
    # Results
    # ==========================================================

    print("=" * 60)
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("=" * 60)

    print("\nClassification Report\n")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix\n")

    print(cm)

if __name__ == "__main__":
    main()