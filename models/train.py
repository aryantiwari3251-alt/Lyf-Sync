import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import build_model
from dataloader import get_dataloaders


# =====================================================
# Configuration
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# =====================================================
# Training Function
# =====================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, dim=1)

        running_correct += (predictions == labels).sum().item()

        total_samples += labels.size(0)

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy


# =====================================================
# Validation Function
# =====================================================

def validate_one_epoch(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, dim=1)

            running_correct += (predictions == labels).sum().item()

            total_samples += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy


# =====================================================
# Main Function
# =====================================================

def main():

    train_loader, valid_loader, test_loader = get_dataloaders(
        train_csv=os.path.join(BASE_DIR, "data", "processed", "train.csv"),
        valid_csv=os.path.join(BASE_DIR, "data", "processed", "valid.csv"),
        test_csv=os.path.join(BASE_DIR, "data", "processed", "test.csv"),
        image_dir=os.path.join(BASE_DIR, "data", "images"),
        batch_size=BATCH_SIZE
    )

    model = build_model(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    best_accuracy = 0.0

    train_loss_history = []
    train_accuracy_history = []

    valid_loss_history = []
    valid_accuracy_history = []

    print("\n==============================")
    print("Starting Training")
    print("==============================\n")

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        valid_loss, valid_accuracy = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            DEVICE
        )

        scheduler.step()

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        valid_loss_history.append(valid_loss)
        valid_accuracy_history.append(valid_accuracy)

        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Train Accuracy  : {train_accuracy:.4f}")

        print(f"Valid Loss      : {valid_loss:.4f}")
        print(f"Valid Accuracy  : {valid_accuracy:.4f}")

        if valid_accuracy > best_accuracy:

            best_accuracy = valid_accuracy

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print("✅ Best model saved.")

        print("-" * 50)

    print("\nTraining Completed Successfully!")

    print(f"Best Validation Accuracy : {best_accuracy:.4f}")


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()