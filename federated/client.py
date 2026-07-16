import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from federated.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOCAL_EPOCHS
)


class FederatedClient:

    def __init__(
        self,
        client_name,
        model,
        train_loader
    ):

        self.client_name = client_name

        self.device = DEVICE

        self.model = copy.deepcopy(model).to(self.device)

        self.train_loader = train_loader

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

    # -------------------------------------------------------
    # Receive Global Model
    # -------------------------------------------------------

    def set_parameters(self, global_model):

        self.model.load_state_dict(
            global_model.state_dict()
        )

    # -------------------------------------------------------
    # Local Training
    # -------------------------------------------------------

    def train(self):

        self.model.train()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for epoch in range(LOCAL_EPOCHS):

            progress_bar = tqdm(
                self.train_loader,
                desc=f"{self.client_name} | Epoch {epoch+1}",
                leave=False
            )

            for images, labels in progress_bar:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

                _, predictions = torch.max(outputs, dim=1)

                running_correct += (
                    predictions == labels
                ).sum().item()

                total_samples += labels.size(0)

                progress_bar.set_postfix(
                    loss=loss.item()
                )

        epoch_loss = running_loss / total_samples
        epoch_accuracy = running_correct / total_samples

        return {
            "weights": copy.deepcopy(
                self.model.state_dict()
            ),
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "samples": total_samples
        }

    # -------------------------------------------------------
    # Return Local Model
    # -------------------------------------------------------

    def get_model(self):

        return self.model

    # -------------------------------------------------------
    # Return Model Weights
    # -------------------------------------------------------

    def get_parameters(self):

        return copy.deepcopy(
            self.model.state_dict()
        )