import os

from models.model import build_model
from models.dataloader import get_dataloaders
from federated.utils import load_checkpoint

from federated.client import FederatedClient
from federated.server import FederatedServer

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from federated.config import (
    BASE_DIR,
    BATCH_SIZE,
    NUM_CLASSES,
    CLIENT_NAMES,
    COMMUNICATION_ROUNDS,
    CENTRALIZED_MODEL_PATH,
    GLOBAL_MODEL_PATH,
    DEVICE
)

# ==========================================================
# Create Clients
# ==========================================================

def create_clients(global_model):

    clients = []

    for client_name in CLIENT_NAMES:

        csv_path = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            f"{client_name}.csv"
        )

        train_loader, _, _ = get_dataloaders(
            train_csv=csv_path,
            valid_csv=csv_path,
            test_csv=csv_path,
            image_dir=os.path.join(
                BASE_DIR,
                "data",
                "images"
            ),
            batch_size=BATCH_SIZE
        )

        client = FederatedClient(
            client_name=client_name,
            model=global_model,
            train_loader=train_loader
        )

        clients.append(client)

    return clients


# ==========================================================
# Federated Training
# ==========================================================

def train():

    print("=" * 60)
    print("Starting Federated Learning")
    print("=" * 60)

    # -----------------------------
    # Global Model
    # -----------------------------

    global_model = build_model(NUM_CLASSES)

    if os.path.exists(CENTRALIZED_MODEL_PATH):

        global_model = load_checkpoint(
           global_model,
           CENTRALIZED_MODEL_PATH,
           DEVICE
         )

        print("Loaded centralized model.")

    else:
       print("Centralized model not found. Starting from scratch.")

    # -----------------------------
    # Server
    # -----------------------------

    server = FederatedServer(global_model)

    # -----------------------------
    # Clients
    # -----------------------------

    clients = create_clients(global_model)

    # -----------------------------
    # Communication Rounds
    # -----------------------------

    for round_number in range(COMMUNICATION_ROUNDS):

        print("\n")
        print("=" * 60)
        print(
            f"Communication Round "
            f"{round_number + 1}/{COMMUNICATION_ROUNDS}"
        )
        print("=" * 60)

        # Send Global Model

        server.distribute_model(clients)

        # Local Training

        client_results = server.collect_updates(
            clients
        )
        for client, result in zip(clients, client_results):

          print(
          f"{client.client_name}"
          f" | Loss: {result['loss']:.4f}"
          f" | Accuracy: {result['accuracy']:.4f}"
          )

        # Aggregate

        server.aggregate(client_results)

        # Metrics

        metrics = server.aggregate_metrics(
            client_results
        )

        print(
            f"Global Loss     : "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"Global Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        # Save Models

        server.save_model()

        server.save_best_model(
            metrics["accuracy"]
        )

    print("\n")
    server.save_model()

    print("=" * 60)
    print("Federated Training Completed")
    print("=" * 60)

    print(f"\nGlobal Model Saved At:\n{GLOBAL_MODEL_PATH}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train()