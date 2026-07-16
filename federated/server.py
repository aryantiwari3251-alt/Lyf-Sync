import copy
import torch

from federated.strategy import FedAvgStrategy
from federated.config import (
    DEVICE,
    GLOBAL_MODEL_PATH,
    BEST_GLOBAL_MODEL_PATH
)


class FederatedServer:

    def __init__(self, global_model):

        self.device = DEVICE

        self.global_model = copy.deepcopy(
            global_model
        ).to(self.device)

        self.strategy = FedAvgStrategy()

        self.best_accuracy = 0.0

    # ==========================================================
    # Send Global Model
    # ==========================================================

    def distribute_model(self, clients):

        for client in clients:
            client.set_parameters(self.global_model)

    # ==========================================================
    # Collect Client Updates
    # ==========================================================

    def collect_updates(self, clients):

        client_results = []

        for client in clients:

            result = client.train()

            client_results.append(result)

        return client_results

    # ==========================================================
    # Aggregate Models
    # ==========================================================

    def aggregate(self, client_results):

        global_weights = self.strategy.aggregate(
            client_results
        )

        self.global_model.load_state_dict(
            global_weights
        )

    # ==========================================================
    # Aggregate Metrics
    # ==========================================================

    def aggregate_metrics(self, client_results):

        return self.strategy.aggregate_metrics(
            client_results
        )

    # ==========================================================
    # Return Global Model
    # ==========================================================

    def get_global_model(self):

        return self.global_model

    # ==========================================================
    # Save Global Model
    # ==========================================================

    def save_model(self):

        torch.save(
            self.global_model.state_dict(),
            GLOBAL_MODEL_PATH
        )

    # ==========================================================
    # Save Best Global Model
    # ==========================================================

    def save_best_model(self, accuracy):

        if accuracy > self.best_accuracy:

            self.best_accuracy = accuracy

            torch.save(
                self.global_model.state_dict(),
                BEST_GLOBAL_MODEL_PATH
            )

            print(
                f"Best Global Model Saved "
                f"(Accuracy = {accuracy:.4f})"
            )