import copy
import torch


class FedAvgStrategy:

    def __init__(self):
        pass

    # ==========================================================
    # Federated Averaging
    # ==========================================================

    def aggregate(self, client_results):

        """
        Parameters
        ----------
        client_results : list

        [
            {
                "weights": state_dict,
                "samples": 1000,
                "loss": ...,
                "accuracy": ...
            },
            ...
        ]

        Returns
        -------
        global_weights : OrderedDict
        """

        # Total number of training samples
        total_samples = sum(
            client["samples"]
            for client in client_results
        )

        # Copy first client's weights
        global_weights = copy.deepcopy(
            client_results[0]["weights"]
        )

        # Initialize weighted sum
        for key in global_weights.keys():

            global_weights[key] *= (
                client_results[0]["samples"] /
                total_samples
            )

        # Aggregate remaining clients
        for client in client_results[1:]:

            client_weight = (
                client["samples"] /
                total_samples
            )

            for key in global_weights.keys():

                global_weights[key] += (
                    client["weights"][key]
                    * client_weight
                )

        return global_weights

    # ==========================================================
    # Average Metrics
    # ==========================================================

    def aggregate_metrics(self, client_results):

        total_samples = sum(
            client["samples"]
            for client in client_results
        )

        avg_loss = 0.0
        avg_accuracy = 0.0

        for client in client_results:

            weight = client["samples"] / total_samples

            avg_loss += client["loss"] * weight
            avg_accuracy += client["accuracy"] * weight

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy
        }