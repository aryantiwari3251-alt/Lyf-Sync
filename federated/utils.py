import random
import os

import numpy as np
import torch


# ==========================================================
# Set Random Seed
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(model, filepath):

    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        filepath
    )


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(model, filepath, device):

    state_dict = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(state_dict)

    return model


# ==========================================================
# Count Trainable Parameters
# ==========================================================

def count_parameters(model):

    return sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )


# ==========================================================
# Print Round Summary
# ==========================================================

def print_round_summary(

    round_number,

    total_rounds,

    loss,

    accuracy

):

    print()

    print("=" * 60)

    print(

        f"Communication Round "

        f"{round_number}/{total_rounds}"

    )

    print("=" * 60)

    print(f"Loss     : {loss:.4f}")

    print(f"Accuracy : {accuracy:.4f}")

    print("=" * 60)


# ==========================================================
# Print Client Summary
# ==========================================================

def print_client_summary(

    client_name,

    loss,

    accuracy,

    samples

):

    print(

        f"{client_name:<15}"

        f" Loss: {loss:.4f}"

        f" Accuracy: {accuracy:.4f}"

        f" Samples: {samples}"

    )