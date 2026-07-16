import os
import torch

# ==========================================================
# Base Directories
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# Dataset
# ==========================================================

NUM_CLASSES = 6

BATCH_SIZE = 32

# ==========================================================
# Federated Learning Configuration
# ==========================================================

NUM_CLIENTS = 3

CLIENT_NAMES = [
    "hospital_A",
    "hospital_B",
    "hospital_C"
]

COMMUNICATION_ROUNDS = 10

LOCAL_EPOCHS = 2

CLIENT_FRACTION = 1.0

# ==========================================================
# Optimizer
# ==========================================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# ==========================================================
# Checkpoints
# ==========================================================

GLOBAL_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "global_model.pth"
)

BEST_GLOBAL_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_global_model.pth"
)

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42