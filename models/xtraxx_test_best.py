import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import build_model

# 1. Create model architecture
model = build_model(num_classes=6)

# 2. Load trained weights
model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location="cpu"
    )
)

# 3. Evaluation mode
model.eval()

print("Model loaded successfully!")