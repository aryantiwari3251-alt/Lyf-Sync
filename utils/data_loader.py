from pathlib import Path
import pandas as pd

# Resolves the absolute path to the root 'DELL--LYFSYNC' directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Data Directories
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
RESOURCE_DIR = BASE_DIR / "resource_management"

def load_confusion_matrix():
    """Reads the generated CSV matrix for the Analytics page."""
    csv_path = RESULTS_DIR / "confusion_matrix.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    return None

def load_classification_report():
    """Reads the raw text classification metrics."""
    txt_path = RESULTS_DIR / "classification_report.txt"
    if txt_path.exists():
        with open(txt_path, "r") as f:
            return f.read()
    return "Classification report not found."

def load_federated_metrics():
    """Reads communication rounds and loss/accuracy steps."""
    csv_path = RESULTS_DIR / "federated_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

def get_model_path(model_type="federated"):
    """Returns absolute path to the designated pre-trained weights."""
    if model_type == "federated":
        return MODELS_DIR / "best_global_model.pth"
    return MODELS_DIR / "centralized_model.pth"