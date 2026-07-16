import os

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):
        """
        Args:
            csv_file : Path to train.csv / valid.csv / hospital_A.csv
            image_dir : Folder containing all X-ray images
            transform : torchvision transforms
        """

        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Label Encoding
        self.label_map = {
            "No Finding": 0,
            "Effusion": 1,
            "Atelectasis": 2,
            "Pneumothorax": 3,
            "Cardiomegaly": 4,
            "Pneumonia": 5
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_name = row["Image Index"]

        label = row["Finding Labels"]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        if not os.path.exists(image_path):
           raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if label not in self.label_map:
            raise ValueError(f"Unknown label found: {label}")

        label = self.label_map[label]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label