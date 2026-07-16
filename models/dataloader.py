from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from transforms import (
    train_transform,
    valid_transform,
    test_transform
)


def get_dataloaders(
    train_csv,
    valid_csv,
    test_csv,
    image_dir,
    batch_size=32,
    num_workers=0
):

    train_dataset = ChestXrayDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=train_transform
    )

    valid_dataset = ChestXrayDataset(
        csv_file=valid_csv,
        image_dir=image_dir,
        transform=valid_transform
    )

    test_dataset = ChestXrayDataset(
        csv_file=test_csv,
        image_dir=image_dir,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, valid_loader, test_loader

