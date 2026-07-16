import torch
from dataloader import get_dataloaders
from model import build_model


def main():
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_csv="data/processed/train.csv",
        valid_csv="data/processed/valid.csv",
        test_csv="data/processed/test.csv",
        image_dir="data/images",
        batch_size=16
    )

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)
    print(labels)

    model = build_model(7)

    dummy = torch.randn(4, 3, 224, 224)

    output = model(dummy)

    print(output.shape)


if __name__ == "__main__":
    main()