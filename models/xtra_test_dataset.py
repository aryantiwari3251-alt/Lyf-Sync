from dataset import ChestXrayDataset
from transforms import train_transform

dataset = ChestXrayDataset(
    csv_file="data/processed/train.csv",
    image_dir="data/images",
    transform=train_transform
)

print("Dataset Size :", len(dataset))

image, label = dataset[0]

print("Image Shape :", image.shape)
print("Label :", label)