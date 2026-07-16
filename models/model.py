from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch


def build_model(num_classes):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model

###--------TESTING---------###

if __name__ == "__main__":

    model = build_model(7)

    dummy = torch.randn(8,3,224,224)

    output = model(dummy)

    print(output.shape)