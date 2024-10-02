import torch
import torchvision
from torch import nn
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, resnet18,
                                resnet50)


class ResNetClassifierNetwork(nn.Module):
    def __init__(self, num_classes: int, resnet_version="resnet18"):
        super().__init__()
        if resnet_version == "resnet18":
            self.classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_version == "resnet50":
            self.classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown ResNet version: {resnet_version}")

        # Modify the first convolutional layer
        # Average the weights across the RGB channels to initialize the new conv1
        weight = self.classifier.conv1.weight
        self.classifier.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.classifier.conv1.out_channels,
            kernel_size=self.classifier.conv1.kernel_size,
            stride=self.classifier.conv1.stride,
            padding=self.classifier.conv1.padding,
            bias=False,
        )
        self.classifier.conv1.weight = nn.Parameter(weight.mean(dim=1, keepdim=True))

        # Remove the existing fully connected layer
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Identity()

        # Add new fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.fc_layers(x)
        return x
