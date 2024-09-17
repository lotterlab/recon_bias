import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.classifier = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer
        self.classifier.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.classifier.conv1.out_channels,
            kernel_size=self.classifier.conv1.kernel_size,
            stride=self.classifier.conv1.stride,
            padding=self.classifier.conv1.padding,
            bias=False
        )

        # Remove the existing fully connected layer
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Identity()

        # Add new fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.fc_layers(x)
        return x
