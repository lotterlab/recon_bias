import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Use the updated weights parameter
        self.classifier = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.classifier.conv1 = nn.Conv2d(
            in_channels=1,  # Change this to 1 channel
            out_channels=self.classifier.conv1.out_channels,
            kernel_size=self.classifier.conv1.kernel_size,
            stride=self.classifier.conv1.stride,
            padding=self.classifier.conv1.padding,
            bias=False
        )
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Identity()
        self.linear = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.classifier(x)
        x = self.linear(x)
        return x
