import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.classification_dataset import ClassificationDataset
from src.model.classification.classification_model import TGradeBCEClassifier, TTypeBCEClassifier, NLLSurvClassifier
from src.model.classification.classification_network import ResNetClassifierNetwork
from src.trainer.trainer import train_model


def classify():
    print("Starting classification ...")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 20
    learning_rate = 0.0001
    batch_size = 16
    log_dir = "./runs/ttype-classification"

    root_dir = "../../data/UCSF-PDGM/processed/"

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Datasets and DataLoaders
    train_dataset = ClassificationDataset(
        data_root=root_dir, transform=train_transform, split="train", number_of_samples=100
    )
    val_dataset = ClassificationDataset(
        data_root=root_dir, transform=val_transform, split="val", number_of_samples=10
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Model
    network = ResNetClassifierNetwork(num_classes=1)
    network = network.to(device)

    # Loss function
    model = TTypeBCEClassifier(network=network)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        device,
        log_dir,
    )

    # save the trained model
    if not os.path.exists("./output"):
        os.makedirs("./output")
    torch.save(trained_model.state_dict(), "./output/ttype_classification_model.pth")


if __name__ == "__main__":
    classify()
