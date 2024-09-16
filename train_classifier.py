import torch
import os
import torchvision.transforms as transforms

from src.data.classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader

from src.model.classification.classification_model import ResNetClassifier
from src.model.classification.NLLSurvLoss import NLLSurvLoss

from src.trainer.trainer import train_model

def classify():
    print("Starting classification ...")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_time_steps = 20  # Number of discrete time intervals
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32
    log_dir = './runs/os-classification'

    root_dir = '../../data/UCSF-PDGM/processed/'

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets and DataLoaders
    train_dataset = ClassificationDataset(data_root=root_dir, transform=train_transform, split='train')
    val_dataset = ClassificationDataset(data_root=root_dir, transform=val_transform, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = ResNetClassifier(num_classes=num_time_steps)
    model = model.to(device)

    # Loss function
    criterion = NLLSurvLoss()
    criterion = criterion.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir)

    # save the trained model
    if not os.path.exists('../output'):
        os.makedirs('../output')
    torch.save(trained_model.state_dict(), '../output/os_classification_model.pth')

if __name__ == "__main__":
    classify()