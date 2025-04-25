import argparse
import datetime
import os

import torch
import torchvision.transforms as transforms
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import create_balanced_sampler

# Import your dataset, models, and trainer
from src.data.chex_dataset import ChexDataset
from src.data.ucsf_dataset import UcsfDataset
from src.model.classification.classification_model import (
    AgeCEClassifier,
    GenderBCEClassifier,
)
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.chex_unet import ChexUNet
from src.model.reconstruction.ucsf_unet import UcsfUNet
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from src.trainer.trainer import Trainer
from src.utils.transformations import min_max_slice_normalization
from torch.utils.data import WeightedRandomSampler


def main():
    parser = argparse.ArgumentParser(description="Train a reconstruction model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters from the configuration
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"] * 0.01
    batch_size = config["batch_size"]
    model_path = config.get("model_path", None)
    save_interval = config.get("save_interval", 1)
    early_stopping_patience = config.get("early_stopping_patience", None)

    # Append timestamp to output_name to make it unique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{output_name}_{timestamp}"
    output_dir = os.path.join(output_dir, output_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the configuration back into the output directory for tracking
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    if config["dataset"] == "chex":
    # Datasets and DataLoaders
        train_dataset = ChexDataset(
            opt=config,
        )
        val_dataset = ChexDataset(
            opt=config,
            train=False,
        )
    elif config["dataset"] == "ucsf":
        train_dataset = UcsfDataset(
            opt=config,
        )
        val_dataset = UcsfDataset(
            opt=config,
            train=False,
        )

    print("Before compute_sample_weights")

    val_weights = val_dataset.compute_sample_weights()
    val_sampler = WeightedRandomSampler(val_weights, len(val_weights), replacement=True)

    train_weights = train_dataset.compute_sample_weights()
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    shuffle = False

    print("After compute_sample_weights")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
    )

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Classifier
    model = ReconstructionModel()
    model = model.to(device)

    if config["dataset"] == "chex":
        network = ChexUNet()
    elif config["dataset"] == "ucsf":
        network = UcsfUNet()
        
    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        log_dir=os.path.join(output_dir, "logs"),
        output_dir=output_dir,
        output_name=output_name,
        save_interval=save_interval,
        early_stopping_patience=early_stopping_patience,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
