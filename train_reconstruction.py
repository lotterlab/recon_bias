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
from src.data.ucsf_dataset import UsfcDataset
from src.model.classification.classification_model import (
    AgeCEClassifier,
    GenderBCEClassifier,
)
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from src.trainer.trainer import Trainer
from src.utils.transformations import min_max_slice_normalization


def load_task_models(config, device):
    if config["dataset"] == "chex":
        classifier = torch.load(config["path"], map_location=device)
        classifier.eval()
        return classifier
    elif config["dataset"] == "ucsf":
        classifier = torch.load(config["path"], map_location=device)
        return classifier

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
    csv_path = config["csv_path"]
    output_name = config["output_name"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    num_train_samples = config.get("num_train_samples", None)
    num_val_samples = config.get("num_val_samples", None)
    network_type = config.get("network_type", "VGG")
    model_path = config.get("model_path", None)
    network_path = config.get("network_path", None)
    data_root = config["data_root"]
    seed = config.get("seed", 31415)
    save_interval = config.get("save_interval", 1)
    early_stopping_patience = config.get("early_stopping_patience", None)
    photon_count = config.get("photon_count", 1e5)

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
            data_root=data_root,
            csv_path=csv_path,
            split="val_recon",
            number_of_samples=num_train_samples,
            seed=seed,
            photon_count=photon_count,
        )
        val_dataset = ChexDataset(
            data_root=data_root,
            csv_path=csv_path,
            split="train_recon",
            number_of_samples=50,
            seed=seed,
            photon_count=photon_count,
        )
    elif config["dataset"] == "ucsf":
        transform = [min_max_slice_normalization, lambda x: transforms.functional.resize(x.unsqueeze(0), (256, 256)).squeeze(0)]
        transform = transforms.Compose(transform)
        train_dataset = UsfcDataset(
            data_root=data_root,
            transform=transform,
            split="train",
            number_of_samples=num_train_samples,
            seed=seed,
            type=config.get("type", "FLAIR"),
            sampling_mask=config.get("sampling_mask", "radial"),
            lower_slice=config.get("lower_slice", None),
            upper_slice=config.get("upper_slice", None),
            age_bins=config.get("age_bins", [0, 58, 100]),
            num_rays=config.get("num_rays", 60),
        )
        val_dataset = UsfcDataset(
            data_root=data_root,
            transform=transform,
            split="val",
            number_of_samples=num_val_samples,
            seed=seed,
            type=config.get("type", "FLAIR"),
            sampling_mask=config.get("sampling_mask", "radial"),
            lower_slice=config.get("lower_slice", None),
            upper_slice=config.get("upper_slice", None),
            age_bins=config.get("age_bins", [0, 58, 100]),
            num_rays=config.get("num_rays", 60),
        )

    val_sampler = None
    train_sampler = None
    shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        sampler=val_sampler,
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classifier
    model = ReconstructionModel()
    model = model.to(device)

    # Model
    if network_type == "VGG":
        network = VGGReconstructionNetwork(get_configs("vgg16"), network_path)
    elif network_type == "UNet":
        network = UNet()
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # load classifier
    task_models = load_task_models(config["task_models"], device)

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
        task_models=task_models,
        dataset_type=config["dataset"],
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
