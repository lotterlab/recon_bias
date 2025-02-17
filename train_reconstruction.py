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
from fairness.classification_model import TGradeBCEClassifier, TTypeBCEClassifier
from fairness.resnet_classification_network import ResNetClassifierNetwork


def load_classifier_models(config, device):
    if config["dataset"] == "chex":
        classifier = torch.load(config["classifier_path"], map_location=device)
        for param in classifier.parameters():
            param.requires_grad = False
        return classifier
    elif config["dataset"] == "ucsf":
        task_models = {}
        for classifier_config in config["classifiers"]:
            if classifier_config["name"] == "TGradeBCEClassifier":
                classifier = TGradeBCEClassifier()
            elif classifier_config["name"] == "TTypeBCEClassifier":
                classifier = TTypeBCEClassifier()
            classifier = classifier.to(device)

            network = ResNetClassifierNetwork(num_classes=classifier.num_classes
                                                , resnet_version="resnet18")
            
            network = network.to(device)
            classifier.set_network(network)
            classifier.load_state_dict(torch.load(classifier_config["path"], map_location=device))
            for param in classifier.parameters():
                param.requires_grad = False
            task_models[classifier_config["name"]] = classifier

        def apply_task_models(x):
            first_output = task_models["TGradeBCEClassifier"](x)
            second_output = task_models["TTypeBCEClassifier"](x)
            return torch.cat((first_output, second_output), dim=1)
        return apply_task_models

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
    learning_rate = config["learning_rate"]
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
            config=config,
        )
        val_dataset = ChexDataset(
            config=config,
        )
    elif config["dataset"] == "ucsf":
        train_dataset = UcsfDataset(
            config=config,
        )
        val_dataset = UcsfDataset(
            config=config,
        )

    val_sampler = None
    train_sampler = None
    shuffle = True

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

    # load classifier
    classifier_models = load_classifier_models(config, device)

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
        classifier_models=classifier_models,
        fairness_lambda=config["fairness_lambda"],
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
