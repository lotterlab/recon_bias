import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml

from src.data.reconstruction_dataset import ReconstructionDataset
from src.evaluation.chex_prediction import classifier_predictions
from src.evaluation.chex_evaluation import plot_classifier_metrics, plot_fairness_metrics
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.GAN import UnetGenerator
from src.model.classification.torchxrayvision import (
    CheX_Dataset,
    XRayCenterCrop,
    XRayResizer,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # For reproducibility
    torch.backends.cudnn.benchmark = False


def load_metadata(metadata_path: str) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def load_classifier(
    model_path: str,
    num_classes,
    device,
):
    """model = DenseNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model = model.to(device)

    model.network.eval()

    return model"""
    model = torch.load(model_path, map_location=device)  # Load the full model
    model.to(device)
    return model


def load_reconstruction_model(network_type, model_path, device) -> torch.nn.Module:
    if network_type == "UNet":
        model = ReconstructionModel()
        model = model.to(device)

        network = UNet()
        network = network.to(device)

        model.set_network(network)

        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.network.eval()

    elif network_type == "GAN":
        model = UnetGenerator(1, 1, 7)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

    else:
        raise ValueError(f"Unknown network type: {network_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification models.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract paths and classifier information from the config
    data_root = config["data_root"]
    csv_path = config["csv_path"]
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    number_of_samples = config.get("number_of_samples", None)
    results_path = config.get("results_path", None)

    # Create output directory for evaluation
    output_name = f"{output_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)

    transforms_list = [XRayCenterCrop(), XRayResizer(224)]
    transforms_list = transforms.Compose(transforms_list)

    # Process classifiers
    classifier_dataset = CheX_Dataset(
        imgpath=data_root + "/CheXpert-v1.0-small",
        csvpath=csv_path,
        transform=transforms_list,
        data_aug=None,
        unique_patients=False,
        min_window_width=None,
        views="all",
        labels_to_use=None,
        use_class_balancing=False,
        use_no_finding=True,
        split="test",
    )

    # Save config to output directory for reproducibility
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load metadata
    metadata = load_metadata(csv_path)
    metadata = metadata[metadata["split"] == "test"]

    metadata.sort_values(by="Path", inplace=True)
    metadata = metadata.reset_index(drop=True)

    # Device configuration
    device = torch.device("cpu")

    num_classes = classifier_dataset.labels.shape[1]

    # Initialize classifier
    classifier_config = config.get("classifier", None)
    classifier = load_classifier(classifier_config["model_path"], num_classes, device)
    classifier_model = {
        "network": "CheXpert",
        "model": classifier,
        "dataset": classifier_dataset,
    }

    # Accessing reconstruction config
    reconstruction_config = config["reconstruction"]
    reconstruction_models = []
    photon_count = {100000, 10000, 3000}
    for reconstruction_cfg in reconstruction_config["models"]:
        network_type = reconstruction_cfg["network"]
        photon_count = reconstruction_cfg["photon_count"]
        model_path = reconstruction_cfg["model_path"]
        reconstruction = load_reconstruction_model(
            network_type,
            model_path,
            device,
        )

        reconstruction_dataset = ReconstructionDataset(
            data_root=data_root,
            csv_path=csv_path,
            split="test",
            photon_count=photon_count,
        )

        reconstruction_models.append(
            {
                "network": network_type,
                "model": reconstruction,
                "photon_count": photon_count,
                "dataset": reconstruction_dataset,
            }
        )

    if results_path:
        results_df = pd.read_csv(results_path)
    else:
        # Process and evaluate classification
        results = classifier_predictions(
            metadata=metadata,
            classifier_model=classifier_model,
            reconstruction_models=reconstruction_models,
            num_samples=number_of_samples,
            device=device,
        )

        # Create DataFrame for results
        results_df = pd.DataFrame(results)

        # Save results to output directory
        results_df.to_csv(
            os.path.join(output_path, f"{output_name}_chex_results.csv"),
            index=False,
        )

    # Evaluate predictions
    #plot_classifier_metrics(
    #    results_df, classifier_dataset.pathologies, reconstruction_models, output_path
    #)
    plot_fairness_metrics(
        results_df, classifier_dataset.pathologies, reconstruction_models, output_path
    )


if __name__ == "__main__":
    main()
