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
from src.evaluation.chex_evaluation import plot_auroc_and_significance
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.classification.torchxrayvision import CheX_Dataset, XRayCenterCrop, XRayResizer


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
    model_path: str, num_classes, device,
):
    
    """model = DenseNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model = model.to(device)

    model.network.eval()

    return model"""
    model = torch.load(model_path, map_location=device)  # Load the full model
    model.to(device)
    return model


def load_reconstruction_model(model_path, device) -> torch.nn.Module:
    model = ReconstructionModel()
    model = model.to(device)

    network = UNet()

    network = network.to(device)

    model.set_network(network)

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.network.eval()

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

    if results_path:
        results_df = pd.read_csv(results_path)
    else:

        # Save config to output directory for reproducibility
        with open(os.path.join(output_path, "config.yaml"), "w") as f:
            yaml.dump(config, f)

        # Load metadata
        metadata = load_metadata(csv_path)
        metadata = metadata[metadata["split"] == "test"]

        metadata.sort_values(by='Path', inplace=True)
        metadata = metadata.reset_index(drop=True)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transforms_list = [XRayCenterCrop(), XRayResizer(224)]
        transforms_list = transforms.Compose(transforms_list)

        # Process classifiers
        classifier_dataset = CheX_Dataset(
                    imgpath=data_root + "/CheXpert-v1.0-small",
                    csvpath=csv_path,
                    transform=transforms_list, data_aug=None, unique_patients=False,
                    min_window_width=None, views='all',
                    labels_to_use=None, use_class_balancing=False,
                    use_no_finding=True, 
                    split='test')
        
        num_classes = classifier_dataset.labels.shape[1]

        # Initialize classifiers
        classifier_config = config.get("classifier", None)
        classifier = load_classifier(classifier_config["model_path"], num_classes, device)

            # Accessing reconstruction config
        reconstruction_config = config.get("reconstruction", None)  # Get the reconstruction config, if it exists
        reconstruction = load_reconstruction_model(reconstruction_config["model_path"], device)

        reconstruction_dataset = ReconstructionDataset(
            data_root=data_root,
            csv_path=csv_path,
            split="test",
        )


        # Process and evaluate classification
        results = classifier_predictions(
            classification_dataset = classifier_dataset,
            reconstruction_dataset=reconstruction_dataset,
            metadata=metadata,
            classifier=classifier,
            reconstruction=reconstruction,
            num_samples=number_of_samples,
            transforms=transforms_list,
        )

        # Create DataFrame for results
        results_df = pd.DataFrame(results)

        # Save results to output directory
        results_df.to_csv(
            os.path.join(output_path, f"{output_name}_chex_results.csv"),
            index=False,
        )        

    # Evaluate predictions
    plot_auroc_and_significance(results_df, output_path)

if __name__ == "__main__":
    main()
