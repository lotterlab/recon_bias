import argparse
import datetime
import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml

from src.data.segmentation_dataset import SegmentationDataset
from src.data.reconstruction_dataset import ReconstructionDataset
from src.evaluation.segmentation_prediction import segmentation_predictions
from src.evaluation.segmentation_evaluation import evaluate_segmentation
from src.model.segmentation.segmentation_model import SegmentationModel
import segmentation_models_pytorch as smp  # For pretrained UNet
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from src.utils.transformations import min_max_slice_normalization


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


def load_segmentation_model(model_path: str, device) -> SegmentationModel:
    """Loads the appropriate classifier based on type and network."""

    model = SegmentationModel()

    model = model.to(device)

    # TODO how to load?
    network = smp.Unet(
        in_channels=1,
        classes=1,
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights=None,
        decoder_channels=(256, 128, 64, 32, 16),
        activation="sigmoid",
    )

    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.network.eval()

    return model


def load_reconstruction_model(network_type, model_path, device) -> torch.nn.Module:
    model = ReconstructionModel()
    model = model.to(device)

    if network_type == "VGG":
        network = VGGReconstructionNetwork(get_configs("vgg16"))
    elif network_type == "UNet":
        network = UNet()
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network.to(device)

    model.set_network(network)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    image_type = config["type"]
    pathology = config["pathology"]
    num_samples = config.get("num_samples", None)
    lower_slice = config.get("lower_slice", None)
    upper_slice = config.get("upper_slice", None)
    age_bins = config.get("age_bins", [0, 58, 100])
    results_path = config.get("results_path", None)

    # Create output directory for evaluation
    output_name = f"{output_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)

    # Save config to output directory for reproducibility
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load metadata
    metadata = load_metadata(data_root + "/metadata.csv")
    metadata = metadata[metadata["split"] == "test"]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            min_max_slice_normalization,
        ]
    )

    seed = config.get("seed", 42)

    # Process classifiers
    segmentation_dataset = SegmentationDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_samples,
        seed=seed,
        type=image_type,
        pathology=pathology,
        lower_slice=lower_slice,
        upper_slice=upper_slice,
        age_bins=age_bins,
        evaluation=True,
    )

    segmentation_config = config["segmentation"]

    segmentation_model = load_segmentation_model(
        segmentation_config["model_path"], device
    )

    # Initialize classifiers
    reconstruction_config = config["reconstruction"]
    sampling_mask = reconstruction_config.get("sampling_mask", None)
    reconstruction_models = []
    for reconstruction_cfg in reconstruction_config["models"]:
        network_type = reconstruction_cfg["network"]
        name = reconstruction_cfg["name"]
        model_path = reconstruction_cfg["model_path"]
        reconstruction = load_reconstruction_model(
            network_type,
            model_path,
            device,
        )
        reconstruction_models.append({"model": reconstruction, "name": name})

    reconstruction_dataset = ReconstructionDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_samples,
        seed=seed,
        type=image_type,
        pathology=pathology,
        lower_slice=lower_slice,
        upper_slice=upper_slice,
        evaluation=True,
        sampling_mask=sampling_mask,
    )

    if results_path is None:
        # Process and evaluate classification
        results = segmentation_predictions(
            data_root=data_root,
            segmentation_dataset=segmentation_dataset,
            reconstruction_dataset=reconstruction_dataset,
            metadata=metadata,
            segmentation_model=segmentation_model,
            reconstruction_models=reconstruction_models,
            num_samples=num_samples,
        )

        # Create DataFrame for results
        results_df = pd.DataFrame(results)

        # Save results to output directory
        results_df.to_csv(
            os.path.join(output_path, f"{output_name}_results.csv"),
            index=False,
        )
    else:
        results_df = pd.read_csv(results_path)

    # Evaluate predictions
    evaluate_segmentation(
        df=results_df,
        segmentation_model=segmentation_model,
        reconstruction_models=reconstruction_models,
        output_dir=output_path,
    )


if __name__ == "__main__":
    main()
