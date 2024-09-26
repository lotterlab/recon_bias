import os
import argparse
import yaml
from typing import List, Optional
import numpy as np
import torch
import datetime
import pandas as pd
from torch import nn

from src.model.classification.resnet_classification_network import ResNetClassifierNetwork
from src.model.classification.classification_model import (
    ClassifierModel,
    TGradeBCEClassifier,
    TTypeBCEClassifier,
    NLLSurvClassifier,
)
from src.evaluation.prediction import process_patients
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.vgg import get_configs, VGGReconstructionNetwork
from src.model.reconstruction.unet import UNet


def load_metadata(metadata_path: str) -> pd.DataFrame:
    return pd.read_csv(metadata_path)

def load_classifier(classifier_type: str, network_type: str, model_path: str, device, config) -> ClassifierModel:
    """Loads the appropriate classifier based on type and network."""
    # Classifier
    if classifier_type == 'TTypeBCEClassifier':
        model = TTypeBCEClassifier()
    elif classifier_type == 'TGradeBCEClassifier':
        model = TGradeBCEClassifier()
    elif classifier_type == 'NLLSurvClassifier':
        bin_size = config.get('bin_size', 1000)
        eps = config.get('eps', 1e-8)
        model = NLLSurvClassifier(bin_size=bin_size, eps=eps)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    model = model.to(device)

    # Model
    if network_type == 'ResNet18':
        network = ResNetClassifierNetwork(num_classes=model.target_size)
    elif network_type == 'ResNet50':
        network = ResNetClassifierNetwork(num_classes=model.target_size, resnet_version='resnet50')
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    model.load_state_dict(torch.load(model_path))
    model.network.eval()

    return model


def load_reconstruction_model(network_type, model_path, device) -> torch.nn.Module:
    model = ReconstructionModel() 
    model = model.to(device)

    print(f"Loading reconstruction model {network_type} from {model_path}...")
    if network_type == 'VGG':
        network = VGGReconstructionNetwork(get_configs("vgg16"))
    elif network_type == 'UNet':
        network = UNet()
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    network = network.to(device)

    model.set_network(network)

    model.load_state_dict(torch.load(model_path))
    model.network.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification models.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract paths and classifier information from the config
    data_root = config["data_root"]
    classifiers_config = config["classifiers"]
    output_dir = config["output_dir"]
    output_name = config["output_name"]

    # Create output directory for evaluation
    output_name = f"{output_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classifier_samples = None
    if "num_samples" in classifiers_config:
            num_classifier_samples = classifiers_config["num_samples"]
    
    # Initialize classifiers
    classifiers = []
    for classifier_cfg in classifiers_config["models"]:
        classifier_type = classifier_cfg["type"]
        network_type = classifier_cfg["network"]
        model_path = classifier_cfg["model_path"]
        classifier = load_classifier(classifier_type, network_type, model_path, device, classifier_cfg)
        classifiers.append({"classifier": classifier, "name": classifier_type})

        # Accessing reconstruction config
    reconstruction_config = config.get('reconstruction', None)  # Get the reconstruction config, if it exists

    reconstruction_model = None  
    num_reconstruction_samples = None

    if reconstruction_config is not None and "model" in reconstruction_config and reconstruction_config["model"] is not None and len(reconstruction_config["model"]) > 0:
        if "num_samples" in reconstruction_config:
            num_reconstruction_samples = reconstruction_config["num_samples"]

        reconstruction_cfg = reconstruction_config["model"][0]  

        if "network" in reconstruction_cfg and "model_path" in reconstruction_cfg:
            network_type = reconstruction_cfg["network"]
            model_path = reconstruction_cfg["model_path"]

            reconstruction_model = load_reconstruction_model(network_type, model_path, device)
            print(f"Reconstruction model {network_type} loaded from {model_path}.")
        else:
            print("Reconstruction model configuration is incomplete or missing required fields.")
        
    else:
        print("No reconstruction model specified.")

    print(f"Number of classifier samples: {num_classifier_samples}")
    print(f"Number of reconstruction samples: {num_reconstruction_samples}")

    # Load metadata
    metadata = load_metadata(data_root + "/metadata.csv")
    metadata = metadata[metadata["split"] == "test"]

    results = process_patients(data_root, metadata, classifiers, reconstruction_model, num_classifier_samples)

    # Create DataFrame for results
    results_df = pd.DataFrame(results)

    # Save results to output directory
    results_df.to_csv(os.path.join(output_path, f"{output_name}_results.csv"), index=False)

    # Save config to output directory for reproducibility
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    print(f"Results saved to {output_path}")

    # Evaluate predictions
    #evaluate_classifiers(results_df, classifiers, output_path)


if __name__ == "__main__":
    main()
