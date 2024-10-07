import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml
from torch import nn

from src.data.classification_dataset import ClassificationDataset
from src.data.reconstruction_dataset import ReconstructionDataset
from src.evaluation.classifier_prediction import classifier_predictions
from src.evaluation.evaluation import (classifier_evaluation,
                                       reconstruction_evaluation)
from src.evaluation.reconstruction_prediction import reconstruction_predictions
from src.model.classification.classification_model import (ClassifierModel,
                                                           NLLSurvClassifier,
                                                           TGradeBCEClassifier,
                                                           TTypeBCEClassifier, 
                                                           AgeCEClassifier,
                                                           GenderBCEClassifier)
from src.model.classification.resnet_classification_network import \
    ResNetClassifierNetwork
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from src.utils.mock_data import get_mock_data
from src.utils.transformations import min_max_slice_normalization
from scipy.stats import hmean, gmean, trim_mean


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
    classifier_type: str, network_type: str, model_path: str, device, config, dataset
) -> ClassifierModel:
    """Loads the appropriate classifier based on type and network."""
    # Classifier
    if classifier_type == "TTypeBCEClassifier":
        model = TTypeBCEClassifier()
    elif classifier_type == "TGradeBCEClassifier":
        model = TGradeBCEClassifier()
    elif classifier_type == "NLLSurvClassifier":
        eps = config.get("eps", 1e-8)
        bin_size = dataset.os_bin_size
        os_bins = dataset.os_bins
        model = NLLSurvClassifier(bins=os_bins, bin_size=bin_size, eps=eps)
    elif classifier_type == "AgeCEClassifier":
        age_bins = dataset.age_bins
        model = AgeCEClassifier(age_bins=age_bins)
    elif classifier_type == "GenderBCEClassifier":
        model = GenderBCEClassifier()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    model = model.to(device)

    # Model
    if network_type == "ResNet18":
        network = ResNetClassifierNetwork(num_classes=model.num_classes)
    elif network_type == "ResNet50":
        network = ResNetClassifierNetwork(
            num_classes=model.num_classes, resnet_version="resnet50"
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    model.load_state_dict(torch.load(model_path))
    model.network.eval()

    return model

def aggregate_patient_scores(metadata, classifier_results):
    """
    Aggregates slice-level scores using mean, median, max, harmonic mean, and other metrics.
    Stores results in a dataframe for downstream AUROC or C-Index calculations.
    
    Args:
        metadata: DataFrame containing patient metadata.
        classifier_results: Dictionary containing classifier predictions for each patient and slice.
        output_path: Path to save the output CSV file.

    Returns:
        DataFrame with patient_id and aggregated scores for each classifier.
    """
    
    # Initialize a list to hold aggregated results for all patients
    aggregated_results = []

    # Loop through each patient
    for patient_id in metadata["patient_id"].unique():
        
        patient_results = {"patient_id": patient_id}  # Start a dictionary for this patient
        
        # Loop through each classifier's results for this patient
        for classifier_name, predictions_dict in classifier_results.items():
            slice_scores = predictions_dict[patient_id]  # Scores for all slices for this patient
            
            # Aggregate the slice-level scores for this classifier
            patient_results[f"{classifier_name}_mean"] = np.mean(slice_scores)
            patient_results[f"{classifier_name}_median"] = np.median(slice_scores)
            patient_results[f"{classifier_name}_max"] = np.max(slice_scores)
            patient_results[f"{classifier_name}_min"] = np.min(slice_scores)
            patient_results[f"{classifier_name}_harmonic_mean"] = hmean(slice_scores)
            patient_results[f"{classifier_name}_geometric_mean"] = gmean(slice_scores)
            patient_results[f"{classifier_name}_trimmed_mean"] = trim_mean(slice_scores, 0.05)  # Trim 5% extremes
            
        # Append this patient's results to the list
        aggregated_results.append(patient_results)
    
    # Convert the list of dictionaries into a pandas DataFrame
    df_aggregated = pd.DataFrame(aggregated_results)
        
    return df_aggregated

def classifier_predictions(classification_dataset, metadata, classifiers, num_samples=None):
    """
    Generate slice-level predictions (raw scores) and ground truth for each classifier and patient.
    
    Args:
        classification_dataset: The dataset containing patient data.
        metadata: Metadata for the patients.
        classifiers: List of classifiers to generate predictions.
        num_samples: Optional limit for the number of patients to process.
        
    Returns:
        Dictionary of slice-level predictions and ground truth for each classifier.
    """
    predictions = {}
    ground_truth = {}

    index = 0
    for patient_id, patient_df in metadata.groupby("patient_id"):

        if num_samples is not None and index >= num_samples:
            break
        index += 1
        print(f"Processing patient {patient_id}...")
        patient_data = classification_dataset.get_patient_data(patient_id)

        for classifier in classifiers:
            classifier_name = classifier["name"]
            classifier_model = classifier["model"]

            # Initialize predictions and ground truth for this classifier if first time processing
            if classifier_name not in predictions:
                predictions[classifier_name] = {}

            # Collect slice-level predictions and ground truth for the patient
            slice_scores = []
            slice_labels = []

            for slice_data, slice_label in patient_data:
                if patient_id not in ground_truth: 
                    ground_truth[patient_id] = slice_label.cpu().numpy()
                # Unsqueeze to match expected input dimensions for the model
                x = slice_data.unsqueeze(0)
                y = slice_label.unsqueeze(0)

                # Run the classifier on the slice and collect the raw score
                with torch.no_grad():
                    class_output = classifier_model(x)
                    class_score = classifier_model.final_activation(class_output).cpu().numpy()
                    slice_scores.append(class_score)
                    slice_labels.append(y)  # Collect ground truth

            # Store slice-level scores and ground truth for this patient
            predictions[classifier_name][patient_id] = np.array(slice_scores)

    return predictions, ground_truth

def aggregate_patient_scores(metadata, classifier_results, ground_truth):
    """
    Aggregates slice-level scores using mean, median, max, and other metrics.
    Stores results in a dataframe for downstream AUROC or C-Index calculations.
    
    Args:
        metadata: DataFrame containing patient metadata.
        classifier_results: Dictionary containing classifier predictions for each patient and slice.
        ground_truth: Dictionary containing ground truth labels for each patient.
    
    Returns:
        DataFrame with patient_id, aggregated scores for each classifier, and ground truth labels.
    """
    
    # Initialize a list to hold aggregated results for all patients
    aggregated_results = []

    # Loop through each patient
    for patient_id in metadata["patient_id"].unique():
        
        patient_results = {"patient_id": patient_id}  # Start a dictionary for this patient
        
        # Loop through each classifier's results for this patient
        for classifier_name, predictions_dict in classifier_results.items():
            if patient_id not in predictions_dict:
                continue
            slice_scores = predictions_dict[patient_id]  # Scores for all slices for this patient
            
            # Aggregate the slice-level scores for this classifier
            patient_results[f"{classifier_name}_mean"] = np.mean(slice_scores)
            patient_results[f"{classifier_name}_median"] = np.median(slice_scores)
            patient_results[f"{classifier_name}_max"] = np.max(slice_scores)
            patient_results[f"{classifier_name}_min"] = np.min(slice_scores)
            patient_results[f"{classifier_name}_harmonic_mean"] = hmean(slice_scores)
            patient_results[f"{classifier_name}_geometric_mean"] = gmean(slice_scores)
            patient_results[f"{classifier_name}_trimmed_mean"] = trim_mean(slice_scores, 0.05)  # Trim 5% extremes
            
        # Append this patient's results to the list
        aggregated_results.append(patient_results)
    
    # Convert the list of dictionaries into a pandas DataFrame
    df_aggregated = pd.DataFrame(aggregated_results)
        
    return df_aggregated

def calculate_performance_metrics(aggregated_results, ground_truth, metadata, classifiers, output_path):
    """
    Calculate the performance metrics for each classifier and aggregation method.
    
    Args:
        aggregated_results: DataFrame with patient_id and aggregated scores for each classifier.
        classifiers: List of classifiers to generate predictions.
        output_path: Path to save the performance metrics CSV file.
    
    Returns:
        DataFrame containing the performance metrics for each classifier and aggregation method.
    """
    
    performance_results = []

    for classifier_info in classifiers:
        classifier = classifier_info["model"]
        classifier_name = classifier_info["name"]
        
        # Loop through each aggregation method and calculate performance metrics
        for agg_method in ["mean", "median", "max", "min", "harmonic_mean", "geometric_mean", "trimmed_mean"]:

            predictions = []
            ground_truth = {}

            for patient_id in metadata["patient_id"].unique():
                if patient_id not in ground_truth:
                    continue
                prediction = aggregated_results[aggregated_results["patient_id"] == patient_id][f"{classifier_name}_{agg_method}"]
                gt = ground_truth[patient_id]

                predictions.append(prediction)
                ground_truth.append(gt)
                
            metric = classifier.evaluation_performance_metric(predictions, ground_truth)

            # Store results
            performance_results.append({
                "classifier": classifier_name,
                "aggregation_method": agg_method,
                "performance_metric": metric
            })

    # Convert results to DataFrame
    df_performance = pd.DataFrame(performance_results)

    # Save the performance metrics to a CSV file
    df_performance.to_csv(os.path.join(output_path, "performance_metrics.csv"), index=False)
    
    return df_performance

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

    classifiers_config = config["classifiers"]
    num_classifier_samples = classifiers_config.get("num_samples", None)
    lower_slice_classifier = classifiers_config.get("lower_slice", None)
    upper_slice_classifier = classifiers_config.get("upper_slice", None)
    pathology_classifier = classifiers_config.get("pathology", None)
    type_classifier = classifiers_config.get("type", "T2")
    os_bins = config.get("os_bins", 4)
    age_bins = config.get("age_bins", [0, 60, 100])

    transform = transforms.Compose(
        [
            min_max_slice_normalization,
        ]
    )

    seed = config.get("seed", 42)

    # Process classifiers
    classifier_dataset = ClassificationDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_classifier_samples,
        seed=seed,
        type=type_classifier,
        pathology=pathology_classifier,
        lower_slice=lower_slice_classifier,
        upper_slice=upper_slice_classifier,
        age_bins=age_bins, 
        os_bins=os_bins,
        evaluation=True,
    )

    # Initialize classifiers
    classifiers = []
    for classifier_cfg in classifiers_config["models"]:
        classifier_type = classifier_cfg["type"]
        network_type = classifier_cfg["network"]
        model_path = classifier_cfg["model_path"]
        classifier = load_classifier(
            classifier_type, network_type, model_path, device, classifier_cfg, classifier_dataset
        )
        classifiers.append({"model": classifier, "name": classifier_type})

    # Generate slice-level predictions and aggregate the scores
    results, ground_truth = classifier_predictions(classification_dataset=classifier_dataset, metadata=metadata, classifiers=classifiers, num_samples=num_classifier_samples)
    aggregated_results = aggregate_patient_scores(metadata, results, ground_truth)

    # Calculate performance metrics and save them
    df_performance = calculate_performance_metrics(aggregated_results=aggregated_results, ground_truth=ground_truth, metadata=metadata, classifiers=classifiers, output_path=output_path)

    print("Performance metrics saved at:", os.path.join(output_path, "performance_metrics.csv"))

if __name__ == "__main__":
    main()
