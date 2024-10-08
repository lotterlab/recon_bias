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
            if classifier_name not in ground_truth:
                ground_truth[classifier_name] = {}

            slice_scores = []
            patient_gt = None

            for slice_data, slice_label in patient_data:
                if patient_gt is None:
                    slice_label = slice_label.unsqueeze(0)
                    patient_gt = classifier_model.target_transformation(slice_label)

                x = slice_data.unsqueeze(0)

                with torch.no_grad():
                    class_output = classifier_model(x)
                    class_score = classifier_model.final_activation(class_output).cpu().numpy().item()  # Get scalar score
                    slice_scores.append(class_score)

            # Store slice-level scores and classifier-specific ground truth for this patient
            predictions[classifier_name][patient_id] = np.array(slice_scores)
            ground_truth[classifier_name][patient_id] = patient_gt

    return predictions, ground_truth

def aggregate_patient_scores(metadata, classifier_results, ground_truth):
    """
    Aggregates slice-level scores using mean, median, max, most confident, and other metrics.
    
    Args:
        metadata: DataFrame containing patient metadata.
        classifier_results: Dictionary containing classifier predictions for each patient and slice.
        ground_truth: Dictionary containing classifier-specific ground truth labels for each patient.
    
    Returns:
        DataFrame with patient_id, aggregated scores for each classifier, and ground truth labels.
    """
    aggregated_results = []

    for patient_id in metadata["patient_id"].unique():

        patient_exists = False
        for classifier_name, predictions_dict in classifier_results.items():
            if patient_id in predictions_dict:
                patient_exists = True

        if not patient_exists:
            continue

        patient_results = {"patient_id": patient_id}

        for classifier_name, predictions_dict in classifier_results.items():

            slice_scores = predictions_dict[patient_id]

            # Aggregate the slice-level scores for this classifier
            patient_results[f"{classifier_name}_mean"] = np.mean(slice_scores)
            patient_results[f"{classifier_name}_median"] = np.median(slice_scores)
            patient_results[f"{classifier_name}_max"] = np.max(slice_scores)
            patient_results[f"{classifier_name}_min"] = np.min(slice_scores)
            patient_results[f"{classifier_name}_geometric_mean"] = gmean(slice_scores)
            patient_results[f"{classifier_name}_trimmed_mean"] = trim_mean(slice_scores, 0.05)
            patient_results[f"{classifier_name}_most_confident"] = slice_scores[np.argmax(np.abs(slice_scores - 0.5))]
            patient_results[f"{classifier_name}_ground_truth"] = ground_truth[classifier_name][patient_id].numpy().item()


        aggregated_results.append(patient_results)

    df_aggregated = pd.DataFrame(aggregated_results)
    return df_aggregated

def calculate_performance_metrics(aggregated_results, classifiers, output_path):
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

        for agg_method in ["mean", "median", "max", "min", "geometric_mean", "trimmed_mean", "most_confident"]:
            predictions = []
            labels = []

            for _, row in aggregated_results.iterrows():
                prediction = row[f"{classifier_name}_{agg_method}"]
                gt = row[f"{classifier_name}_ground_truth"]

                predictions.append(prediction)
                labels.append(gt)

            print(f"Calculating performance metrics for {classifier_name} using {agg_method} aggregation...")
            print("Predictions:", predictions)
            print("Labels:", labels)
            metric = classifier.evaluation_performance_metric(np.array(predictions), np.array(labels))

            # Store results
            performance_results.append({
                "classifier": classifier_name,
                "aggregation_method": agg_method,
                "performance_metric": metric
            })

    df_performance = pd.DataFrame(performance_results)
    df_performance.to_csv(os.path.join(output_path, "performance_metrics.csv"), index=False)
    
    return df_performance

def main():
    parser = argparse.ArgumentParser(description="Evaluate classification models.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_root = config["data_root"]
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    output_name = f"{output_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, output_name)
    num_samples = config.get("num_samples", None)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    metadata = load_metadata(data_root + "/metadata.csv")
    metadata = metadata[metadata["split"] == "test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifiers_config = config["classifiers"]

    classifier_dataset = ClassificationDataset(
        data_root=data_root,
        transform=transforms.Compose([min_max_slice_normalization]),
        split="test",
        number_of_samples=num_samples,
        seed=config.get("seed", 42),
        type=classifiers_config.get("type", "T2"),
        pathology=classifiers_config.get("pathology", None),
        lower_slice=classifiers_config.get("lower_slice", None),
        upper_slice=classifiers_config.get("upper_slice", None),
        age_bins=config.get("age_bins", [0, 60, 100]), 
        os_bins=config.get("os_bins", 4),
        evaluation=True,
    )

    classifiers = []
    for classifier_cfg in classifiers_config["models"]:
        classifier = load_classifier(
            classifier_cfg["type"], classifier_cfg["network"], classifier_cfg["model_path"],
            device, classifier_cfg, classifier_dataset
        )
        classifiers.append({"model": classifier, "name": classifier_cfg["type"]})

    results, ground_truth = classifier_predictions(classifier_dataset, metadata, classifiers, num_samples)
    aggregated_results = aggregate_patient_scores(metadata, results, ground_truth)
    calculate_performance_metrics(aggregated_results, classifiers, output_path)

    print("Performance metrics saved at:", os.path.join(output_path, "performance_metrics.csv"))

if __name__ == "__main__":
    main()