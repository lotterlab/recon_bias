import pandas as pd
import os
import argparse
import yaml
from typing import List, Optional
import numpy as np
import torch
import datetime
from src.utils.labels import extract_labels_from_row

def majority_voting(predictions: List[int]) -> int:
    """Aggregate predictions via majority voting."""
    return max(set(predictions), key=predictions.count)

def process_patient_data(
    df: pd.DataFrame,
    classifiers: List[dict],
    reconstruction_model: Optional[torch.nn.Module] = None
) -> dict:
    """Process each patient, returning a dictionary of results."""
    patient_info = {
        "patient_id": df["patient_id"].iloc[0],
        "sex": df["sex"].iloc[0],
        "age": df["age_at_mri"].iloc[0],
    }

    for classifier_info in classifiers:
        classifier_name = classifier_info["name"]
        classifier = classifier_info["classifier"]

        patient_predictions = []
        patient_recon_predictions = []
        patient_gt = None

        for _, row in df.iterrows():
            image = np.load(row["file_path"])
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            labels = extract_labels_from_row(row).unsqueeze(0)

            with torch.no_grad():
                # Prediction on original image
                output = classifier(image_tensor)
                pred = classifier.classification_criteria(output)
                patient_predictions.append(pred.item())

                # Ground truth
                if patient_gt is None:  # We only need to calculate GT once
                    patient_gt = classifier.target_transformation(labels).item()

                # Prediction on reconstruction if reconstruction model exists
                if reconstruction_model is not None:
                    recon_image = reconstruction_model(image_tensor)
                    recon_output = classifier(recon_image)
                    recon_pred = classifier.classification_criteria(recon_output)
                    patient_recon_predictions.append(recon_pred.item())

        # Aggregate predictions using majority voting
        majority_pred = majority_voting(patient_predictions)
        patient_info[f"{classifier_name}_gt"] = patient_gt
        patient_info[f"{classifier_name}_pred"] = majority_pred

        if reconstruction_model is not None:
            majority_recon_pred = majority_voting(patient_recon_predictions)
            patient_info[f"{classifier_name}_recon"] = majority_recon_pred

    return patient_info


def process_patients(
    metadata: pd.DataFrame,
    classifiers: List[dict],
    reconstruction_model: Optional[torch.nn.Module] = None,
    number_of_images = None,
) -> List[dict]:
    """Process all patients in the metadata file."""
    patient_infos = []

    index = 0
    for patient_id, patient_df in metadata.groupby("patient_id"):
        if number_of_images is not None:
            if index >= number_of_images: 
                break
        index += 1
        print(f"Processing patient {patient_id}...")
        patient_info = process_patient_data(patient_df, classifiers, reconstruction_model)
        patient_infos.append(patient_info)

    return patient_infos