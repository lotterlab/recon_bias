from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision import transforms

from src.utils.labels import extract_labels_from_row
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity

from torchvision.utils import save_image
import os

def majority_voting(predictions: List[int]) -> int:
    """Aggregate predictions via majority voting."""
    return max(set(predictions), key=predictions.count)

def dice_coefficient(pred, target):
    """Calculate the Dice coefficient."""
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return 2 * intersection / union


def process_patient_data(
    patient_info,
    patient_segmentation_data,
    segmentation_model,
    reconstruction_models,
) -> dict:
    """Process each patient, returning a dictionary of results."""

    gt_sums = []
    segmentation_sums = []
    segementation_dices = []
    modified_patient_info = patient_info.copy()

    for i, (x, y) in enumerate(patient_segmentation_data):
        with torch.no_grad():
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            segmentation_output = segmentation_model(x)
            segmentation_output[segmentation_output > 0.5] = 1
            segmentation_output[segmentation_output <= 0.5] = 0

            segmentation_sum = torch.sum(segmentation_output).item()
            segmentation_sums.append(segmentation_sum)

            segmentation_dice = dice_coefficient(segmentation_output, y)
            segementation_dices.append(segmentation_dice.item())

            gt_sum = torch.sum(y).item()
            gt_sums.append(gt_sum)

    
    modified_patient_info["gt_sum"] = np.sum(gt_sums)
    modified_patient_info["segmentation_sum"] = np.sum(segmentation_sums)

    # First mean calculation
    modified_patient_info["segmentation_dice"] = np.mean(segementation_dices)

    for reconstruction_info in reconstruction_models: 
        reconstruction_model = reconstruction_info["model"]
        reconstruction_network = reconstruction_info["network"]
        acceleration = reconstruction_info["acceleration"]
        dataset = reconstruction_info["dataset"]
        patient_reconstruction_data = dataset.get_patient_data(patient_info["patient_id"])

        segmentation_sums = []
        segementation_dices = []
        psnr = []
        ssim = []
        nrmse = []


        for i, (x, y) in enumerate(patient_segmentation_data):
            with torch.no_grad():
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                #save image x_recon to disk 

                x_recon, y_recon_gt = patient_reconstruction_data[i]
                x_recon = x_recon.unsqueeze(0)
                y_recon = reconstruction_model(x_recon)

                segmentation_output = segmentation_model(y_recon)

                segmentation_output[segmentation_output > 0.5] = 1
                segmentation_output[segmentation_output <= 0.5] = 0

                segmentation_sum = torch.sum(segmentation_output).item()
                segmentation_sums.append(segmentation_sum)

                segmentation_dice = dice_coefficient(segmentation_output, y)
                segementation_dices.append(segmentation_dice.item())

                psnr.append(peak_signal_noise_ratio(y_recon_gt.detach().numpy().squeeze(), y_recon.detach().numpy().squeeze(), data_range=1))
                ssim.append(structural_similarity(y_recon_gt.detach().numpy().squeeze(), y_recon.detach().numpy().squeeze(), data_range=1))
                nrmse.append(mean_squared_error(y_recon_gt.detach().numpy().squeeze(), y_recon.detach().numpy().squeeze()))   
        
        modified_patient_info[f"{reconstruction_network}_{acceleration}_segmentation_sum"] = np.sum(segmentation_sums)
        modified_patient_info[f"{reconstruction_network}_{acceleration}_segmentation_dice"] = np.mean(segementation_dices)
        modified_patient_info[f"{reconstruction_network}_{acceleration}_psnr"] = np.mean(psnr)
        modified_patient_info[f"{reconstruction_network}_{acceleration}_ssim"] = np.mean(ssim)
        modified_patient_info[f"{reconstruction_network}_{acceleration}_nrmse"] = np.mean(nrmse)
    
    return modified_patient_info

def segmentation_predictions(
    segmentation_dataset,
    metadata: pd.DataFrame,
    segmentation_model: List[dict],
    reconstruction_models: Optional[torch.nn.Module] = None,
    num_samples=None,
) -> List[dict]:
    """Process all patients in the metadata file."""
    patient_predictions = []

    index = 0
    for patient_id, patient_df in metadata.groupby("patient_id"):
        if num_samples is not None:
            if index >= num_samples:
                break
        index += 1
        print(f"Processing patient {patient_id}...")
        patient_info = {
            "patient_id": patient_id,
            "sex": patient_df["sex"].iloc[0],
            "age": patient_df["age_at_mri"].iloc[0],
            "who_cns_grade": patient_df["who_cns_grade"].iloc[0],
            "final_diagnosis": patient_df["final_diagnosis"].iloc[0],
            "alive": patient_df["alive"].iloc[0],
            "os": patient_df["os"].iloc[0],
        }
        patient_segmentation_data = segmentation_dataset.get_patient_data(
            patient_id
        )

        patient_prediction = process_patient_data(
            patient_info,
            patient_segmentation_data,
            segmentation_model,
            reconstruction_models,
        )
        patient_predictions.append(patient_prediction)

    return patient_predictions
