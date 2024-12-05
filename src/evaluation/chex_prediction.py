from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from datetime import datetime

def save_images(classification_image, reconstruction_image, save_dir='.', file_prefix='comparison'):
    """
    Save classification and reconstruction images for comparison without overwriting previous files.

    Args:
        classification_image (torch.Tensor or np.ndarray): Image from classification model (C x H x W or H x W).
        reconstruction_image (torch.Tensor or np.ndarray): Image from reconstruction model (C x H x W or H x W).
        save_dir (str): Directory to save the images.
        file_prefix (str): Prefix for the saved file names.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to NumPy arrays if needed
    if isinstance(classification_image, torch.Tensor):
        classification_image = classification_image.squeeze().cpu().numpy()
    if isinstance(reconstruction_image, torch.Tensor):
        reconstruction_image = reconstruction_image.squeeze().cpu().numpy()

    # Ensure images are 2D for grayscale or 3D for RGB
    if classification_image.ndim == 3 and classification_image.shape[0] in [1, 3]:  # Channels first
        classification_image = np.transpose(classification_image, (1, 2, 0))  # Convert to H x W x C
    if reconstruction_image.ndim == 3 and reconstruction_image.shape[0] in [1, 3]:  # Channels first
        reconstruction_image = np.transpose(reconstruction_image, (1, 2, 0))  # Convert to H x W x C

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_filename = f'{file_prefix}_{timestamp}.png'

    # Plot and save the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(classification_image, cmap='gray' if classification_image.ndim == 2 else None)
    axes[0].set_title('Classification Image')
    axes[0].axis('off')

    axes[1].imshow(reconstruction_image, cmap='gray' if reconstruction_image.ndim == 2 else None)
    axes[1].set_title('Reconstruction Image')
    axes[1].axis('off')

    # Save the comparison plot
    comparison_path = os.path.join(save_dir, unique_filename)
    plt.savefig(comparison_path)
    plt.close(fig)
    print(f"Saved comparison image at: {comparison_path}")

def process_patient_data(
    row,
    pathologies,
    classification_data,
    reconstruction_data,
    classifier,
    reconstruction,
    transforms,
):
    """Process each patient, returning a dictionary of results."""

    # csv fields: Path (key), patient_id, pathology, gt, pred, gt_recon
    result = []
    
    x_class = classification_data["img"]
    x_class = torch.tensor(x_class, dtype=torch.float32).unsqueeze(0)

    x_recon, y_recon = reconstruction_data
    x_recon = x_recon.unsqueeze(0)
    y_recon = y_recon.unsqueeze(0)

    pred = classifier(x_class)
    pred = pred.squeeze(0)
    pred = torch.sigmoid(pred)

    recon = reconstruction(x_recon)

    recon_np = recon.squeeze(0).detach().cpu().numpy()
    transformed_recon_np = transforms(recon_np)
    recon = torch.tensor(transformed_recon_np, dtype=torch.float32).unsqueeze(0)

    pred_recon = classifier(recon)
    pred_recon = pred_recon.squeeze(0)
    pred_recon = torch.sigmoid(pred_recon)

    for i, pathology in enumerate(pathologies):
        gt = row[pathology]
        if gt != 0 and gt != 1:
            continue

        if gt != classification_data["lab"][i]:
            print(f"Error: GT mismatch for pathology {pathology} at index {i}")
        
        row_info = {
            "Path": row["Path"],
            "patient_id": row["PatientID"],
            "pathology": pathology,
            "gt": gt,
            "pred": float(pred[i]),
            "pred_recon": float(pred_recon[i]),
        }


        result.append(row_info)

    return result

def classifier_predictions(
    classification_dataset,
    reconstruction_dataset,
    metadata,
    classifier,
    reconstruction,
    num_samples=None,
    transforms=None,
) -> List[dict]:
    """Process all patients in the metadata file."""
    predictions = []

    index = 0
    # iterate all rows of metadata
    for i, row in metadata.iterrows():
        if num_samples is not None:
            if index >= num_samples:
                break
        index += 1
        if i % 100 == 0:
            print(f"Processing row {row['Path']} at index {i}")
        patient_classification_data = classification_dataset.__getitem__(i)
        patient_reconstruction_data = reconstruction_dataset.__getitem__(i)

        prediction = process_patient_data(
            row,
            classification_dataset.pathologies,
            patient_classification_data,
            patient_reconstruction_data,
            classifier,
            reconstruction,
            transforms
        )
        predictions += prediction

    return predictions
