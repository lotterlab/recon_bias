from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from datetime import datetime
from skimage.metrics import (
    peak_signal_noise_ratio,
    mean_squared_error,
    structural_similarity,
)


def save_images_with_difference(
    classification_image, reconstruction_image, save_dir=".", file_prefix="comparison"
):
    """
    Save classification, reconstruction, and difference images for comparison without overwriting previous files,
    and display min, max, mean, and std values below the classification and reconstruction images.

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
        classification_image = classification_image.squeeze().cpu().detach().numpy()
    if isinstance(reconstruction_image, torch.Tensor):
        reconstruction_image = reconstruction_image.squeeze().cpu().detach().numpy()

    # Ensure images are 2D for grayscale or 3D for RGB
    if classification_image.ndim == 3 and classification_image.shape[0] in [
        1,
        3,
    ]:  # Channels first
        classification_image = np.transpose(
            classification_image, (1, 2, 0)
        )  # Convert to H x W x C
    if reconstruction_image.ndim == 3 and reconstruction_image.shape[0] in [
        1,
        3,
    ]:  # Channels first
        reconstruction_image = np.transpose(
            reconstruction_image, (1, 2, 0)
        )  # Convert to H x W x C

    # Calculate the difference image
    difference_image = np.abs(classification_image - reconstruction_image)

    # Normalize the difference image for better visualization (optional)
    difference_image = (
        difference_image / np.max(difference_image)
        if np.max(difference_image) > 0
        else difference_image
    )

    # Compute statistics
    classification_stats = {
        "min": np.min(classification_image),
        "max": np.max(classification_image),
        "mean": np.mean(classification_image),
        "std": np.std(classification_image),
    }
    reconstruction_stats = {
        "min": np.min(reconstruction_image),
        "max": np.max(reconstruction_image),
        "mean": np.mean(reconstruction_image),
        "std": np.std(reconstruction_image),
    }

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_filename = f"{file_prefix}_{timestamp}.png"

    # Plot and save the images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Classification image
    axes[0].imshow(
        classification_image, cmap="gray" if classification_image.ndim == 2 else None
    )
    axes[0].set_title("Classification Image")
    axes[0].axis("off")
    axes[0].text(
        0.5,
        -0.1,
        f"Min: {classification_stats['min']:.2f}\nMax: {classification_stats['max']:.2f}\n"
        f"Mean: {classification_stats['mean']:.2f}\nStd: {classification_stats['std']:.2f}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # Reconstruction image
    axes[1].imshow(
        reconstruction_image, cmap="gray" if reconstruction_image.ndim == 2 else None
    )
    axes[1].set_title("Reconstruction Image")
    axes[1].axis("off")
    axes[1].text(
        0.5,
        -0.1,
        f"Min: {reconstruction_stats['min']:.2f}\nMax: {reconstruction_stats['max']:.2f}\n"
        f"Mean: {reconstruction_stats['mean']:.2f}\nStd: {reconstruction_stats['std']:.2f}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # Difference image
    axes[2].imshow(
        difference_image, cmap="gray" if difference_image.ndim == 2 else None
    )
    axes[2].set_title("Difference Image")
    axes[2].axis("off")

    # Save the comparison plot
    comparison_path = os.path.join(save_dir, unique_filename)
    plt.savefig(comparison_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison image at: {comparison_path}")


def process_patient_data(
    row, index, pathologies, classifier_model, reconstruction_models, device
):
    """Process each patient, returning a dictionary of results."""

    # csv fields: Path (key), patient_id, pathology, gt, pred, gt_recon
    classifier_dataset = classifier_model["dataset"]
    classifier = classifier_model["model"]
    result = {
        "Path": row["Path"],
        "patient_id": row["PatientID"],
        "Sex": row["Sex"],
        "Age": row["Age"],
        "Race": row["Mapped_Race"],
    }

    classifier_data = classifier_dataset.__getitem__(index)
    x_class = classifier_data["img"]
    x_class = torch.tensor(x_class, dtype=torch.float32).unsqueeze(0).to(device)
    y_class = classifier_data["lab"]
    y_class = torch.tensor(y_class, dtype=torch.float32).squeeze()

    # predictions on gt
    pred = classifier(x_class)
    pred = pred.squeeze(0)
    pred = torch.sigmoid(pred)

    for i, pathology in enumerate(pathologies):

        if (
            not np.isnan(row[pathology])
            and not np.isnan(y_class[i])
            and row[pathology] != y_class[i]
        ):
            print(
                f"Error: GT mismatch for pathology {pathology} at index {i} for row {row['Path']}: {row[pathology]} != {y_class[i]}"
            )
            temp_paths = [(i, row[i]) for _, i in enumerate(pathologies)]
            print(f"row: {temp_paths}, y_class: {y_class}")

        result[pathology] = float(row[pathology])
        result[f"{pathology}_class"] = float(pred[i])

    for reconstruction_info in reconstruction_models:
        reconstruction_model = reconstruction_info["model"]
        reconstruction_network = reconstruction_info["network"]
        photon_count = reconstruction_info["photon_count"]
        reconstruction_dataset = reconstruction_info["dataset"]
        reconstruction_data = reconstruction_dataset.__getitem__(index)

        x_recon, x_recon_gt = reconstruction_data
        x_recon = x_recon.unsqueeze(0).to(device)

        recon = reconstruction_model(x_recon)
        pred_recon = classifier(recon)
        pred_recon = pred_recon.squeeze(0)
        pred_recon = torch.sigmoid(pred_recon)

        for i, pathology in enumerate(pathologies):
            result[f"{pathology}_{reconstruction_network}_{photon_count}"] = float(
                pred_recon[i]
            )

        # calculate metrics
        result[f"{reconstruction_network}_{photon_count}_psnr"] = (
            peak_signal_noise_ratio(
                x_recon_gt.detach().numpy().squeeze(),
                recon.detach().numpy().squeeze(),
                data_range=1,
            )
        )
        result[f"{reconstruction_network}_{photon_count}_ssim"] = structural_similarity(
            x_recon_gt.detach().numpy().squeeze(),
            recon.detach().numpy().squeeze(),
            data_range=1,
        )
        result[f"{reconstruction_network}_{photon_count}_nrmse"] = mean_squared_error(
            x_recon_gt.detach().numpy().squeeze(), recon.detach().numpy().squeeze()
        )

    return result


def classifier_predictions(
    metadata,
    classifier_model,
    reconstruction_models,
    device,
    num_samples=None,
) -> List[dict]:
    """Process all patients in the metadata file."""
    predictions = []
    classifier_dataset = classifier_model["dataset"]

    index = 0
    # iterate all rows of metadata
    for i, row in metadata.iterrows():
        if num_samples is not None:
            if index >= num_samples:
                break
        index += 1
        if i % 100 == 0:
            print(f"Processing row {row['Path']} at index {i}/{len(metadata)}")

        prediction = process_patient_data(
            row,
            i,
            classifier_dataset.pathologies,
            classifier_model,
            reconstruction_models,
            device,
        )

        predictions += [prediction]

    return predictions
