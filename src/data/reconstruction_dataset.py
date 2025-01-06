import pathlib
from typing import Callable, Optional

import numpy as np
from skimage.io import imread
from skimage.transform import radon, iradon, resize
import torch
from torch.utils.data import Dataset
import os
import polars as pl
import torchvision.transforms as transforms

from src.data.dataset import BaseDataset
from src.utils.transformations import min_max_slice_normalization

def apply_bowtie_filter(sinogram):
    """
    Apply a bowtie filter to the Sinogram.

    Parameters:
    - sinogram: 2D numpy array of the Sinogram.

    Returns:
    - filtered_sinogram: Sinogram with the bowtie filter applied.
    """
    rows, cols = sinogram.shape
    profile = np.linspace(0.05, 1.0, cols // 2)
    filter_profile = np.concatenate([profile[::-1], profile])[:cols]
    return sinogram * filter_profile[np.newaxis, :]


class ReconstructionDataset(BaseDataset):
    """Dataset to load X-ray images and process with bowtie filtering and noise."""

    def __init__(
        self,
        data_root: pathlib.Path,
        csv_path: pathlib.Path,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        evaluation=False,
        photon_count: float = 1e5,
    ):
        """
        Initialize the X-ray Dataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            csv_path (pathlib.Path): Path to the metadata CSV file.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            seed (Optional[int]): The seed for reproducibility.
            split (Optional[str]): The dataset split (train/test/val).
            photon_count (float): The photon count for Poisson noise simulation.
        """
        super().__init__(
            data_root=data_root,
            csv_path=csv_path,
            number_of_samples=number_of_samples,
            seed=seed,
            split=split,
            evaluation=evaluation,
        )
        self.photon_count = photon_count
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy to tensor
            transforms.Lambda(min_max_slice_normalization),
            transforms.Lambda(lambda x: x.float())  # Ensure float32
        ])
        print(f"Photon count: {self.photon_count}")

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Process the X-ray image with controllable noise levels.
        
        Parameters:
        - image: 2D numpy array of the original image
        - photon_count: Number of photons (lower = more noise)
        
        Returns:
        - reconstructed_image: Reconstructed image after processing
        - metrics: Dictionary with noise metrics
        """
        # Normalize input image to [0,1] range
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Steps 2-5: Same as before
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta, circle=False)
        filtered_sinogram = apply_bowtie_filter(sinogram)
        
        max_val = np.max(filtered_sinogram)
        scaled_sinogram = (filtered_sinogram / max_val) * self.photon_count
        noisy_sinogram = np.random.poisson(scaled_sinogram).astype(float)
        noisy_sinogram = (noisy_sinogram / self.photon_count) * max_val

        reconstructed_padded_image = iradon(noisy_sinogram, theta=theta, filter_name='hann', circle=False)
        reconstructed_image = resize(reconstructed_padded_image, image.shape, mode='reflect', anti_aliasing=True)
        
        # Normalize reconstructed image to [0,1] range
        reconstructed_image = (reconstructed_image - np.min(reconstructed_image)) / (np.max(reconstructed_image) - np.min(reconstructed_image))

        return reconstructed_image

    def _get_item_from_row(self, row) -> tuple:
        """
        Load and process an X-ray image.

        Parameters:
        - row: A single row of metadata.

        Returns:
        - processed_tensor: The processed (reconstructed) image tensor.
        - original_tensor: The original image tensor.
        """
        # Load the original image
        image_path = os.path.join(self.data_root, row["Path"])
        original_image = imread(image_path, as_gray=True).astype(np.float32)  # Convert to float32
        original_image = min_max_slice_normalization(original_image)
        original_image = resize(original_image, (256, 256), anti_aliasing=True)
        
        # Process the image before any resizing
        degraded_image = self.process_image(original_image)
        
        # Apply transforms to both images
        if self.transform:
            original_tensor = self.transform(original_image)
            degraded_tensor = self.transform(degraded_image)

        return degraded_tensor, original_tensor

    def get_random_sample(self):
        """
        Fetch a random sample from the dataset.

        Returns:
        - A single data sample (processed_tensor, original_tensor).
        """
        idx = np.random.randint(0, len(self.metadata))
        return self.__getitem__(idx)

    def get_patient_data(self, patient_id):
        """
        Fetch all slices for a given patient.

        Parameters:
        - patient_id: Unique identifier for the patient.

        Returns:
        - slices: List of tuples (processed_tensor, original_tensor) for all slices.
        """
        patient_slices_metadata = self.metadata.filter(
            pl.col("PatientID") == patient_id
        )
        patient_slices_metadata = patient_slices_metadata.sort("slice_id")

        # If no slices found, return empty
        if len(patient_slices_metadata) == 0:
            print(f"No slices found for PatientID={patient_id}")
            return []

        # Collect all slices for the patient
        slices = []
        for row_idx in range(len(patient_slices_metadata)):
            row = patient_slices_metadata.row(row_idx, named=True)
            slice_tensor, labels = self._get_item_from_row(row)
            slices.append((slice_tensor, labels))

        return slices