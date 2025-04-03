import os
import pathlib
from typing import Optional
from skimage.transform import resize

import fastmri
import nibabel as nib
import numpy as np
import polars as pl
import torch
from fastmri import fft2c, ifft2c
from fastmri.data.subsample import RandomMaskFunc
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class UcsfDataset(Dataset):
    """Dataset for MRI reconstruction."""

    def __init__(self, opt, train=True):
        """Initialize this dataset class.

        Parameters:
            opt (dict) -- stores all the experiment flags
            train (bool) -- whether we are in training mode
        """
        self.data_root = pathlib.Path(opt["dataroot"])
        self.sampling_mask = opt.get("sampling_mask", "radial")
        self.number_of_samples = opt.get("number_of_samples", None)
        self.seed = opt.get("seed", 31415)
        self.type = opt.get("type", "FLAIR")
        self.pathology = opt.get("pathology", [])
        self.lower_slice = opt.get("lower_slice", 60)
        self.upper_slice = opt.get("upper_slice", 130)
        self.train = train
        self.num_rays = opt.get("num_rays", 140)
        print(f"num_rays: {self.num_rays}")

        # Load metadata
        self.metadata = self._load_metadata()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                min_max_slice_normalization,
                transforms.Resize((256, 256), antialias=True),
            ]
        )

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        import polars as pl

        metadata_file = self.data_root / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pl.read_csv(metadata_file)

        # Apply filters based on parameters
        if self.type:
            df = df.filter(pl.col("type") == self.type)
        if self.pathology:
            df = df.filter(pl.col("pathology").is_in(self.pathology))
        if self.lower_slice is not None:
            df = df.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice is not None:
            df = df.filter(pl.col("slice_id") <= self.upper_slice)

        # Filter for validation set first
        df = df.filter(pl.col("split") == "val")
        
        # For training, use first 90% of validation set
        if self.train:
            total_samples = len(df)
            train_size = int(0.9 * total_samples)
            df = df.slice(0, train_size)
        # For validation, use last 10% of validation set
        else:
            total_samples = len(df)
            train_size = int(0.9 * total_samples)
            df = df.slice(train_size, total_samples)
            
        # Sample if number_of_samples is specified
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df = df.sample(n=self.number_of_samples, seed=self.seed)

        return df

    def convert_to_complex(self, image_slice):
        """Convert a real-valued 2D image slice to complex format."""
        complex_tensor = torch.stack(
            (image_slice, torch.zeros_like(image_slice)), dim=-1
        )
        return complex_tensor

    def create_radial_mask(self, shape):
        """Create a radial mask for undersampling k-space."""
        H, W = shape
        center = (H // 2, W // 2)
        Y, X = np.ogrid[:H, :W]
        mask = np.zeros((H, W), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        for angle in angles:
            line_x = np.cos(angle)
            line_y = np.sin(angle)
            for r in range(max(H, W) // 2):
                x = int(center[1] + r * line_x)
                y = int(center[0] + r * line_y)
                if 0 <= x < W and 0 <= y < H:
                    mask[y, x] = 1
        return mask

    def apply_radial_mask_to_kspace(self, kspace):
        """Apply a radial mask to the k-space data."""
        H, W, _ = kspace.shape
        radial_mask = self.create_radial_mask((H, W))
        radial_mask = torch.from_numpy(radial_mask).to(kspace.device).unsqueeze(-1)
        return kspace * radial_mask

    def apply_linear_mask_to_kspace(self, kspace):
        """Apply a linear mask to the k-space data."""
        mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[6])
        mask = mask_func(kspace.shape, seed=None)[0]
        mask = mask.to(kspace.device).unsqueeze(-1)
        return kspace * mask

    def undersample_slice(self, slice_tensor):
        """Undersample an MRI slice using specified mask."""
        # Convert real slice to complex-valued tensor
        complex_slice = self.convert_to_complex(slice_tensor)

        # Transform to k-space
        kspace = fft2c(complex_slice)

        # Apply mask
        if self.sampling_mask == "radial":
            undersampled_kspace = self.apply_radial_mask_to_kspace(kspace)
        elif self.sampling_mask == "linear":
            undersampled_kspace = self.apply_linear_mask_to_kspace(kspace)
        else:
            raise ValueError(f"Unsupported sampling mask: {self.sampling_mask}")

        # Inverse transform
        undersampled_image = ifft2c(undersampled_kspace)
        return torch.abs(undersampled_image[..., 0])

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        # Convert index to integer to prevent TypeError
        index = int(index)
        row = self.metadata.row(index, named=True)

        # Load the original image
        nifti_img = nib.load(str(self.data_root / row["file_path"]))
        scan = nifti_img.get_fdata()
        slice_tensor = torch.from_numpy(scan[:, :, row["slice_id"]]).float()

        # Add channel dimension before applying transforms
        slice_tensor = slice_tensor.unsqueeze(0)  # Shape becomes [1, H, W]

        # Apply transforms
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        # Remove channel dimension after applying transforms
        slice_tensor = slice_tensor.squeeze(0)  # Shape becomes [H, W]

        # Create undersampled version
        undersampled_tensor = self.undersample_slice(slice_tensor)

        slice_tensor = slice_tensor.unsqueeze(0)
        undersampled_tensor = undersampled_tensor.unsqueeze(0)

        sex = float(0 if row["sex"] == "F" else 1)
        age = float(row["age_at_mri"] <= 58)

        grade = float(0 if int(row["who_cns_grade"]) <= 3 else 1)
        ttype = float(1 if row["final_diagnosis"] == "Glioblastoma, IDH-wildtype" else 0)
        
        # Return in CycleGAN format but using your file paths
        return undersampled_tensor, slice_tensor, torch.tensor([sex, age]), torch.tensor([ttype])

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.metadata)

    def compute_sample_weights(self):

            df = self.metadata.clone()  # or dataset.metadata if you don't mind modifying it
            
            # Define the sensitive columns. For example, here we use 'sex' and 'age_at_mri'
            sensitive_cols = ['sex', 'age_at_mri']
            
            # Compute group counts using Polars
            group_counts = df.group_by(sensitive_cols).agg(pl.count()).rename({'count': 'group_count'})
            
            # Join the group counts back to the original dataframe
            df = df.join(group_counts, on=sensitive_cols, how='left')
            
            # Compute the weight as the inverse of the group_count
            weights = 1.0 / df['group_count']
            
            # Return as a list
            return weights.to_list()