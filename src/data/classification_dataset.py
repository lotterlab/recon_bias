import os
import pathlib
from typing import Callable, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
import nibabel as nib

from src.utils.labels import diagnosis_map, sex_map, extract_labels_from_row


class ClassificationDataset(Dataset):
    """Classification dataset to load MRI images"""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        type: Optional[str] = "T2",
        pathology: Optional[list] = ["edema","non_enhancing","enhancing"]
    ):
        """
        Initialize the MRIDataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            seed (Optional[int]): The seed for reproducibility.
        """
        self.data_root: pathlib.Path = data_root
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.split = split
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root + "/metadata.csv")
        self.type = type
        self.pathology = pathology
        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset.

        This is done by creating a DataFrame that contains the metadata and paths to the relevant files.

        Returns:
            None
        """
        self.metadata = self.metadata.filter(pl.col("split") == self.split)
        self.metadata = self.metadata.filter(pl.col("type") == self.type)

        # Filter by pathology OR
        if self.pathology and len(self.pathology) > 0:  # Ensure pathology list is not empty
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= (pl.col(path) == True)

            self.metadata = self.metadata.filter(pathology_filter)

        if self.number_of_samples:
            self.metadata = self.metadata.collect().sample(
                n=self.number_of_samples, seed=self.seed
            )
        else:
            self.metadata = self.metadata.collect()
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.row(idx, named=True)
        # Load np array from file
        nifti_img = nib.load(self.data_root + "/" + row["file_path"])
    
        # Extract the image data as a numpy array
        scan = nifti_img.get_fdata()
        slice = scan[:, :, row["slice_id"]]
        slice_tensor = torch.from_numpy(slice).float()
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        labels = extract_labels_from_row(row)

        slice_tensor = slice_tensor.unsqueeze(0)

        return slice_tensor, labels
