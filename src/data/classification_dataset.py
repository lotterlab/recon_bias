import os
import pathlib
from typing import Callable, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

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
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root + "metadata.csv")
        self.fullysampled_tiles: torch.Tensor = torch.empty(0)
        self.undersampled_tiles: torch.Tensor = torch.empty(0)
        self.fullysampled_column_index = None
        self.undersampled_column_index = None
        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset.

        This is done by creating a DataFrame that contains the metadata and paths to the relevant files.

        Returns:
            None
        """
        self.metadata = self.metadata.filter(pl.col("split_type") == self.split)
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
        slice = np.load(row["file_path"])
        labels = extract_labels_from_row(row)

        if self.transform:
            slice = self.transform(slice)
        return slice, labels
