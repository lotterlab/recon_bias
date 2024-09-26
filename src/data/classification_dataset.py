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
        pathology: Optional[list] = ["edema","non_enhancing","enhancing"], 
        lower_slice = None,
        upper_slice = None, 
        evaluation = False
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
        self.lower_slice = lower_slice
        self.upper_slice = upper_slice
        self.evaluation = evaluation
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

        if self.lower_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") >= self.lower_slice)

        if self.upper_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") <= self.upper_slice)

        if self.number_of_samples and not self.evaluation:
            self.metadata = self.metadata.collect().sample(
                n=self.number_of_samples, seed=self.seed
            )
        else:
            self.metadata = self.metadata.collect()
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.row(idx, named=True)

        return self._get_item_from_row(row)

    def _get_item_from_row(self, row):
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

    def get_random_sample(self):
        idx = np.random.randint(0, len(self.metadata))
        return self.__getitem__(idx)
    
    def get_patient_data(self, patient_id): 
        patient_slices_metadata = self.metadata.filter(pl.col("patient_id") == patient_id).collect()

        # If no slices found, raise an error or return empty
        if len(patient_slices_metadata) == 0:
            print(f"No slices found for patient_id={patient_id}")
            return []

        # Collect all slices for the patient
        slices = []
        for row_idx in range(len(patient_slices_metadata)):
            row = patient_slices_metadata.row(row_idx, named=True)
            
            # Load the slice for this row directly
            slice_tensor, labels = self._get_item_from_row(row)
            slices.append((slice_tensor, labels))
        
        return slices


    

