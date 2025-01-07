import os
import pathlib
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import polars as pl
import torch

from src.data.dataset import BaseDataset
from src.utils.labels import diagnosis_map, extract_labels_from_row, sex_map
from torchvision import transforms


class SegmentationDataset(BaseDataset):
    """Classification dataset to load MRI images"""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        type: Optional[str] = "T2",
        pathology: Optional[list] = ["edema", "non_enhancing", "enhancing"],
        lower_slice=None,
        upper_slice=None,
        evaluation=False,
        age_bins=[0, 68, 100],
    ):
        """
        Initialize the MRIDataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            seed (Optional[int]): The seed for reproducibility.
        """
        super().__init__(
            data_root=data_root,
            transform=transform,
            number_of_samples=number_of_samples,
            seed=seed,
            split=split,
            type=type,
            pathology=pathology,
            lower_slice=lower_slice,
            upper_slice=upper_slice,
            evaluation=evaluation,
            age_bins=age_bins,
        )

    def _get_item_from_row(self, row):
        nifti_img = nib.load(self.data_root + "/" + row["file_path"])

        # Extract the image data as a numpy array
        scan = nifti_img.get_fdata()
        slice = scan[:, :, row["slice_id"]]
        slice_tensor = torch.from_numpy(slice).float()

        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        segmentation_path = row["file_path"].replace(self.type, "tumor_segmentation")

        nifti_img = nib.load(self.data_root + "/" + segmentation_path)
        segmentation = nifti_img.get_fdata()
        segmentation_slice = segmentation[:, :, row["slice_id"]]
        segmentation_slice[segmentation_slice == 2] = 1
        segmentation_slice[segmentation_slice == 4] = 0

        segmentation_slice = torch.from_numpy(segmentation_slice).float().unsqueeze(0)
        segmentation_slice = transforms.functional.resize(segmentation_slice, (256, 256))
        slice_tensor = slice_tensor.unsqueeze(0)

        return slice_tensor, segmentation_slice

    def get_random_sample(self):
        idx = np.random.randint(0, len(self.metadata))
        return self.__getitem__(idx)

    def get_patient_data(self, patient_id):
        patient_slices_metadata = self.metadata.filter(
            pl.col("patient_id") == patient_id
        )
        patient_slices_metadata = patient_slices_metadata.sort("slice_id")

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
