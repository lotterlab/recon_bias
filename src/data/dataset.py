import pathlib
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.utils.labels import diagnosis_map, extract_labels_from_row, sex_map


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        data_root: pathlib.Path,
        csv_path: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        evaluation=False,
    ):
        self.data_root = data_root
        self.csv_path = csv_path
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.split = split
        self.evaluation = evaluation
        self.metadata = pd.read_csv(self.csv_path)
        self.data = self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset.

        This is done by creating a DataFrame that contains the metadata and paths to the relevant files.

        Returns:
            None
        """
        # Filter rows based on 'split' column
        self.metadata = self.metadata.loc[self.metadata["split"] == self.split]

        # Random sampling if needed
        if self.number_of_samples and not self.evaluation:
            self.metadata = self.metadata.sample(
                n=self.number_of_samples, random_state=self.seed
            )

        # Sort by 'Path' column
        self.metadata.sort_values(by="Path", inplace=True)

        # Reset the index
        self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        return self._get_item_from_row(row)

    @abstractmethod
    def _get_item_from_row(self, row):
        pass

    @abstractmethod
    def get_random_sample(self):
        pass

    @abstractmethod
    def get_patient_data(self, patient_id):
        pass

    def get_class_labels(self):
        """Returns the class labels for each sample in the dataset."""
        class_labels = [
            extract_labels_from_row(row, self.age_bins)
            for row in self.metadata.iter_rows(named=True)
        ]
        class_labels_tensor = torch.stack(class_labels)
        return class_labels_tensor


def create_balanced_sampler(dataset, classifier):
    """Creates a WeightedRandomSampler for balanced class sampling."""
    dataset_class_labels = dataset.get_class_labels()

    class_labels = classifier.target_transformation(dataset_class_labels)

    # Count occurrences of each class (assuming binary or multi-class)
    class_counts = np.bincount(class_labels)
    class_weights = 1.0 / class_counts

    # Create weights for each sample based on its class
    sample_weights = np.array([class_weights[int(label)] for label in class_labels])

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,  # Sample with replacement to ensure balanced sampling
    )

    return sampler
