import os
import pathlib
from typing import Optional
from skimage.transform import radon, iradon, resize
from skimage.io import imread
import numpy as np
import polars as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd

def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

class ChexDataset(Dataset):

    def __init__(self, config, train=True):
        super().__init__()
        self.data_root_A = pathlib.Path(config["dataroot_A"])
        self.data_root_B = pathlib.Path(config["dataroot_B"])
        self.csv_path_A = pathlib.Path(config["csv_path_A"])
        self.csv_path_B = pathlib.Path(config["csv_path_B"])
        self.number_of_samples = config["number_of_samples"] if "number_of_samples" in config else None
        self.seed = config["seed"] if "seed" in config else 31415
        self.train = train
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert numpy to tensor
                transforms.Lambda(min_max_slice_normalization),
                transforms.Lambda(lambda x: x.float()),  # Ensure float32
            ]
        )
        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            "No Finding"
        ]
        self.pathologies = sorted(self.pathologies)
        # Load metadata
        self.metadata_A, self.metadata_B = self._load_metadata()
        self.labels = self._process_labels(self.metadata_A)

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        if not self.csv_path_A.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_A}")
        if not self.csv_path_B.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_B}")
        
        df_A = pd.read_csv(self.csv_path_A)
        df_B = pd.read_csv(self.csv_path_B)

        # Filter for validation split first
        if self.train:
            df_A = df_A[df_A["split"] == "val_recon"]
            df_B = df_B[df_B["split"] == "val_recon"]
        else:
            df_A = df_A[df_A["split"] == "val_class"]
            df_B = df_B[df_B["split"] == "val_class"]
        
        # Get total number of validation samples
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df_A = df_A.sample(n=self.number_of_samples, random_state=self.seed)
            df_B = df_B.sample(n=self.number_of_samples, random_state=self.seed)
        
        return df_A, df_B

    def _process_labels(self, df):
        # First identify healthy cases
        healthy = df["No Finding"] == 1
        
        labels = []
        for pathology in self.pathologies:
            assert pathology in df.columns
            
            if pathology == "No Finding":
                # Handle NaN in No Finding when other pathologies exist
                for idx, row in df.iterrows():
                    if row["No Finding"] != row["No Finding"]:  # check for NaN
                        if (row[self.pathologies] == 1).sum():  # if any pathology present
                            df.loc[idx, "No Finding"] = 0
            elif pathology != "Support Devices":
                # If healthy, other pathologies (except Support Devices) must be 0
                df.loc[healthy, pathology] = 0
                
            mask = df[pathology]
            labels.append(mask.values)
        
        # Convert to numpy array and transpose to get samples x labels
        labels = np.asarray(labels).T
        labels = labels.astype(np.float32)
        
        # Convert -1 to NaN
        labels[labels == -1] = np.nan
        
        return torch.from_numpy(labels)

    def __getitem__(self, index):
        row_A = self.metadata_A.iloc[index]
        row_B = self.metadata_B.iloc[index]
        
        # Load the original image
        image_path_A = os.path.join(self.data_root_A, row_A["Path"])
        image_A = imread(image_path_A, as_gray=True).astype(np.float32)  # Convert to float32
        image_A = min_max_slice_normalization(image_A)
        image_A = torch.from_numpy(image_A).float().unsqueeze(0)
        image_path_B = os.path.join(self.data_root_B, row_B["Path"])
        image_B = imread(image_path_B, as_gray=True).astype(np.float32)  # Convert to float32
        image_B = min_max_slice_normalization(image_B)
        image_B = resize(image_B, (256, 256))
        image_B = torch.from_numpy(image_B).float().unsqueeze(0) 

        sex = float(0 if row_A["Sex"] == "F" else 1)  # Assuming binary F/M encoding
        age = float(row_A["Age"] <= 61)  # Already boolean, convert to float
        
        # Map race to numeric values
        race_mapping = {
            'Other': 0,
            'White': 1,
            'Black': 2,
            'Native Hawaiian or Other Pacific Islander': 3,
            'Asian': 4,
            'American Indian or Alaska Native': 5
        }
        race = float(race_mapping.get(row_A["Mapped_Race"], 0))  # Default to 'Other' if not found
        
        # Add protected attributes to the tensor
        protected_attrs = torch.tensor([sex, age, race])

        
        return image_A, image_B, protected_attrs, self.labels[index]

    def __len__(self):
        return len(self.metadata_A)