import csv
import os
import random

import nibabel as nib
import numpy as np
import pandas as pd
import torch


# Function to extract patient ID from folder name
def extract_metadata_patient_id(folder_name):
    base_name = folder_name.split("_")[0]
    patient_id = "UCSF-PDGM-" + base_name[-3:]
    return patient_id


def extract_folder_patient_id(folder_name):
    split = folder_name.split("_")
    base_name = split[0]

    if len(split) == 3:
        base_name += "_" + split[1]

    return base_name


# Function to load patient metadata from a separate metadata file
def load_patient_metadata(patient_id, metadata_file):
    """
    Load patient-specific metadata from a metadata CSV file.

    Parameters:
    - patient_id: The patient's folder ID (to look up in the metadata file)
    - metadata_file: The CSV file containing patient metadata

    Returns:
    - A dictionary containing the patient's metadata (sex, age, WHO CNS grade, etc.)
    """
    with open(metadata_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ID"] == patient_id:
                return {
                    "sex": row["Sex"],
                    "age_at_mri": row["Age at MRI"],
                    "who_cns_grade": row["WHO CNS Grade"],
                    "final_diagnosis": row["Final pathologic diagnosis (WHO 2021)"],
                    "alive": row["1-dead 0-alive"],
                    "os": row["OS"],
                }
    return None  # If patient metadata not found


# Function to extract slice-level metadata for a patient
def process_patient_folder(
    parent_dir, patient_folder, metadata_file, split, output_file
):
    """
    Process a patient folder and extract metadata for each slice.

    Parameters:
    - patient_folder: Path to the patient's folder containing the NIfTI files
    - metadata_file: CSV file containing patient-specific metadata
    - split: Train, val, test split
    - output_file: CSV file where the metadata will be saved
    """
    patient_dir = os.path.join(parent_dir, patient_folder)
    metadata_patient_id = extract_metadata_patient_id(patient_folder)
    patient_id = extract_folder_patient_id(patient_folder)

    # Load patient metadata
    patient_metadata = load_patient_metadata(metadata_patient_id, metadata_file)
    if not patient_metadata:
        print(f"Metadata for patient {metadata_patient_id} not found!")
        return

    # Dictionary to map modality to file suffix
    modalities = {
        "T1": "_T1.nii.gz",
        "T2": "_T2.nii.gz",
        "FLAIR": "_FLAIR.nii.gz",
        "segmentation": "_tumor_segmentation.nii.gz",
    }

    # Load the segmentation mask
    seg_file = os.path.join(patient_dir, patient_id + modalities["segmentation"])
    if not os.path.exists(seg_file):
        print(f"Segmentation file not found for {patient_id}!")
        return

    segmentation_img = nib.load(seg_file)
    segmentation_data = segmentation_img.get_fdata()

    # Process each modality
    for modality, suffix in modalities.items():
        if modality == "segmentation":
            continue  # Skip segmentation here, as it's handled slice by slice

        file_path = os.path.join(patient_dir, patient_id + suffix)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
            continue

        mri_img = nib.load(file_path)
        mri_data = mri_img.get_fdata()
        num_slices = mri_data.shape[2]  # Assume 3rd dimension is slices (H, W, D)

        if num_slices != segmentation_data.shape[2]:
            print(f"Number of slices mismatch for {patient_id}!")
            continue

        for slice_id in range(num_slices):
            slice_data = mri_data[:, :, slice_id]
            seg_slice_data = segmentation_data[:, :, slice_id]

            # Slice metadata
            width, height = slice_data.shape

            # Check if this slice contains specific tumor types based on segmentation labels
            edema_present = 1 in seg_slice_data
            non_enhancing_present = 2 in seg_slice_data
            enhancing_present = 4 in seg_slice_data

            # Write metadata to output CSV
            with open(output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        patient_folder
                        + "/"
                        + patient_id
                        + suffix,  # relative file path
                        metadata_patient_id,
                        slice_id,
                        width,
                        height,
                        patient_metadata["sex"], # 5
                        patient_metadata["age_at_mri"],
                        patient_metadata["who_cns_grade"],
                        patient_metadata["final_diagnosis"],
                        patient_metadata["alive"],
                        patient_metadata["os"],
                        split,
                        modality,
                        edema_present,
                        non_enhancing_present,
                        enhancing_present,
                    ]
                )


# Main function to process all patients
def process_all_patients(source_directory, metadata_file, output_file):
    """
    Process all patient folders and save the metadata for each slice in a CSV file.

    Parameters:
    - source_directory: Directory containing all patient folders
    - metadata_file: CSV file containing patient-specific metadata
    - output_file: CSV file to store the collected metadata
    - split: Data split (train/val/test)
    """
    # Initialize CSV with headers
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "file_path",
                "patient_id",
                "slice_id",
                "width",
                "height",
                "sex",
                "age_at_mri",
                "who_cns_grade",
                "final_diagnosis",
                "alive",
                "os",
                "split",
                "type",
                "edema",
                "non_enhancing",
                "enhancing",
            ]
        )

    # Process each patient folder
    patient_folders = [
        f
        for f in os.listdir(source_directory)
        if os.path.isdir(os.path.join(source_directory, f))
    ]

    random.shuffle(patient_folders)

    # Split patients: 70% train, 10% val, 20% test
    total_patients = len(patient_folders)
    train_split = int(0.7 * total_patients)
    val_split = int(0.1 * total_patients)

    train_patients = patient_folders[:train_split]
    val_patients = patient_folders[train_split : train_split + val_split]
    test_patients = patient_folders[train_split + val_split :]

    for patient_folder in patient_folders:
        print(f"Processing {patient_folder}...")
        split = (
            "train"
            if patient_folder in train_patients
            else "val" if patient_folder in val_patients else "test"
        )
        patient_path = os.path.join(source_directory, patient_folder)
        if os.path.isdir(patient_path):
            process_patient_folder(
                source_directory, patient_folder, metadata_file, split, output_file
            )


# Example usage:
source_directory = "/homes9/matteow/data/UCSF-PDGM"
metadata_file = "/homes9/matteow/data/UCSF-PDGM/UCSF-PDGM-metadata.csv"
output_file = "/homes9/matteow/data/UCSF-PDGM/metadata.csv"

# Process all patients and generate the metadata file
process_all_patients(source_directory, metadata_file, output_file)
