import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import random

# Define the directories
parent_directory = '/Users/matteowohlrapp/Documents/Uni/DFCI_L/data/UCSF-PDGM'
metadata_file = os.path.join(parent_directory, 'UCSF-PDGM-metadata.csv')
processed_directory = os.path.join(parent_directory, 'processed')

# Get list of patient folders
patient_folders = [f for f in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, f))]

# Train, val, test directories
train_dir = os.path.join(processed_directory, 'train')
val_dir = os.path.join(processed_directory, 'val')
test_dir = os.path.join(processed_directory, 'test')

# Create processed directories if they don't exist
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load metadata
metadata = pd.read_csv(metadata_file)

# Initialize list to store processed slice info
processed_data = []

# Function to find files containing specific keywords in a directory
def find_file_by_keyword(directory, keyword):
    for filename in os.listdir(directory):
        if keyword in filename:
            return os.path.join(directory, filename)
    return None

# Normalization function
def normalize_scan(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

# Function to extract patient ID from folder name
def extract_patient_id(folder_name):
    base_name = folder_name.split('_')[0]
    patient_id = 'UCSF-PDGM-' + base_name[-3:]
    return patient_id

# Function to process individual patients
def process_patient(patient_id, flair_path, seg_path, save_dir, split_type):
    flair_img = nib.load(flair_path)
    seg_img = nib.load(seg_path)
    
    flair_data = flair_img.get_fdata()
    seg_data = seg_img.get_fdata()
    
    if flair_data.shape[-1] != seg_data.shape[-1]:
        print(f"WARNING: Patient {patient_id} has different slice counts in FLAIR and segmentation.")
    
    for slice_id in range(flair_data.shape[-1]):
        flair_slice = flair_data[:, :, slice_id]
        seg_slice = seg_data[:, :, slice_id]
        
        if np.any(seg_slice > 0):
            flair_slice_tensor = torch.tensor(flair_slice, dtype=torch.float32)
            normalized_flair_slice = normalize_scan(flair_slice_tensor)
            
            output_filename = f"{patient_id}_slice_{slice_id}.npy"
            output_filepath = os.path.join(save_dir, output_filename)
            
            np.save(output_filepath, normalized_flair_slice.numpy())
            
            patient_meta = metadata[metadata['ID'] == patient_id].iloc[0]
            
            processed_data.append({
                'file_path': output_filepath,
                'patient_id': patient_id,
                'slice_id': slice_id,
                'width': flair_slice.shape[0],
                'height': flair_slice.shape[1],
                'Sex': patient_meta['Sex'],
                'Age at MRI': patient_meta['Age at MRI'],
                'WHO CNS Grade': patient_meta['WHO CNS Grade'],
                'Final pathologic diagnosis (WHO 2021)': patient_meta['Final pathologic diagnosis (WHO 2021)'],
                '1-dead 0-alive': patient_meta['1-dead 0-alive'],
                'OS': patient_meta['OS'],
                'split_type': split_type  # Add the split type (train, val, test)
            })

# Shuffle the patient folders for random splitting
random.shuffle(patient_folders)

# Split patients: 70% train, 10% val, 20% test
total_patients = len(patient_folders)
train_split = int(0.7 * total_patients)
val_split = int(0.1 * total_patients)

train_patients = patient_folders[:train_split]
val_patients = patient_folders[train_split:train_split + val_split]
test_patients = patient_folders[train_split + val_split:]

# Process each patient and allocate slices to the respective set
for patient_folder in patient_folders:
    patient_dir = os.path.join(parent_directory, patient_folder)
    patient_id = extract_patient_id(patient_folder)
    
    flair_path = find_file_by_keyword(patient_dir, 'FLAIR.nii.gz')
    seg_path = find_file_by_keyword(patient_dir, 'tumor_segmentation.nii.gz')
    
    if flair_path and seg_path:
        if patient_folder in train_patients:
            print(f"Processing patient {patient_id} into train set")
            process_patient(patient_id, flair_path, seg_path, train_dir, 'train')
        elif patient_folder in val_patients:
            print(f"Processing patient {patient_id} into val set")
            process_patient(patient_id, flair_path, seg_path, val_dir, 'val')
        else:
            print(f"Processing patient {patient_id} into test set")
            process_patient(patient_id, flair_path, seg_path, test_dir, 'test')
    else:
        print(f"WARNING: Missing FLAIR or segmentation for patient {patient_id}")

# Save processed data as CSV
processed_df = pd.DataFrame(processed_data)
output_csv_path = os.path.join(processed_directory, 'metadata.csv')
processed_df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Saved processed slices and metadata to {output_csv_path}")
