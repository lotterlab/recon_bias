import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch

# Define the directories
parent_directory = '/Users/matteowohlrapp/Documents/Uni/DFCI_L/dfci/dataset_exploration/USCF-PDGM/UCSF-PDGM-v3'
metadata_file = os.path.join(parent_directory, 'UCSF-PDGM-metadata.csv')
processed_directory = os.path.join(parent_directory, 'processed')

patient_folders = os.listdir(parent_directory)

# Create the processed directory if it doesn't exist
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

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
    """
    Normalize the MRI scan.

    Args:
        scan (torch.Tensor): The MRI scan to normalize.

    Returns:
        torch.Tensor: The normalized MRI scan.
    """
    scan_min = scan.min()
    scan_max = scan.max()
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

# Function to extract patient ID from folder name
def extract_patient_id(folder_name):
    # Example: UCSF-PDGM-1234_something -> UCSF-PDGM-234
    base_name = folder_name.split('_')[0]  # Get 'UCSF-PDGM-xxxx'
    patient_id = 'UCSF-PDGM-' + base_name[-3:]  # Take last 3 digits of xxxx
    return patient_id

# Function to process individual patients
def process_patient(patient_id, flair_path, seg_path):
    # Load FLAIR and Segmentation masks
    flair_img = nib.load(flair_path)
    seg_img = nib.load(seg_path)
    
    flair_data = flair_img.get_fdata()
    seg_data = seg_img.get_fdata()
    
    # Check for mismatch in number of slices
    if flair_data.shape[-1] != seg_data.shape[-1]:
        print(f"WARNING: Patient {patient_id} has different slice counts in FLAIR and segmentation.")
    
    for slice_id in range(flair_data.shape[-1]):
        flair_slice = flair_data[:, :, slice_id]
        seg_slice = seg_data[:, :, slice_id]
        
        # Save slices where segmentation has values
        if np.any(seg_slice > 0):
            # Convert the flair slice to a tensor and normalize it
            flair_slice_tensor = torch.tensor(flair_slice, dtype=torch.float32)
            normalized_flair_slice = normalize_scan(flair_slice_tensor)
            
            output_filename = f"{patient_id}_slice_{slice_id}.npy"
            output_filepath = os.path.join(processed_directory, output_filename)
            
            # Save the normalized FLAIR slice as a numpy array
            np.save(output_filepath, normalized_flair_slice.numpy())
            
            # Find the patient's metadata
            patient_meta = metadata[metadata['ID'] == patient_id].iloc[0]
            
            # Append processed data information for CSV
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
                'OS': patient_meta['OS']
            })

# Iterate through patient folders
for patient_folder in patient_folders:
    patient_dir = os.path.join(parent_directory, patient_folder)
    
    if os.path.isdir(patient_dir):
        print(f"Processing patient: {patient_folder}")
        # Extract patient ID from folder name
        patient_id = extract_patient_id(patient_folder)
        
        # Find FLAIR and tumor files by keyword
        flair_path = find_file_by_keyword(patient_dir, 'FLAIR.nii.gz')
        seg_path = find_file_by_keyword(patient_dir, 'tumor_segmentation.nii.gz')
        
        if flair_path and seg_path:
            process_patient(patient_id, flair_path, seg_path)
        else:
            print(f"WARNING: Missing FLAIR or segmentation for patient {patient_id}")

# Save processed data as CSV
processed_df = pd.DataFrame(processed_data)
output_csv_path = os.path.join(processed_directory, 'processed_data.csv')
processed_df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Saved processed slices and metadata to {output_csv_path}")
