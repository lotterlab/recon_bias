import os
import nibabel as nib
import numpy as np
import torch
import shutil
from fastmri import fft2c, ifft2c, tensor_to_complex_np

# Function to load NIfTI data as a PyTorch tensor
def load_nifti_gz_as_tensor(file_path):
    """
    Load a NIfTI file and convert to a PyTorch tensor.
    
    Parameters:
    - file_path: Path to the .nii.gz file
    
    Returns:
    - image_tensor: The image data as a PyTorch tensor
    """
    img = nib.load(file_path)
    image_data = img.get_fdata()
    image_tensor = torch.from_numpy(image_data).unsqueeze(0).float()  # Shape: [1, H, W, D]
    return image_tensor, img.affine

# Function to save the undersampled image as NIfTI
def save_nifti_gz_from_tensor(tensor, affine, output_path):
    """
    Save a PyTorch tensor as a NIfTI file (.nii.gz).
    
    Parameters:
    - tensor: The image tensor to save (as numpy array).
    - affine: The affine matrix for the NIfTI image.
    - output_path: Path where the .nii.gz file will be saved.
    """
    image_np = tensor.squeeze().numpy()
    img = nib.Nifti1Image(image_np, affine)
    nib.save(img, output_path)

# Radial mask function
def create_radial_mask(shape, num_rays=60):
    """
    Create a radial mask for undersampling k-space.
    
    Parameters:
    - shape: The shape of the mask (H, W)
    - num_rays: Number of radial lines in the mask
    
    Returns:
    - mask: A radial mask with values 0 (masked) and 1 (unmasked)
    """
    H, W = shape
    center = (H // 2, W // 2)
    mask = np.zeros((H, W), dtype=np.float32)

    # Define angles for rays, covering full 360 degrees
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    for angle in angles:
        line_x = np.cos(angle)
        line_y = np.sin(angle)
        for r in range(max(H, W) // 2):
            x = int(center[1] + r * line_x)
            y = int(center[0] + r * line_y)
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = 1
    return mask

# Function to apply undersampling and return a real-valued tensor using magnitude (absolute value)
def undersample_image_with_radial_mask(image_tensor):
    """
    Apply a radial undersampling mask to an MRI image and return a real-valued tensor based on the magnitude.
    
    Parameters:
    - image_tensor: The 3D PyTorch tensor of image data
    
    Returns:
    - undersampled_image_magnitude: The reconstructed image after undersampling with magnitude (absolute value)
    """
    slices = image_tensor.shape[-1]  # Get the number of slices
    undersampled_images = []
    
    # Apply undersampling to each slice
    for slice_index in range(slices):
        slice_2d = image_tensor[0, :, :, slice_index]  # Get the 2D slice
        complex_slice = torch.stack((slice_2d, torch.zeros_like(slice_2d)), dim=-1)  # Make complex
        
        # Fourier transform to k-space
        kspace = fft2c(complex_slice)
        
        # Apply radial mask
        H, W = kspace.shape[:2]
        radial_mask = create_radial_mask((H, W))
        radial_mask = torch.from_numpy(radial_mask).to(kspace.device).unsqueeze(-1)
        undersampled_kspace = kspace * radial_mask

        # Perform inverse Fourier transform to reconstruct the image
        undersampled_image = ifft2c(undersampled_kspace)

        # Convert to numpy format
        undersampled_image_np = tensor_to_complex_np(undersampled_image)

        # Calculate the magnitude (absolute value) of the complex image
        magnitude = np.abs(undersampled_image_np)
        
        # Append the magnitude of each slice to the list
        undersampled_images.append(magnitude)

    # Stack all slices back into a 3D volume
    return torch.from_numpy(np.stack(undersampled_images, axis=-1)).float()  # Final tensor with real magnitude values

# Script to process all patients and create undersampled images
source_directory = "/homes9/matteow/data/UCSF-PDGM"

# Process each patient
for patient_folder in os.listdir(source_directory):
    print(f"Processing patient: {patient_folder}")
    patient_path = os.path.join(source_directory, patient_folder)

    if os.path.isdir(patient_path):
        file_endings = ["FLAIR.nii.gz", "T1.nii.gz", "T2.nii.gz"]
        undersampled_file_endings = ["FLAIR_undersampled.nii.gz", "T1_undersampled.nii.gz", "T2_undersampled.nii.gz"]

        # Ensure patient folder exists in destination
        patient_destination = os.path.join(source_directory, patient_folder)
        if not os.path.exists(patient_destination):
            os.makedirs(patient_destination)

        already_processed = False
        for file_name in os.listdir(patient_path): 
            if any(file_ending in file_name for file_ending in undersampled_file_endings):
                already_processed = True
                break

        if already_processed:
            print("Patient already processed. Skipping...")
            continue
        
        for file_name in os.listdir(patient_path):
            if any(file_ending in file_name for file_ending in file_endings):
                file_path = os.path.join(patient_path, file_name)
                
                # Load the image
                image_tensor, affine = load_nifti_gz_as_tensor(file_path)

                # Apply undersampling
                undersampled_image_tensor = undersample_image_with_radial_mask(image_tensor)

                # Save the undersampled image
                output_file_name = file_name.replace(".nii.gz", "_undersampled.nii.gz")
                output_path = os.path.join(patient_destination, output_file_name)
                save_nifti_gz_from_tensor(undersampled_image_tensor, affine, output_path)

print("Undersampling and saving completed.")
