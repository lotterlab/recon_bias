import os
import shutil

# Define source and destination directories
source_directory = (
    "/Volumes/Lotter_Seagate/USCF-PDGM/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3"
)
destination_directory = (
    "/Volumes/Lotter_Seagate/USCF-PDGM/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3-reduced"
)

# Ensure the destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Loop through each patient's folder
for patient_folder in os.listdir(source_directory):
    print(f"Processing patient: {patient_folder}")
    patient_path = os.path.join(source_directory, patient_folder)

    # Only proceed if it's a directory
    if os.path.isdir(patient_path):
        # Initialize variables for the files
        flair_image = None
        tumor_segmentation = None

        # Search for the desired files within the patient's folder
        for file_name in os.listdir(patient_path):
            file_path = os.path.join(patient_path, file_name)

            # Look for the file that ends with 'FLAIR.nii.gz'
            if file_name.endswith("FLAIR.nii.gz") and "bias" not in file_name:
                flair_image = file_path

            # Look for the file that contains 'tumor_segmentation'
            if "tumor_segmentation" in file_name:
                tumor_segmentation = file_path

        # Ensure the patient folder exists in the destination
        patient_destination = os.path.join(destination_directory, patient_folder)
        if not os.path.exists(patient_destination):
            os.makedirs(patient_destination)

        # Copy the files if they were found
        if flair_image and os.path.exists(flair_image):
            shutil.copy(
                flair_image,
                os.path.join(patient_destination, os.path.basename(flair_image)),
            )

        if tumor_segmentation and os.path.exists(tumor_segmentation):
            shutil.copy(
                tumor_segmentation,
                os.path.join(patient_destination, os.path.basename(tumor_segmentation)),
            )

print("Files copied successfully.")
