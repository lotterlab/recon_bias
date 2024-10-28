import argparse
import os
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


# Function to check if a folder contains reconstruction results
def is_reconstruction_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if any(metric in file_name.lower() for metric in ["psnr", "nrmse", "ssim"]):
            return True
    return False


# Function to check if a folder contains Grad-CAM results
def is_gradcam_folder(folder_path):
    # Assume any non-empty folder under the main directory that matches the Grad-CAM pattern
    for file_name in os.listdir(folder_path):
        if "Sex_" in file_name and "Age_" in file_name:
            return True
    return False


# Function to find image files for classifier results based on folder structure
def get_classifier_images(classifier_folder):
    categories = ["predictions", "score", "performance"]
    images = {}

    # Iterate over each subfolder, treating each as a group
    group_folders = [f.name for f in os.scandir(classifier_folder) if f.is_dir()]
    for group in group_folders:
        group_path = os.path.join(classifier_folder, group)
        images[group] = {}

        # Search for images in each category
        for category in categories:
            base_image = None
            significance_image = None

            # Iterate over files in the group folder
            for file_name in os.listdir(group_path):
                if (
                    category in file_name.lower()
                    and "significance" not in file_name.lower()
                ):
                    base_image = os.path.join(group_path, file_name)
                elif (
                    "significance" in file_name.lower()
                    and category in file_name.lower()
                ):
                    significance_image = os.path.join(group_path, file_name)

            images[group][category] = {
                "base": base_image,
                "significance": significance_image,
            }

    return images


# Function to find image files for reconstruction results based on filename suffixes
def get_reconstruction_images(reconstruction_folder):
    metrics = ["psnr", "nrmse", "ssim"]
    images = {}

    for file_name in os.listdir(reconstruction_folder):
        # Only consider PNG files
        if file_name.endswith(".png"):
            for metric in metrics:
                if metric in file_name.lower():
                    group_suffix = file_name.split(f"{metric}_")[1].split(".png")[0]

                    if group_suffix not in images:
                        images[group_suffix] = {}

                    # Identify if it's a base or significance image
                    if "significance" in file_name.lower():
                        images[group_suffix][metric] = {
                            "significance": os.path.join(
                                reconstruction_folder, file_name
                            )
                        }
                    else:
                        images[group_suffix][metric] = {
                            "base": os.path.join(reconstruction_folder, file_name)
                        }

    # Ensure every metric (psnr, nrmse, ssim) has a base and significance entry
    for group, metrics_dict in images.items():
        for metric in metrics:
            if metric not in metrics_dict:
                metrics_dict[metric] = {"base": None, "significance": None}
            else:
                if "base" not in metrics_dict[metric]:
                    metrics_dict[metric]["base"] = None
                if "significance" not in metrics_dict[metric]:
                    metrics_dict[metric]["significance"] = None

    return images


# Function to find image files for Grad-CAM results based on filename patterns
def get_gradcam_images(gradcam_folder):
    images = {
        "male_age_leq_58": [],
        "male_age_gt_58": [],
        "female_age_leq_58": [],
        "female_age_gt_58": [],
    }

    # Define a regular expression pattern to match 'Age_x' where x is a number
    age_pattern = re.compile(r"Age_(\d+)")

    # Iterate over files in the Grad-CAM folder
    for file_name in os.listdir(gradcam_folder):
        if file_name.endswith(".png"):
            # Determine gender based on the presence of 'Sex_M' or 'Sex_F' in the filename
            gender = "male" if "Sex_M" in file_name else "female"

            # Search for the age pattern in the filename
            age_match = age_pattern.search(file_name)

            if age_match:
                # Extract the age as an integer
                age = int(age_match.group(1))
                age_group = "age_leq_58" if age <= 58 else "age_gt_58"

                # Form the key based on gender and age
                key = f"{gender}_{age_group}"
                images[key].append(os.path.join(gradcam_folder, file_name))

    return images


# Function to create plots for classifier results
def create_classifier_plots(images, classifier_name, pdf):
    create_matrix_plot(
        images,
        classifier_name,
        pdf,
        ["predictions", "score", "performance"],
        "Classifier",
    )


# Function to create plots for reconstruction results
def create_reconstruction_plots(images, reconstruction_name, pdf):
    create_matrix_plot(
        images, reconstruction_name, pdf, ["psnr", "nrmse", "ssim"], "Reconstruction"
    )


# General function to create matrix plots for both classifiers and reconstruction results
def create_matrix_plot(images, name, pdf, categories, title_prefix):
    num_groups = len(images)
    num_categories = len(categories)

    if num_groups == 0 or num_categories == 0:
        return  # Skip if there are no images

    fig, axes = plt.subplots(
        num_categories * 2, num_groups, figsize=(num_groups * 4, num_categories * 6)
    )

    # Handle the case where axes might be 1D if num_groups or num_categories is 1
    if num_groups == 1:
        axes = axes.reshape((num_categories * 2, 1))
    if num_categories == 1:
        axes = axes.reshape((2, num_groups))

    col_names = list(images.keys())

    for i, category in enumerate(categories):
        for j, col_name in enumerate(col_names):
            base_image_path = images[col_name][category]["base"]
            significance_image_path = images[col_name][category]["significance"]

            if base_image_path and os.path.exists(base_image_path):
                base_image = Image.open(base_image_path)
                axes[i * 2, j].imshow(base_image)
                axes[i * 2, j].axis("off")
            else:
                axes[i * 2, j].axis("off")

            if significance_image_path and os.path.exists(significance_image_path):
                significance_image = Image.open(significance_image_path)
                axes[i * 2 + 1, j].imshow(significance_image)
                axes[i * 2 + 1, j].axis("off")

            if i == 0:
                axes[i * 2, j].set_title(col_name, fontsize=14)
            if j == 0:
                axes[i * 2, j].set_ylabel(category, fontsize=14, labelpad=40)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle(f"{title_prefix}: {name}", fontsize=16)
    pdf.savefig(fig, dpi=600)
    plt.close()


# Function to create plots for Grad-CAM results
def create_gradcam_plots(images, classifier_name, pdf):
    categories = [
        "male_age_leq_58",
        "male_age_gt_58",
        "female_age_leq_58",
        "female_age_gt_58",
    ]
    num_categories = len(categories)

    # Determine the maximum number of images in any category to set up the figure size dynamically
    max_images_per_category = max(len(images[category]) for category in categories)

    # Set up the figure with one row per category
    fig, axes = plt.subplots(
        nrows=num_categories,
        ncols=max_images_per_category,
        figsize=(max_images_per_category * 4, num_categories * 4),
        gridspec_kw={"hspace": 0.4, "wspace": 0.4},
    )

    # Handle cases where there is only one column or one row
    if max_images_per_category == 1:
        axes = axes.reshape((num_categories, 1))
    elif num_categories == 1:
        axes = axes.reshape((1, max_images_per_category))

    for i, category in enumerate(categories):
        images_in_category = images[category]
        for j in range(max_images_per_category):
            if j < len(images_in_category):
                image_path = images_in_category[j]
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    axes[i, j].imshow(image)
                    axes[i, j].axis("off")
                else:
                    axes[i, j].axis("off")
            else:
                # Turn off axes for empty columns in the row
                axes[i, j].axis("off")

        # Set the row title for each category
        category_label = (
            category.replace("male", "Male")
            .replace("female", "Female")
            .replace("age_leq_58", "Age ≤ 58")
            .replace("age_gt_58", "Age > 58")
        )
        axes[i, 0].set_ylabel(category_label, fontsize=14, labelpad=20)

    plt.suptitle(f"Grad-CAM: {classifier_name}", fontsize=16)
    pdf.savefig(fig, dpi=600)
    plt.close()


# Main function to process and generate PDF
def generate_pdf(main_folder):
    pdf_filename = f"{os.path.basename(main_folder)}_combined_plots.pdf"
    pdf_path = os.path.join(main_folder, pdf_filename)

    with PdfPages(pdf_path) as pdf:
        pdf_has_content = False

        for subfolder in [f for f in os.scandir(main_folder) if f.is_dir()]:
            subfolder_path = subfolder.path

            if is_reconstruction_folder(subfolder_path):
                images = get_reconstruction_images(subfolder_path)
                if images:
                    create_reconstruction_plots(images, subfolder.name, pdf)
                    pdf_has_content = True
            elif is_gradcam_folder(subfolder_path):
                images = get_gradcam_images(subfolder_path)
                if images:
                    create_gradcam_plots(images, subfolder.name, pdf)
                    pdf_has_content = True
            else:
                images = get_classifier_images(subfolder_path)
                if images:
                    create_classifier_plots(images, subfolder.name, pdf)
                    pdf_has_content = True

        if not pdf_has_content:
            print("No images found to add to the PDF. No file will be created.")
            return

    print(f"PDF created: {pdf_filename}")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate combined PDF plots for classifiers, reconstruction, and Grad-CAM results."
    )
    parser.add_argument(
        "main_folder",
        type=str,
        help="Path to the main folder containing all subfolders",
    )
    args = parser.parse_args()

    generate_pdf(args.main_folder)
