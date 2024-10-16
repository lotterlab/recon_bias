import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to find image files based on keywords in filenames
def get_image_files(classifier_folder):
    group_folders = [f.name for f in os.scandir(classifier_folder) if f.is_dir()]
    categories = ['predictions', 'score', 'performance']
    images = {}
    
    for group in group_folders:
        group_path = os.path.join(classifier_folder, group)
        images[group] = {}

        # Find files dynamically based on keywords
        for category in categories:
            base_image = None
            significance_image = None

            # Loop over files in the group folder
            for file_name in os.listdir(group_path):
                if category in file_name and 'significance' not in file_name:
                    base_image = os.path.join(group_path, file_name)
                elif 'significance' in file_name and category in file_name:
                    significance_image = os.path.join(group_path, file_name)

            images[group][category] = {
                'base': base_image,
                'significance': significance_image
            }

    return images

# Function to create the matrix plot for a classifier
def create_matrix_plot(images, classifier_name, pdf):
    num_groups = len(images)
    num_categories = 3  # predictions, score, performance
    fig, axes = plt.subplots(num_categories * 2, num_groups, figsize=(num_groups * 4, num_categories * 6))  # Original figure size
    
    row_names = ['predictions', 'score', 'performance']
    col_names = list(images.keys())  # Dynamically get the group names (age, sex, sex_age, etc.)

    for i, row_name in enumerate(row_names):
        for j, col_name in enumerate(col_names):
            # Get base and significance images
            base_image_path = images[col_name][row_name]['base']
            significance_image_path = images[col_name][row_name]['significance']
            
            # Show base image on top row for this category
            if base_image_path and os.path.exists(base_image_path):
                base_image = Image.open(base_image_path)
                axes[i * 2, j].imshow(base_image)
                axes[i * 2, j].axis('off')  # Turn off axes
                axes[i * 2, j].spines[:].set_visible(False)  # Remove border
            else:
                axes[i * 2, j].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                axes[i * 2, j].axis('off')  # Hide axes if image not found

            # Show significance image on the row below
            if significance_image_path and os.path.exists(significance_image_path):
                significance_image = Image.open(significance_image_path)
                axes[i * 2 + 1, j].imshow(significance_image)
                axes[i * 2 + 1, j].axis('off')  # Turn off axes
                axes[i * 2 + 1, j].spines[:].set_visible(False)  # Remove border
            else:
                axes[i * 2 + 1, j].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                axes[i * 2 + 1, j].axis('off')

            # Set the column and row titles
            if i == 0:
                axes[i * 2, j].set_title(col_name, fontsize=14)
            if j == 0:
                axes[i * 2, j].set_ylabel(row_name, fontsize=14, labelpad=40)

    # Remove horizontal lines
    for ax in axes.flatten():
        ax.axhline(visible=False)

    # Add vertical lines to separate the groups (columns)
    for j in range(1, num_groups):
        plt.axvline(x=j - 0.5, color='black', linewidth=1.5)  # Line between groups

    # Adjust layout to make the gap smaller between significance and metric
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reducing space between plots

    plt.suptitle(f'Classifier: {classifier_name}', fontsize=16)
    pdf.savefig(fig, dpi=600)  # Save figure to PDF with high DPI for better quality
    plt.close()

# Main function to process all classifiers and generate PDF
def generate_pdf(output_folder, pdf_filename):
    classifiers = [f.name for f in os.scandir(output_folder) if f.is_dir()]  # Dynamically get classifier folders
    with PdfPages(output_folder + "/" + pdf_filename) as pdf:
        for classifier in classifiers:
            classifier_folder = os.path.join(output_folder, classifier)
            if os.path.exists(classifier_folder):
                images = get_image_files(classifier_folder)
                if images:  # Proceed only if images are found
                    create_matrix_plot(images, classifier, pdf)
            else:
                print(f'Folder for classifier {classifier} does not exist.')

    # After creating PDF, convert it to high-quality PNG
    convert_pdf_to_png(output_folder, pdf_filename)

def convert_pdf_to_png(output_folder, pdf_filename):
    from pdf2image import convert_from_path

    pdf_path = os.path.join(output_folder, pdf_filename)
    images = convert_from_path(pdf_path, dpi=500)  # High DPI for conversion
    
    for i, image in enumerate(images):
        png_filename = f"{pdf_filename[:-4]}_page_{i + 1}.png"  # PNG filename for each page
        image.save(os.path.join(output_folder, png_filename), 'PNG')
    
    print(f'PNG(s) created from PDF at {output_folder}')

# Set your output folder path
output_folder = './output/evaluation-rebalanced-plots_20241016_140431'  # Update with the actual path
pdf_filename = 'classification_plots.pdf'

# Run the script
generate_pdf(output_folder, pdf_filename)

print(f'PDF and PNG created: {pdf_filename}')
