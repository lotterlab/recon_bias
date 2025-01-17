import os
from PIL import Image
import matplotlib.pyplot as plt

def combine_pathology_plots(base_dir, pathology):
    """
    Combine plots with overall on left and aligned group plots:
    
    [Overall] [AUROC Sex] [AUROC Race] [AUROC Age]
             [Fairness Sex] [Fairness Race] [Fairness Age]
    
    Args:
        base_dir: Base directory containing pathology subdirectories
        pathology: Name of the pathology to process
    """
    # Collect images by type
    auroc_images = {}
    fairness_images = {}
    
    # Define groups and files
    groups = ["Sex", "Race", "Age"]
    
    # Load images from pathology directory
    pathology_dir = os.path.join(base_dir, pathology)
    
    # Load overall AUROC
    overall_path = os.path.join(pathology_dir, f"{pathology}_overall.png")
    if os.path.exists(overall_path):
        overall_image = Image.open(overall_path)
    else:
        print(f"Missing overall AUROC file: {overall_path}")
        return
        
    # Load AUROC images
    for group in groups:
        path = os.path.join(pathology_dir, f"{pathology}_predictions_{group}.png")
        if os.path.exists(path):
            auroc_images[group] = Image.open(path)
        else:
            print(f"Missing AUROC file: {path}")
            return
    
    # Load Fairness images
    for group in groups:
        path = os.path.join(pathology_dir, f"{pathology}_fairness_{group}.png")
        if os.path.exists(path):
            fairness_images[group] = Image.open(path)
        else:
            print(f"Missing Fairness file: {path}")
            return
    
    # Calculate dimensions
    target_height = 400  # Height per row
    
    # Resize overall image to be double height (to match two rows)
    overall_scaling = (target_height * 2)/overall_image.size[1]
    overall_new_width = int(overall_image.size[0] * overall_scaling)
    overall_resized = overall_image.resize((overall_new_width, target_height * 2), Image.Resampling.LANCZOS)
    
    # Calculate group image dimensions
    group_widths = []
    for group in groups:
        auroc_width = auroc_images[group].size[0]
        fairness_width = fairness_images[group].size[0]
        group_widths.append(max(auroc_width, fairness_width))
    
    # Scale group images
    scaled_group_images = {
        'auroc': {},
        'fairness': {}
    }
    
    for i, group in enumerate(groups):
        # Scale AUROC
        auroc_scaling = target_height/auroc_images[group].size[1]
        auroc_new_width = int(group_widths[i] * auroc_scaling)
        scaled_group_images['auroc'][group] = auroc_images[group].resize(
            (auroc_new_width, target_height), 
            Image.Resampling.LANCZOS
        )
        
        # Scale Fairness
        fairness_scaling = target_height/fairness_images[group].size[1]
        fairness_new_width = int(group_widths[i] * fairness_scaling)
        scaled_group_images['fairness'][group] = fairness_images[group].resize(
            (fairness_new_width, target_height), 
            Image.Resampling.LANCZOS
        )
    
    # Calculate total dimensions
    group_total_width = sum(int(group_widths[i] * target_height/auroc_images[group].size[1]) 
                          for i, group in enumerate(groups))
    total_width = overall_new_width + group_total_width
    total_height = target_height * 2  # Two rows
    
    # Create the combined image
    combined_image = Image.new('RGB', (total_width, total_height), 'white')
    
    # Paste Overall image (left side, full height)
    combined_image.paste(overall_resized, (0, 0))
    
    # Paste AUROC group images (top row)
    x_offset = overall_new_width
    for group in groups:
        combined_image.paste(scaled_group_images['auroc'][group], (x_offset, 0))
        x_offset += scaled_group_images['auroc'][group].size[0]
    
    # Paste Fairness images (bottom row)
    x_offset = overall_new_width
    for group in groups:
        combined_image.paste(
            scaled_group_images['fairness'][group], 
            (x_offset, target_height)
        )
        x_offset += scaled_group_images['fairness'][group].size[0]
    
    # Save combined image
    output_dir = os.path.join(base_dir, "combined")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{pathology}_combined.png")
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

def combine_all_pathologies(base_dir):
    """Combine plots for all pathologies in the directory."""
    # Get all directories that are pathologies (exclude image_metrics and combined)
    pathologies = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))
                  and d not in ['image_metrics', 'combined']]
    
    for pathology in pathologies:
        print(f"Processing {pathology}...")
        combine_pathology_plots(base_dir, pathology)

def combine_image_metric_plots(base_dir, metric):
    """
    Combine plots for image metrics with overall on left and aligned group plots:
    
    [Overall] [Sex] [Race] [Age]
    
    Args:
        base_dir: Base directory containing 'image_metrics' subdirectory
        metric: Name of the metric to process (psnr, ssim, or nrmse)
    """
    # Collect images
    metric_images = {}
    
    # Define groups and their file naming
    groups = ["Sex", "Race", "Age"]
    
    # Load images from image_metrics directory
    metrics_dir = os.path.join(base_dir, "image_metrics")
    
    # Load overall separately - note the different filename format
    overall_path = os.path.join(metrics_dir, f"{metric}_overall.png")
    if os.path.exists(overall_path):
        overall_image = Image.open(overall_path)
    else:
        print(f"Missing overall metric file: {overall_path}")
        return
        
    for group in groups:
        # Note: files are named like "nrmse_Sex.png" not "nrmse_Sex.png"
        path = os.path.join(metrics_dir, f"{metric}_{group}.png")
        if os.path.exists(path):
            metric_images[group] = Image.open(path)
        else:
            print(f"Missing metric file: {path}")
            return
    
    # Calculate dimensions
    target_height = 400
    
    # Resize overall image
    overall_scaling = target_height/overall_image.size[1]
    overall_new_width = int(overall_image.size[0] * overall_scaling)
    overall_resized = overall_image.resize((overall_new_width, target_height), Image.Resampling.LANCZOS)
    
    # Calculate and scale group images
    group_widths = []
    scaled_group_images = {}
    
    for group in groups:
        # Get original width and calculate scaling
        group_width = metric_images[group].size[0]
        group_widths.append(group_width)
        
        # Scale image
        group_scaling = target_height/metric_images[group].size[1]
        group_new_width = int(group_width * group_scaling)
        scaled_group_images[group] = metric_images[group].resize(
            (group_new_width, target_height), 
            Image.Resampling.LANCZOS
        )
    
    # Calculate total dimensions
    group_total_width = sum(int(group_widths[i] * target_height/metric_images[group].size[1]) 
                          for i, group in enumerate(groups))
    total_width = overall_new_width + group_total_width
    
    # Create the combined image
    combined_image = Image.new('RGB', (total_width, target_height), 'white')
    
    # Paste Overall image (left side)
    combined_image.paste(overall_resized, (0, 0))
    
    # Paste group images
    x_offset = overall_new_width
    for group in groups:
        combined_image.paste(scaled_group_images[group], (x_offset, 0))
        x_offset += scaled_group_images[group].size[0]
    
    # Save combined image
    output_dir = os.path.join(base_dir, "combined")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{metric}_combined.png")
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

def combine_all_metrics(base_dir):
    """Combine plots for all image metrics."""
    metrics = ["psnr", "ssim", "nrmse"]
    
    for metric in metrics:
        print(f"Processing {metric}...")
        combine_image_metric_plots(base_dir, metric)

# Update main block to include both pathologies and metrics
if __name__ == "__main__":
    base_dir = "/lotterlab/users/matteo/code/recon_bias/output/evaluation-chex-graphs_20250115_110643"
    combine_all_pathologies(base_dir)
    combine_all_metrics(base_dir)