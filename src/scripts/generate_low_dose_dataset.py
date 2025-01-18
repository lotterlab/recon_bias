import os
import pathlib
import time
from typing import List, Optional
import multiprocessing as mp
from functools import partial

import polars as pl
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, radon, iradon
import torch

def apply_bowtie_filter(sinogram):
    """
    Apply a bowtie filter to the Sinogram.

    Parameters:
    - sinogram: 2D numpy array of the Sinogram.

    Returns:
    - filtered_sinogram: Sinogram with the bowtie filter applied.
    """
    rows, cols = sinogram.shape
    profile = np.linspace(0.05, 1.0, cols // 2)
    filter_profile = np.concatenate([profile[::-1], profile])[:cols]
    return sinogram * filter_profile[np.newaxis, :]

def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

def process_image(image: np.ndarray, photon_count: float, theta: np.ndarray) -> np.ndarray:
    """Process a single image with the specified photon count."""
    # Input image should already be normalized to [0,1] range and float32
    
    # Generate sinogram and apply processing
    sinogram = radon(image, theta=theta, circle=False)
    filtered_sinogram = apply_bowtie_filter(sinogram)

    # Apply noise based on photon count
    max_val = np.max(filtered_sinogram)
    scaled_sinogram = (filtered_sinogram / max_val) * photon_count
    noisy_sinogram = np.random.poisson(scaled_sinogram).astype(np.float32)
    noisy_sinogram = (noisy_sinogram / photon_count) * max_val

    # Reconstruct image
    reconstructed_padded_image = iradon(
        noisy_sinogram, theta=theta, filter_name="hann", circle=False
    )
    reconstructed_image = resize(
        reconstructed_padded_image, image.shape, mode="reflect", anti_aliasing=True
    )

    # Normalize reconstructed image to [0,1] range
    reconstructed_image = (reconstructed_image - np.min(reconstructed_image)) / (
        np.max(reconstructed_image) - np.min(reconstructed_image)
    )

    return reconstructed_image.astype(np.float32)

def process_single_item(args):
    """Process a single image for all photon counts."""
    source_path, target_paths, photon_counts = args
    try:
        # Read and preprocess original image
        image = imread(source_path, as_gray=True).astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = resize(image, (256, 256), anti_aliasing=True)
        
        # Pre-compute theta for all processing
        theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
        
        results = []
        for photon_count, target_path in zip(photon_counts, target_paths):
            # Process and save image
            processed_image = process_image(image, photon_count, theta)
            processed_image = (processed_image * 255).astype(np.uint8)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            imsave(target_path, processed_image)
            
        return True
    except Exception as e:
        print(f"Error processing {source_path}: {str(e)}")
        return False

def generate_low_dose_dataset(
    source_dir: str,
    target_base_dir: str,
    metadata_path: str,
    photon_counts: List[float],
    num_files: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    """
    Generate low dose CT dataset from source images.
    
    Args:
        source_dir: Path to source directory containing images
        target_base_dir: Base directory where processed images will be saved
        metadata_path: Path to metadata CSV file
        photon_counts: List of photon counts to simulate
        num_files: Optional number of files to process (if None, process all)
        num_workers: Optional number of worker processes (if None, use CPU count)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Read metadata
    df = pl.read_csv(metadata_path)
    
    # Create target directories for each photon count
    target_dirs = {}
    for photon_count in photon_counts:
        dir_name = f"low_dose_{int(photon_count)}"
        target_dir = os.path.join(target_base_dir, dir_name)
        os.makedirs(target_dir, exist_ok=True)
        target_dirs[photon_count] = dir_name

    # Limit number of files if specified
    if num_files is not None:
        df = df.head(num_files)
    
    # Create new dataframes for each photon count
    new_dfs = {count: df.clone() for count in photon_counts}
    
    # Prepare arguments for parallel processing
    process_args = []
    for row in df.iter_rows(named=True):
        source_path = os.path.join(source_dir, row["Path"])
        target_paths = []
        for photon_count in photon_counts:
            relative_path = row["Path"].split("/", 1)[1]  # Remove first directory
            new_path = os.path.join(target_dirs[photon_count], relative_path)
            target_path = os.path.join(target_base_dir, new_path)
            target_paths.append(target_path)
            
            # Update path in corresponding dataframe
            new_dfs[photon_count] = new_dfs[photon_count].with_columns(
                pl.when(pl.col("Path") == row["Path"])
                .then(pl.lit(new_path))
                .otherwise(pl.col("Path"))
                .alias("Path")
            )
        
        process_args.append((source_path, target_paths, photon_counts))
    
    # Process files in parallel
    start_time = time.time()
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, process_args),
            total=len(process_args),
            desc="Processing images"
        ))
    
    processed_count = sum(results)
    
    # Save new metadata files
    for photon_count, new_df in new_dfs.items():
        metadata_name = os.path.splitext(os.path.basename(metadata_path))[0]
        new_metadata_path = os.path.join(
            target_base_dir, 
            f"{metadata_name}_photon_{int(photon_count)}.csv"
        )
        new_df.write_csv(new_metadata_path)
    
    print(f"\nProcessing complete! Total files processed: {processed_count}")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    import argparse
    from ast import literal_eval
    
    parser = argparse.ArgumentParser(description="Generate low dose CT dataset")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory")
    parser.add_argument("--metadata", type=str, required=True, help="Metadata CSV path")
    parser.add_argument(
        "--photon_counts",
        type=str,
        required=True,
        help="Comma-separated list of photon counts (e.g., '1e4,1e5,1e6')"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to process (optional)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    # Convert photon counts string to list of floats
    photon_counts = [float(x.strip()) for x in args.photon_counts.split(",")]
    
    generate_low_dose_dataset(
        args.source_dir,
        args.target_dir,
        args.metadata,
        photon_counts,
        args.num_files,
        args.num_workers
    ) 