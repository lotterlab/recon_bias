import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt

# Fix the import paths by adding the root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

# Now use absolute imports instead of relative ones
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from code.recon_bias.src.model.reconstruction.chex_unet import UNet
from src.model.reconstruction.GAN import UnetGenerator
from chex_dataset import ChexDataset  # assuming this is in the same directory

def load_reconstruction_model(model_path, device) -> torch.nn.Module:
    model = ReconstructionModel()
    model = model.to(device)

    network = UNet()
    network = network.to(device)
    model.set_network(network)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.network.eval()

    return model

def evaluate_models_psnr(
    model_paths: list[str],
    data_root_A: str,
    data_root_B: str,
    csv_path_A: str,
    csv_path_B: str,
    output_path: str,
    batch_size: int = 32,
    unet: bool = False
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    
    # Load models
    for model_path, n_layers in model_paths:
        if not unet:
            model = UnetGenerator(1, 1, n_layers)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
        else:
            model = load_reconstruction_model(model_path, device)
            models.append(model)

    # Create dataset and dataloader
    dataset = ChexDataset(
        data_root_A=data_root_A,
        data_root_B=data_root_B,
        csv_path_A=csv_path_A,
        csv_path_B=csv_path_B
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Initialize PSNR lists
    psnr_models = [[] for _ in models]
    ssim_models = [[] for _ in models]

    # Create output directories
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if output_path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Create sample directory in the same location as output_path
    sample_dir = os.path.join(
        output_dir if output_dir else '.',  # Use current directory if no output_dir
        'sample_comparisons'
    )
    os.makedirs(sample_dir, exist_ok=True)
    
    print(f"Saving samples to: {sample_dir}")
    print(f"Saving results to: {output_path}")

    # Evaluate
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating models")):
            #degraded_images = batch['A'].to(device)
            original_images = batch['B'].to(device)
            degraded_images = original_images.clone()

            # Get reconstructions
            for i, model in enumerate(models):
                recon = model(degraded_images)

                # Calculate PSNR for each image in the batch
                for b in range(degraded_images.shape[0]):
                    # Get individual images from batch
                    original = original_images[b].squeeze().cpu().numpy()
                    reconstructed = recon[b].squeeze().cpu().numpy()

                    # Calculate PSNR for this individual image
                    psnr = peak_signal_noise_ratio(
                        original,
                        reconstructed,
                        data_range=1.0
                    )
                #ssim = structural_similarity(
                #    original_images.squeeze().cpu().numpy() ,
                #    recon.squeeze().cpu().numpy(),
                #    data_range=1.0
                #)

                    psnr_models[i].append(psnr)
                #ssim_models[i].append(ssim)

                    # Save first few samples with statistics
                if batch_idx < 0:
                    # Convert to numpy and get first image of batch
                    degraded = degraded_images[0].cpu().numpy()
                    original = original_images[0].cpu().numpy()
                    reconstructed = recon[0].cpu().numpy()

                    # Print statistics
                    print(f"\nSample {batch_idx}, Model {i}:")
                    print("Degraded image stats:")
                    print(f"Shape: {degraded.shape}")
                    print(f"Range: [{degraded.min():.3f}, {degraded.max():.3f}]")
                    print(f"Mean: {degraded.mean():.3f}")
                    print(f"Std: {degraded.std():.3f}")
                    
                    print("\nOriginal image stats:")
                    print(f"Shape: {original.shape}")
                    print(f"Range: [{original.min():.3f}, {original.max():.3f}]")
                    print(f"Mean: {original.mean():.3f}")
                    print(f"Std: {original.std():.3f}")
                    
                    print("\nReconstructed image stats:")
                    print(f"Shape: {reconstructed.shape}")
                    print(f"Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
                    print(f"Mean: {reconstructed.mean():.3f}")
                    print(f"Std: {reconstructed.std():.3f}")

                        # Create comparison plot
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                    axes[0].imshow(degraded[0], cmap='gray')
                    axes[0].set_title(f'Degraded\nμ={degraded.mean():.2f}, σ={degraded.std():.2f}')
                    axes[0].axis('off')
                        
                    axes[1].imshow(reconstructed[0], cmap='gray')
                    axes[1].set_title(f'Reconstructed\nμ={reconstructed.mean():.2f}, σ={reconstructed.std():.2f}\nPSNR: {psnr:.2f}')
                    axes[1].axis('off')
                        
                    axes[2].imshow(original[0], cmap='gray')
                    axes[2].set_title(f'Original\nμ={original.mean():.2f}, σ={original.std():.2f}')
                    axes[2].axis('off')
                        
                    plt.tight_layout()
                    plt.savefig(os.path.join(sample_dir, f'sample_{batch_idx}_model_{i}.png'))
                    plt.close()

    # Save results
    with open(output_path, 'w') as f:
        for i, model in enumerate(models):
            avg_psnr = np.mean(psnr_models[i])
            f.write(f"Model {i} average PSNR: {avg_psnr:.2f}\n")
            print(f"Model {i} average PSNR: {avg_psnr:.2f}, based on {len(psnr_models[i])} samples")
            #avg_ssim = np.mean(ssim_models[i])
            #f.write(f"Model {i} average SSIM: {avg_ssim:.2f}\n")
            #print(f"Model {i} average SSIM: {avg_ssim:.2f}, based on {len(psnr_models[i])} samples")

if __name__ == "__main__":
    # Example usage with correct absolute paths
    evaluate_models_psnr(
        model_paths=[
            ("/lotterlab/users/matteo/duplicate_code/recon_bias/output/unet-test_20250129_185259/checkpoints/unet-test_20250129_185259_epoch_20_best.pth", 8),
        ],
        data_root_A="/lotterlab/project_data/CheXpert_noise",
        data_root_B="/lotterlab/datasets/",
        csv_path_A="/lotterlab/project_data/CheXpert_noise/metadata_photon_3000.csv",
        csv_path_B="/lotterlab/users/matteo/data/CheXpert/metadata.csv",
        output_path="psnr_results.txt",
        batch_size=32, 
        unet=True
    )