#!/usr/bin/env python3
import os
import subprocess
import time
import shutil
import yaml
import sys

# Define lambda values to test
lambda_values = [0.1, 0.316, 1, 3.16, 10]

# Path to the config template
config_template_path = "/lotterlab/users/matteo/code/recon_bias/configuration/unet_config_lambda_search.yaml"
temp_config_path = "/lotterlab/users/matteo/code/recon_bias/configuration/temp_lambda_config.yaml"

def create_config_with_lambda(lambda_value):
    """Create a temporary config file with the specified lambda value."""
    with open(config_template_path, 'r') as f:
        config_content = f.read()
    
    # Replace all occurrences of <lambda> with the current lambda value
    config_content = config_content.replace('<lambda>', str(lambda_value))
    
    with open(temp_config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created configuration with lambda = {lambda_value}")

def run_training():
    """Run the training script with the temporary config."""
    cmd = ["python", "train_reconstruction.py", "-c", temp_config_path]
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command and wait for it to complete
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Training completed with exit code: {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        return False

def main():
    print(f"Starting lambda search with values: {lambda_values}")
    
    for i, lambda_value in enumerate(lambda_values):
        print(f"\n[{i+1}/{len(lambda_values)}] Running training with lambda = {lambda_value}")
        
        # Create config with the current lambda value
        create_config_with_lambda(lambda_value)
        
        # Run training
        success = run_training()
        
        if not success:
            print(f"Training failed for lambda = {lambda_value}, stopping the search.")
            break
        
        # Small delay between runs
        if i < len(lambda_values) - 1:
            print("Waiting 5 seconds before starting the next run...")
            time.sleep(5)
    
    # Clean up the temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        print("Removed temporary config file")
    
    print("Lambda search completed!")

if __name__ == "__main__":
    main() 