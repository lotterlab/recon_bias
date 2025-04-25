# Bias in X-ray Reconstruction 
This branch has code to train models used in bias evaluation and mitigation of X-ray reconstruction models for CheXpert.

## Dependencies 
To install the necessary dependencies, run `pip install -r requirements.txt`. We use the CheXpert dataset for this project.

## Usage 
This branch combines different functionality. You can train classifiers, segmentation and reconstruction models. Each training run creates an output folder where model checkpoints and metrics are tracked. You can visualize the logs with `tensorboard --logdir=<output_dir/logs>`. 

### Models 
Parameters for reconstruction training are configured through YAML files. Examples can be found in the `configuration` folder. We do not train a classifier for CheXpert in this dataset, but use torchxrayvision. General configuration parameters include: 

- **output_dir**: Directory where all outputs will be saved.
- **output_name**: Name of the output model and logs.
- **num_epochs**: Number of epochs for training.
- **learning_rate**: Learning rate for the optimizer.
- **batch_size**: Batch size for the DataLoader.
- **num_train_samples** (optional): Number of training samples to use.
- **num_val_samples** (optional): Number of validation samples to use.
- **data_root**: Root directory containing the dataset.
- **seed** (optional): Random seed for reproducibility (default: 31415).
- **save_interval** (optional): Interval (in epochs) to save the model (default: 1).
- **early_stopping_patience** (optional): Number of epochs with no improvement to stop training early.

#### Reconstruction
To train a reconstruction model, run `python train_reconstruction.py -c <path to configuration file>`. 
Specific reconstruction parameters are: 
- **network_type** (optional): Type of network to use. Options:
  - `VGG` (default)
  - `UNet`
- **network_path** (optional): Path to a pre-trained network.
- **photon count** (optional): Amount of photons used for noising. The less the better.
