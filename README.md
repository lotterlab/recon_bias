# Bias Mitigation in Medical Image Reconstruction 
This branch contains code for bias mitigation in medical image reconstruction models using two datasets: UCSF-PDGM (brain MRI) and CheXpert (chest X-rays). The code implements a fairness constraint based on adversarial debiasing during the fine-tuning of reconstruction models.

## Dependencies 
To install the necessary dependencies, run `pip install -r requirements.txt`.

## Usage 
This branch focuses specifically on reconstruction model training with bias mitigation through an equalized odds fairness constraint. Each training run creates an output folder where model checkpoints and metrics are tracked. You can visualize the logs with `tensorboard --logdir=<output_dir/logs>`. 

### Configuration Parameters
The reconstruction training is configured through YAML files specific to each dataset. Examples can be found in the `configuration` folder. Configuration parameters include:

Common parameters:
- **output_dir**: Directory where all outputs will be saved
- **output_name**: Name of the output model and logs
- **save_interval**: Interval (in epochs) to save the model
- **dataset**: Dataset identifier ('ucsf' or 'chex')
- **fairness_lambda**: Weight for the fairness constraint in loss function
- **num_epochs**: Number of epochs for training
- **learning_rate**: Learning rate for the optimizer
- **batch_size**: Batch size for the DataLoader
- **early_stopping_patience**: Number of epochs with no improvement to stop training early
- **network_type**: Type of network to use ('UNet')
- **model_path**: Path to the pre-trained model
- **seed**: Random seed for reproducibility

Dataset-specific parameters:
For UCSF-PDGM:
- **dataroot**: Root directory for UCSF-PDGM dataset
- **sampling_mask**: Type of sampling mask (e.g., 'radial')
- **type**: Type of MRI sequence (e.g., 'FLAIR')
- **pathology**: List of pathologies to consider
- **lower_slice**: Lower slice index for the dataset
- **upper_slice**: Upper slice index for the dataset
- **num_rays**: Number of rays for radial sampling
- **classifiers**: List of classifiers used for fairness constraint
  - **name**: Name of the classifier (e.g., 'TGradeBCEClassifier', 'TTypeBCEClassifier')
  - **path**: Path to the pre-trained classifier model

For CheXpert:
- **csv_path_A**: Path to metadata CSV for noisy images
- **csv_path_B**: Path to metadata CSV for clean images
- **dataroot_A**: Directory containing noisy CheXpert images
- **dataroot_B**: Directory containing clean CheXpert images
- **classifier_path**: Path to the pre-trained classifier model for fairness constraint

### Training
To train a reconstruction model with bias mitigation:
```bash
python train_reconstruction.py -c configuration/unet_config_fairness_ucsf.yaml  # for UCSF-PDGM
# or
python train_reconstruction.py -c configuration/unet_config_fairness_chex.yaml  # for CheXpert
```