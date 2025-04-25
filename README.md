# Bias in MRI Reconstruction 
This branch has code to train models used in bias evaluation and mitigation of MRI reconstruction models for UCSF-PDGM.

## Dependencies 
To install the necessary dependencies, run `pip install -r requirements.txt`. We use the UCSF-PDGM dataset for this project.

## Usage 
This branch combines different functionality. You can train classifiers, segmentation and reconstruction models. Each training run creates an output folder where model checkpoints and metrics are tracked. You can visualize the logs with `tensorboard --logdir=<output_dir/logs>`. 

### Models 
Parameters for classifier, segmention, and reconstruction training are configured through YAML files. Examples can be found in the `configuration` folder. Different classifiers exist for attributes, such as tumor type, grade, or survival days. For both classifier and reconstruction, different neural networks can be used. General configuration parameters include: 
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
- **type** (optional): Data type to use (default: "T2").
- **pathology** (optional): List of pathologies to consider (default: ["edema", "non_enhancing", "enhancing"]).
- **lower_slice** (optional): Lower slice index for the dataset.
- **upper_slice** (optional): Upper slice index for the dataset.

#### Classifier
To train a classifier, run `python train_classifier.py -c <path to configuration file>`. 
Specific classifier parameters are: 
- **classifier_type**: Type of classifier to use. Options include:
  - `TTypeBCEClassifier`
  - `TGradeBCEClassifier`
  - `NLLSurvClassifier`
  - `AgeCEClassifier`
  - `GenderBCEClassifier`
- **network_type** (optional): Type of ResNet network to use. Options:
  - `ResNet18` (default)
  - `ResNet50`
- **os_bins** (optional): Number of bins for overall survival classification (default: 4).
- **age_bins** (optional): Age bins for age classification (default: [0, 3, 18, 42, 67, 96]).
- **eps** (optional): Small constant for numerical stability in survival classification (default: 1e-8).
- **balancing** (optional): Specifies if the dataset should be rebalanced during training.

#### Reconstruction
To train a reconstruction model, run `python train_reconstruction.py -c <path to configuration file>`. 
Specific reconstruction parameters are: 
- **network_type** (optional): Type of network to use. Options:
  - `VGG` (default)
  - `UNet`
- **network_path** (optional): Path to a pre-trained network.
- **sampling_mask** (optional): Type of sampling mask used (default: "radial").

#### Segmentation
To train a segmentation model, run `python train_segmentation.py -c <path to configuration file>`. No specific parameters required.