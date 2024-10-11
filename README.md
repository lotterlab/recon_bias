# Bias in MRI Reconstruction 
Understanding (demographic) bias and generalization in MRI reconstruction.

## Dependencies 
To install the necessary dependencies, run `pip install -r requirements.txt`. We use the UCSF-PDGM dataset for this project.

## Usage 
This repository combines different functionality. You can train classifiers and reconstruction models and evaluate them for bias according to various subgroups. Each training run creates an output folder where model checkpoints and metrics are tracked. You can visualize the logs with `tensorboard --logdir=<output_dir/logs>`.In addition, a script evaluates which slice aggregation method provides the best performance for classification. 

### Models 
Parameters for classifier and reconstruction training are configured through YAML files. Examples can be found in the `configuration` folder. Different classifiers exist for attributes, such as tumor type, grade, or survival days. For both classifier and reconstruction, different neural networks can be used. General configuration parameters include: 
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

#### Reconstruction
To train a reconstruction model, run `python train_reconstruction.py -c <path to configuration file>`. 
Specific reconstruction parameters are: 
- **network_type** (optional): Type of network to use. Options:
  - `VGG` (default)
  - `UNet`
- **network_path** (optional): Path to a pre-trained network.
- **sampling_mask** (optional): Type of sampling mask used (default: "radial").

### Evaluation 
After training classifiers and a reconstruction model, you can evaluate them using `evaluate_models.py -c <path to configuration file>`. The classifiers are evaluated on the ground truth and the reconstruction, automatically plotting performance metrics and significance. Parameters include: 
- **data_root**: Root directory containing the dataset.
- **output_dir**: Directory where all evaluation outputs will be saved.
- **output_name**: Name of the output evaluation results.
- **seed** (optional): Random seed for reproducibility (default: 42).
- **os_bins** (optional): Number of bins for overall survival classification (default: 4).
- **age_bins** (optional): Age bins for age classification (default: [0, 3, 18, 42, 67, 96]).

### Classifier Parameters:
- **classifiers.num_samples** (optional): Number of test samples for classification evaluation.
- **classifiers.lower_slice** (optional): Lower slice index for classification dataset.
- **classifiers.upper_slice** (optional): Upper slice index for classification dataset.
- **classifiers.pathology** (optional): Pathologies to include in classification evaluation.
- **classifiers.type** (optional): Type of classification data to use (default: "T2").
- **classifiers.results_path** (optional): Path to saved classification results. If provided, evaluation will use this file instead of recalculating results.
- **classifiers.models**: A list of classifiers to evaluate. Each classifier should have:
  - **type**: Type of classifier (e.g., `TTypeBCEClassifier`, `TGradeBCEClassifier`, `NLLSurvClassifier`, `AgeCEClassifier`, `GenderBCEClassifier`).
  - **network**: Type of network used (e.g., `ResNet18`, `ResNet50`).
  - **model_path**: Path to the pre-trained model file.

### Reconstruction Parameters:
- **reconstruction.num_samples** (optional): Number of test samples for reconstruction evaluation.
- **reconstruction.lower_slice** (optional): Lower slice index for reconstruction dataset.
- **reconstruction.upper_slice** (optional): Upper slice index for reconstruction dataset.
- **reconstruction.pathology** (optional): Pathologies to include in reconstruction evaluation.
- **reconstruction.sampling_mask** (optional): Sampling mask to use (default: "radial").
- **reconstruction.type** (optional): Type of reconstruction data to use (default: "T2").
- **reconstruction.results_path** (optional): Path to saved reconstruction results. If provided, evaluation will use this file instead of recalculating results.
- **reconstruction.model**: A list of reconstruction models to evaluate. Each model should have:
  - **network**: Type of network (e.g., `VGG`, `UNet`).
  - **model_path**: Path to the pre-trained model file.