import os
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
import warnings
from src.model.classification.classification_model import ResNetClassifier



# Define age bins and labels
AGE_BINS = [0, 3, 18, 42, 67, 96]
AGE_LABELS = ['0-2', '3-17', '18-41', '42-66', '67-96']

class Classifier(ABC):
    def __init__(self, model):
        self.model = ResNetClassifier(num_classes=20)
        self.model.load_state_dict(model)
        self.model.eval()  # Set the model to evaluation mode

    @property
    @abstractmethod
    def name(self): 
        pass

    @property
    @abstractmethod
    def key(self): 
        pass

    @abstractmethod
    def interpret_result(self, prediction: torch.Tensor) -> int:
        """
        Interpret the raw output of the model and return a class label.
        """
        pass

    @abstractmethod
    def aggregate_predictions(self, predictions: List[int]) -> int:
        """
        Aggregate slice-level predictions to patient-level prediction.
        """
        pass


class SurvialClassifier(Classifier):
    def interpret_result(self, prediction: torch.Tensor) -> int:
        # Assuming the output is logits for each class
        return int(torch.argmax(prediction).item())

    def aggregate_predictions(self, predictions: List[int]) -> int:
        # Majority vote
        return int(pd.Series(predictions).mode()[0])

    def transform_gt(self, gt: int) -> int:
        return int(gt // 250)

    def key(self):
        return 'OS'
    
    def name(self):
        return 'SurvivalClassifier'

def load_metadata(metadata_path: str) -> pd.DataFrame:
    return pd.read_csv(metadata_path)

def process_patient_data(df: pd.DataFrame, classifier: Classifier, reconstruction_model: Optional[torch.nn.Module] = None) -> dict:
    patient_predictions = []
    patient_recon_predictions = []

    for _, row in df.iterrows():
        # Load the image
        image = np.load(row['file_path'])
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()  # Add batch dimension

        # Classify original image
        with torch.no_grad():
            output = classifier.model(image_tensor)
            pred = classifier.interpret_result(output)
            patient_predictions.append(pred)

            # If reconstruction model is provided
            if reconstruction_model is not None:
                recon_image = reconstruction_model(image_tensor)
                recon_output = classifier.model(recon_image)
                recon_pred = classifier.interpret_result(recon_output)
                patient_recon_predictions.append(recon_pred)

    # Aggregate predictions
    patient_result = classifier.aggregate_predictions(patient_predictions)
    patient_recon_result = None
    if patient_recon_predictions:
        patient_recon_result = classifier.aggregate_predictions(patient_recon_predictions)

    # Collect patient info
    patient_info = {
        'patient_id': df['patient_id'].iloc[0],
        'sex': df['Sex'].iloc[0],
        'age': df['Age at MRI'].iloc[0],
        'gt': classifier.transform_gt(df[classifier.key()].iloc[0]),
        'pred': patient_result,
    }
    if patient_recon_result is not None:
        patient_info['recon_pred'] = patient_re

    return patient_info

def main():
    # Paths and models
    metadata_path = '../../data/UCSF-PDGM/processed/metadata.csv'
    classifier_model = torch.load('./output/os_classification_model.pth')
    # reconstruction_model = torch.load('path/to/reconstruction_model.pt')
    reconstruction_model = None

    # Initialize classifier
    classifier = SurvialClassifier(model=classifier_model)

    # Load metadata
    metadata = load_metadata(metadata_path)
    metadata = metadata[metadata['split_type'] == 'test']

    # Group by patient
    patient_groups = metadata.groupby('patient_id')

    results = []

    for patient_id, group in patient_groups:
        print(f'Processing patient {patient_id} ...')
        patient_info = process_patient_data(group, classifier, reconstruction_model)
        results.append(patient_info)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save results 
    if not os.path.exists('./output/results'):
        os.makedirs('./output/results')
    results_df.to_csv(f'./output/results/{classifier.name()}_results.csv', index=False)

if __name__ == '__main__':
    main()
