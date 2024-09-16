import os
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
import warnings



# Define age bins and labels
AGE_BINS = [0, 3, 18, 42, 67, 96]
AGE_LABELS = ['0-2', '3-17', '18-41', '42-66', '67-96']

class Classifier(ABC):
    def __init__(self, model: torch.nn.Module):
        self.model = model
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



class BinaryClassifier(Classifier):
    def interpret_result(self, prediction: torch.Tensor) -> int:
        # Assuming the output is a single logit
        prob = torch.sigmoid(prediction).item()
        return int(prob >= 0.5)

    def aggregate_predictions(self, predictions: List[int]) -> int:
        # Majority vote
        return int(np.mean(predictions) >= 0.5)

class MultiClassClassifier(Classifier):
    def interpret_result(self, prediction: torch.Tensor) -> int:
        # Assuming the output is logits for each class
        return int(torch.argmax(prediction).item())

    def aggregate_predictions(self, predictions: List[int]) -> int:
        # Majority vote
        return int(pd.Series(predictions).mode()[0])

def load_metadata(metadata_path: str) -> pd.DataFrame:
    return pd.read_csv(metadata_path)

def process_patient_data(df: pd.DataFrame, classifier: Classifier, reconstruction_model: Optional[torch.nn.Module] = None) -> dict:
    patient_predictions = []
    patient_recon_predictions = []

    for _, row in df.iterrows():
        # Load the image
        image = np.load(row['file_path'])
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Add batch dimension

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
        'Sex': df['Sex'].iloc[0],
        'Age at MRI': df['Age at MRI'].iloc[0],
        'Ground Truth': df['Final pathologic diagnosis (WHO 2021)'].iloc[0],
        'Prediction': patient_result,
        'Reconstructed Prediction': patient_recon_result,
        'OS': df['OS'].iloc[0],
        '1-dead 0-alive': df['1-dead 0-alive'].iloc[0],
    }

    return patient_info

def bucket_data(data: pd.DataFrame) -> pd.DataFrame:
    # Assign age bins with labels
    data['Age Group'] = pd.cut(data['Age at MRI'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return data

def main():
    # Paths and models
    metadata_path = 'path/to/metadata.csv'
    classifier_model = torch.load('path/to/classifier_model.pt')
    #reconstruction_model = torch.load('path/to/reconstruction_model.pt')  # Set to None if not available
    reconstruction_model = None

    # Initialize classifier
    classifier = BinaryClassifier(model=classifier_model, name='binary_classifier')

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Group by patient
    patient_groups = metadata.groupby('patient_id')

    # Collect results
    results = []

    for patient_id, group in patient_groups:
        patient_info = process_patient_data(group, classifier, reconstruction_model)
        results.append(patient_info)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Bucket data
    results_df = bucket_data(results_df)

    # Reorder columns for clarity
    columns = ['patient_id', 'Sex', 'Age at MRI', 'Age Group', 'Ground Truth', 'Prediction', 'Reconstructed Prediction', 'OS', '1-dead 0-alive']
    results_df = results_df[columns]

    # Sort data by Sex and Age Group
    results_df.sort_values(by=['Sex', 'Age Group'], inplace=True)

    # Save results to CSV per classifier
    results_filename = f'{classifier.name}_results.csv'
    results_df.to_csv(results_filename, index=False)

    # Create pivot tables or matrices for analysis
    # For Ground Truth
    gt_pivot = results_df.pivot_table(index='Sex', columns='Age Group', values='Ground Truth', aggfunc=lambda x: x.value_counts().index[0], fill_value='')
    gt_pivot.to_csv(f'{classifier.name}_ground_truth_matrix.csv')

    # For Prediction
    pred_pivot = results_df.pivot_table(index='Sex', columns='Age Group', values='Prediction', aggfunc=lambda x: x.value_counts().index[0], fill_value='')
    pred_pivot.to_csv(f'{classifier.name}_prediction_matrix.csv')

    # If Reconstruction Prediction is available
    if 'Reconstructed Prediction' in results_df.columns and not results_df['Reconstructed Prediction'].isnull().all():
        recon_pred_pivot = results_df.pivot_table(index='Sex', columns='Age Group', values='Reconstructed Prediction', aggfunc=lambda x: x.value_counts().index[0], fill_value='')
        recon_pred_pivot.to_csv(f'{classifier.name}_reconstructed_prediction_matrix.csv')

if __name__ == '__main__':
    main()
