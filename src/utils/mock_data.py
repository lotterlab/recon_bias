import pandas as pd
import numpy as np


def get_mock_data(num_patients = 10):
    np.random.seed(42)

    # Define mock data in a dictionary
    mock_data = {
        'patient_id': [f'P{i:03d}' for i in range(1, num_patients + 1)],  # Patient IDs P001, P002, ...
        'sex': np.random.choice(['M', 'F'], num_patients),  # Randomly choose between Male and Female
        'age': np.random.randint(18, 80, num_patients),  # Random age between 18 and 80
        'TGradeBCEClassifier_gt': np.random.randint(0, 2, num_patients),  # Random ground truth labels (binary)
        'TGradeBCEClassifier_pred': np.random.randint(0, 2, num_patients),  # Random predictions (binary)
        'TGradeBCEClassifier_recon': np.random.randint(0, 2, num_patients),  # Random reconstruction predictions (binary)
        'TGradeBCEClassifier_gt_score': np.random.random(num_patients),  # Random scores (between 0 and 1)
        'TGradeBCEClassifier_pred_score': np.random.random(num_patients),  # Random predictions (binary)
        'TGradeBCEClassifier_recon_score': np.random.random(num_patients),  # Random reconstruction scores (0 to 1)
        'TTypeBCEClassifier_gt': np.random.randint(0, 2, num_patients),  # Random ground truth for TType (binary)
        'TTypeBCEClassifier_pred': np.random.randint(0, 2, num_patients),  # Random predictions for TType (binary)
        'TTypeBCEClassifier_recon': np.random.randint(0, 2, num_patients),  # Random recon predictions for TType
        'TTypeBCEClassifier_gt_score': np.random.random(num_patients),  # Random scores for TType (0 to 1)
        'TTypeBCEClassifier_pred_score': np.random.random(num_patients),  # Random scores for TType (0 to 1)
        'TTypeBCEClassifier_recon_score': np.random.random(num_patients)  # Random recon scores for TType (0 to 1)
    }

    # Convert the dictionary to a DataFrame
    mock_df = pd.DataFrame(mock_data)

    # Display the mock DataFrame
    return mock_df
