import numpy as np

def calculate_data_range(original, predicted):
    """
    Calculate the data range between two images.

    Args:
        original (np.ndarray): The original image.
        predicted (np.ndarray): The predicted image.

    Returns:
        float: The data range between the two images.
    """
    data_min = min(np.min(original), np.min(predicted))
    data_max = max(np.max(original), np.max(predicted))
    data_range = data_max - data_min

    return data_range