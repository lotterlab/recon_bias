"""
Classifier wrappers for both training and evaluation.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from ..model_wrapper import ModelWrapper
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.utils.image_metrics import calculate_data_range
import matplotlib.pyplot as plt
import numpy as np


class ReconstructionModel(ModelWrapper):
    """
    Reconstruction base class.
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    @property
    def name(self):
        return "ReconstructionModel"
    
    def target_transformation(self, y):
        return y

    def criterion(self, x, y):
        return self.loss(x, y)

    def evaluation_performance_metric(self, x, y):
        return torch.tensor(0.0)
    
    def epoch_performance_metric(self, x, y):
        return torch.tensor(0.0)

    @property
    def performance_metric_name(self):
        return "n/a"
    
    @property
    def performance_metric_input_value(self):
        return "prediction"
    
    def save_snapshot(self, x, y, y_pred, path, device, epoch):
        # save image next to each other
        plt.clf()
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        x = x.squeeze()
        y = y.squeeze()
        y_pred = y_pred.squeeze()

        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        ax[0].imshow(x.squeeze(), cmap="gray")
        ax[0].set_title("Undersampled")
        ax[0].axis("off")
        ax[1].imshow(y.squeeze(), cmap="gray")
        ax[1].set_title("Original")
        ax[1].axis("off")
        ax[2].imshow(y_pred.squeeze(), cmap="gray")
        ax[2].set_title("Reconstruction")
        ax[2].axis("off")
        ax[3].imshow(
            np.abs(y - y_pred),
            cmap="viridis",
        )
        ax[3].set_title("Difference")
        ax[3].axis("off")
        plt.savefig(path)
        plt.close()

    def evaluation_groups(self):
        return ["age", "sex"]