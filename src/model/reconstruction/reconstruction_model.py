"""
Classifier wrappers for both training and evaluation.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from ..model_wrapper import ModelWrapper
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.utils.image_metrics import calculate_data_range


class ReconstructionModel(ModelWrapper):
    """
    Classifier base class.
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def criterion(self, x, y):
        return self.loss(x, y)

    def performance_metric(self, x, y):
        return torch.tensor(0.0)

    @property
    def performance_metric_name(self):
        return "n/a"
    
    @property
    def name(self):
        return "ReconstructionModel"