from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class ModelWrapper(ABC, nn.Module):
    """
    Classifier base class.
    """

    def __init__(self):
        super().__init__()
        self.network = None

    def set_network(self, network):
        self.network = network

    def forward(self, x):
        return self.network(x)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def criterion(self, x, y):
        pass

    @abstractmethod
    def performance_metric(self, x, y):
        pass

    @property
    @abstractmethod
    def performance_metric_name(self):
        pass