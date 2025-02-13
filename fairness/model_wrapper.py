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
        """Name of the model, used to identify scores in the results."""
        pass

    @abstractmethod
    def target_transformation(self, y):
        """Transform the labels to fit the model output.

        Args:
            y: target
        Returns:
            transformed target
        """
        pass

    @abstractmethod
    def criterion(self, x, y):
        """
        Loss function.

        Args:
            x: input
            y: target
        Returns:
            loss
        """
        pass

    @abstractmethod
    def evaluation_performance_metric(self, x, y):
        """
        Method to evaluate the performance of the model during the evaluation phase.

        Args:
            x: input
            y: target
        Returns:
            performance metric value
        """
        pass

    @abstractmethod
    def epoch_performance_metric(self, x, y):
        """
        Method to evaluate the performance of the model during the training phase.

        Args:
            x: input
            y: target
        Returns:
            performance metric value
        """
        pass

    @property
    @abstractmethod
    def performance_metric_name(self):
        """Name of the performance metric."""
        pass

    @property
    @abstractmethod
    def performance_metric_input_value(self):
        """Decides weather the performance metric uses the prediction or the raw score."""
        pass

    @abstractmethod
    def save_snapshot(self, x, y, y_pred, path, device, epoch):
        """
        Save a snapshot of the model output.

        Args:
            x: input
            y: target
            y_pred: prediction
        """
        pass

    @property
    @abstractmethod
    def evaluation_groups(self):
        """Defines the subgroups for the evaluation phase."""
        pass
