"""
Classifier wrappers for both training and evaluation.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from ..model_wrapper import ModelWrapper


class ClassifierModel(ModelWrapper):
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
    def target_size(self):
        pass

    @property
    @abstractmethod
    def key(self):
        pass

    @abstractmethod
    def target_transformation(self, labels):
        pass

    @abstractmethod
    def classification_criteria(self, logits):
        pass

    @abstractmethod 
    def accumulation_function(self, results):
        pass

    def save_snapshot(self, x, y, y_pred, path, device, epoch):
        y_transformed = self.target_transformation(y)
        y_transformed = y_transformed.squeeze()

        y_pred = self.classification_criteria(y_pred)
        y_pred = y_pred.squeeze()

        # save labels to text file 
        with open(path + ".txt", "w") as f:
            f.write("Ground truth: " + str(y_transformed.cpu().numpy()) + "\n")
            f.write("Predictions: " + str(y_pred.cpu().numpy()) + "\n")


class TTypeBCEClassifier(ClassifierModel):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def target_size(self):
        return 1

    @property
    def name(self):
        return "TType"

    @property
    def key(self):
        return "Final pathologic diagnosis (WHO 2021)"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, labels):
        target_labels = labels[:, 3].clone()
        target_labels[target_labels < 3] = 0
        target_labels[target_labels == 3] = 1
        return target_labels

    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5

    def accumulation_function(self, results):
        """
        Accumulates the percentage of class 1 (diagnosis = 3) in the results.
        
        Args:
            results (pd.Series): The series of predicted or ground truth values.
        
        Returns:
            float: The percentage of class 1 (diagnosis = 3).
        """
        return (results == 1).mean() * 100

    def performance_metric(self, x, y):
        preds = self.classification_criteria(x)
        transformed_labels = self.target_transformation(y)
        return torch.sum(preds == transformed_labels) 

    @property
    def performance_metric_name(self):
        return "Accuracy"


class TGradeBCEClassifier(ClassifierModel):
    """
    Calculates the binary cross entropy loss for the tumor types

    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def target_size(self):
        return 1

    @property
    def name(self):
        return "TGrade"

    @property
    def key(self):
        return "WHO CNS Grade"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, labels):
        target_labels = labels[:, 2].clone()
        target_labels[target_labels < 4] = 0
        target_labels[target_labels == 4] = 1
        return target_labels

    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5
    
    def accumulation_function(self, results):
        """
        Accumulates the percentage of class 1 (diagnosis = 3) in the results.
        
        Args:
            results (pd.Series): The series of predicted or ground truth values.
        
        Returns:
            float: The percentage of class 1 (diagnosis = 3).
        """
        return (results == 1).mean() * 100
    
    def performance_metric(self, x, y):
        preds = self.classification_criteria(x)
        transformed_labels = self.target_transformation(y)
        return torch.sum(preds == transformed_labels) 

    @property
    def performance_metric_name(self):
        return "Accuracy"



class NLLSurvClassifier(ClassifierModel):

    def __init__(self, bin_size=1000, eps=1e-8):
        super().__init__()
        self.bin_size = bin_size
        self.eps = eps

    @property
    def target_size(self):
        return self.bin_size

    @property
    def name(self):
        return "Survival"

    @property
    def key(self):
        return "OS"

    def criterion(self, logits, labels):
        # 1 alive, 0 dead
        censor = labels[:, 4]
        censor = censor.unsqueeze(1)
        #censor = censor * -1 + 1
        os = self.target_transformation(labels)
        os = os.unsqueeze(1)

        # Compute hazard, survival, and offset survival
        haz = logits.sigmoid() + self.eps  # prevent log(0) downstream
        sur = torch.cumprod(1 - haz, dim=1)
        sur_pad = torch.cat([torch.ones_like(censor), sur], dim=1)

        # Get values at ground truth bin
        sur_pre = sur_pad.gather(dim=1, index=os)
        sur_cur = sur_pad.gather(dim=1, index=os + 1)
        haz_cur = haz.gather(dim=1, index=os)

        # Compute NLL loss
        loss = (
            -(1 - censor) * sur_pre.log()
            - (1 - censor) * haz_cur.log()
            - censor * sur_cur.log()
        )

        return loss.mean()

    def target_transformation(self, labels):
        target_labels = (labels[:, 5].clone() // self.bin_size).long()
        return target_labels

    def classification_criteria(self, logits):
        _, preds = torch.max(logits, 1)
        return preds

    def accumulation_function(self, results):
        """
        Accumulates the percentage of class 1 (diagnosis = 3) in the results.
        
        Args:
            results (pd.Series): The series of predicted or ground truth values.
        
        Returns:
            float: The average survival bin index.
        """
        return results.mean()
