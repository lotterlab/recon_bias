"""
Classifier wrappers for both training and evaluation.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from ..model_wrapper import ModelWrapper
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from src.utils.delong import delong_roc_test
import numpy as np

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
    def num_classes(self):
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
        """
        Specifies how to convert logits to predictions.
        """
        pass

    @abstractmethod
    def final_activation(self, logits): 
        """
        Specifies the final activation function for the model.
        """
        pass

    # TODO could go
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

    @abstractmethod
    def significance(self, gt, pred, recon): 
        pass


class TTypeBCEClassifier(ClassifierModel):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def num_classes(self):
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
        # Check if both classes (0 and 1) are present
        if len(np.unique(y)) == 1:
            print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
            return None  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)

    @property
    def performance_metric_name(self):
        return "AUROC"

    def final_activation(self, logits): 
        logits = logits.squeeze()
        return torch.sigmoid(logits)
    
    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value
    
    @property
    def performance_metric_value(self):
        "score"


class TGradeBCEClassifier(ClassifierModel):
    """
    Calculates the binary cross entropy loss for the tumor types

    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def num_classes(self):
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
    
    @property
    def performance_metric_name(self):
        return "AUROC"

    def performance_metric(self, x, y):
        # Check if both classes (0 and 1) are present
        if len(np.unique(y)) == 1:
            print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
            return None  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)

    def final_activation(self, logits): 
        logits = logits.squeeze()
        return torch.sigmoid(logits)

    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value
    
    @property
    def performance_metric_value(self):
        "score"


class NLLSurvClassifier(ClassifierModel):

    def __init__(self, bins, bin_size, eps=1e-8):
        super().__init__()
        self.bins = bins
        self.bin_size = bin_size
        self.eps = eps

    @property
    def num_classes(self):
        return self.bins

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
        target_labels = (np.min(labels[:, 5].clone() // self.bin_size), self.bins).long()
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
    
    @property
    def performance_metric_name(self):
        return "C-Index"
    
    def performance_metric(self, x, y):
        x = x.detach().numpy()
        y = y.detach().numpy()
        x = x.squeeze()
        y = y.squeeze() 
        print(x.shape, y.shape)
        return concordance_index(y, x)

    def final_activation(self, logits):
        logits = logits.squeeze()
        return torch.softmax(logits, dim=1)

    def significance(self, gt, pred, recon):
        observed_diff = concordance_index(gt, pred) - concordance_index(gt, recon)
    
        n = len(gt)
        boot_diffs = []
        
        for _ in range(1000):
            # Create a bootstrap sample with replacement
            indices = np.random.choice(np.arange(n), size=n, replace=True)
            y_true_boot = np.array(gt)[indices]
            y_pred1_boot = np.array(pred)[indices]
            y_pred2_boot = np.array(recon)[indices]
            
            # Calculate C-index difference for the bootstrap sample
            boot_diff = concordance_index(y_true_boot, y_pred1_boot) - concordance_index(y_true_boot, y_pred2_boot)
            boot_diffs.append(boot_diff)
        
        boot_diffs = np.array(boot_diffs)
        
        # Calculate the p-value (two-sided test)
        p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))
        
        return observed_diff, p_value

    @property
    def performance_metric_value(self):
        "prediction"