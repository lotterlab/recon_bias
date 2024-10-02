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
        """
        Number of classes for the classification.
        """
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
    """
    Classifier for tumor type prediction using binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def name(self):
        return "TType"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, y):
        target_labels = y[:, 3].clone()
        target_labels[target_labels < 3] = 0
        target_labels[target_labels == 3] = 1
        return target_labels

    def evaluation_performance_metric(self, x, y):
        # Check if both classes (0 and 1) are present
        x = x.detach().numpy()
        y = y.detach().numpy()
        if len(np.unique(y)) == 1:
            print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
            return 0  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)
    
    def epoch_performance_metric(self, x, y):
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(x, target_transform), 1

    @property
    def performance_metric_name(self):
        return "AUROC"
    
    @property
    def performance_metric_input_value(self):
        "score"
    
    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value

    def evaluation_groups(self):
        return ["age", "sex"]
    
    @property
    def num_classes(self):
        return 1
    
    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5
    
    def final_activation(self, logits): 
        logits = logits.squeeze()
        return torch.sigmoid(logits)


class TGradeBCEClassifier(ClassifierModel):
    """
    Classifier for tumor grade prediction using binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def name(self):
        return "TGrade"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, y):
        target_labels = y[:, 2].clone()
        target_labels[target_labels < 4] = 0
        target_labels[target_labels == 4] = 1
        return target_labels
    
    def evaluation_performance_metric(self, x, y):
        x = x.detach().numpy()
        y = y.detach().numpy()
        # Check if both classes (0 and 1) are present
        if len(np.unique(y)) == 1:
            print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
            return 0  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)
    
    def epoch_performance_metric(self, x, y):
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(x, target_transform), 1
    
    @property
    def performance_metric_name(self):
        return "AUROC"
    
    @property
    def performance_metric_input_value(self):
        "score"

    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value
    
    def evaluation_groups(self):
        return ["age", "sex"]
    
    @property
    def num_classes(self):
        return 1
    
    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5
    
    def final_activation(self, logits): 
        logits = logits.squeeze()
        return torch.sigmoid(logits)

class NLLSurvClassifier(ClassifierModel):
    """
    Classifier for survival prediction using negative log likelihood loss.
    """

    def __init__(self, bins, bin_size, eps=1e-8):
        super().__init__()
        self.bins = bins
        self.bin_size = bin_size
        self.eps = eps

    @property
    def name(self):
        return "Survival"

    def criterion(self, logits, labels):
        # 1 alive, 0 dead
        censor = labels[:, 4]
        censor = censor.unsqueeze(1)
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

    def target_transformation(self, y):
        # extend bins to tensor with same lenght as labels 
        bins = torch.full((y.shape[0],),self.bins - 1)
        target_labels = (torch.min(y[:, 5].clone() // self.bin_size, bins)).long()
        return target_labels
    
    def evaluation_performance_metric(self, x, y):
        if len(np.unique(y)) == 1:
            print("Warning: Only one class present in y_true. C-Index score is not defined.")
            return 0.5
        x = x.detach().numpy()
        y = y.detach().numpy()
        x = x.squeeze()
        y = y.squeeze() 
        c_index = concordance_index(y, x)
        return c_index
    
    def epoch_performance_metric(self, x, y):
        target_transform = self.target_transformation(y)
        _, preds = torch.max(x, 1)
        return self.evaluation_performance_metric(preds, target_transform), 1
    
    @property
    def performance_metric_name(self):
        return "C-Index"
    
    @property
    def performance_metric_input_value(self):
        "prediction"

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
        
        return p_value

    def evaluation_groups(self):
        return ["age", "sex"]
    
    @property
    def num_classes(self):
        return self.bins
    
    def classification_criteria(self, logits):
        _, preds = torch.max(logits, 1)
        return preds
    
    def final_activation(self, logits):
        logits = logits.squeeze()
        return torch.softmax(logits, dim=1)

# TODO: Implement AgeCEClassifier
class AgeCEClassifier(ClassifierModel):
    """
    Classifier for age prediction using cross entropy loss.
    """

    def __init__(self, age_bins):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.age_bins = age_bins
        self.age_labels = list(range(0, len(self.age_bins) - 1))

    @property
    def name(self):
        return "Age"
    
    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        transformed_labels = transformed_labels.long()
        logits = logits.squeeze(1)
        loss = self.ce_loss(logits, transformed_labels)
        return loss
    
    def target_transformation(self, y):
        target_labels = y[:, 6].clone()
        return target_labels
    
    def evaluation_performance_metric(self, x, y):
        # Calculate accuracy for batch 
        correct = (x == y).sum().item()
        return correct / len(y)
    
    def epoch_performance_metric(self, x, y):
        _, preds = torch.max(x, 1)
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(preds, target_transform), 1

    @property
    def performance_metric_name(self):
        return "Accuracy"
    
    @property
    def performance_metric_input_value(self):
        "prediction"
    
    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value

    def evaluation_groups(self):
        return ["sex"]
    
    @property
    def num_classes(self):
        return len(self.age_bins) - 1
    
    def classification_criteria(self, logits):
        _, preds = torch.max(logits, 1)
        return preds
    
    def final_activation(self, logits): 
        logits = logits.squeeze()
        return torch.softmax(logits, dim=1)
    

# TODO: Implement GenderBCEClassifier
