"""
Classifier wrappers for both training and evaluation.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Classifier(ABC, nn.Module):
    """
    Classifier base class.
    """

    def __init__(self, network):
        super().__init__()
        self.network = network

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)

    def forward(self, x):
        return self.network(x)

    @property
    @abstractmethod
    def target_size(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def key(self):
        pass

    @abstractmethod
    def criterion(self, x, y):
        pass

    @abstractmethod
    def target_transformation(self, labels):
        pass

    @abstractmethod
    def classification_criteria(self, logits):
        pass


class TTypeBCEClassifier(Classifier):
    def __init__(self, network):
        super().__init__(network)
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


class TGradeBCEClassifier(Classifier):
    """
    Calculates the binary cross entropy loss for the tumor types

    """

    def __init__(self, network):
        super().__init__(network)
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


class NLLSurvClassifier(Classifier):

    def __init__(self, network, bin_size=1000, eps=1e-8):
        super().__init__(network)
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
