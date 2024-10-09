import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ReconstructionNetwork(nn.Module): 

    @property
    @abstractmethod
    def name(self):
        pass