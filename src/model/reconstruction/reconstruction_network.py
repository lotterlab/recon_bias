from abc import abstractmethod

import torch
import torch.nn as nn


class ReconstructionNetwork(nn.Module):

    @property
    @abstractmethod
    def name(self):
        pass
