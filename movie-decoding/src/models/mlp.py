import torch
import numpy as np
from torch import nn
from typing import Optional, Tuple, Union


class MLPClassification(nn.Module):
    def __init__(self, input_size=37, hidden_size=2048):
        super().__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
        inputs: Optional[torch.Tensor],
        labels=None,
    ):
        hidden = self.fc1(inputs)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        logits = self.sigmoid(output)
        return logits