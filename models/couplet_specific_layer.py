import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupletSpecificLayer(nn.Module):
    def __init__(self, d_model):
        super(CoupletSpecificLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return self.norm(out + residual)
