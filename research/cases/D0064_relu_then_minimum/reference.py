import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.relu(x)
        x = torch.min(x, dim=1).values
        return x


in_features = 100
out_features = 20


def get_inputs():
    return [torch.randn(64, in_features)]


def get_init_inputs():
    return [in_features, out_features]