import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x):
        x = self.groupnorm(x)
        return x - 0.5


batch_size = 16
num_channels = 3
num_groups = 3
height, width = 224, 224
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, num_groups, eps]