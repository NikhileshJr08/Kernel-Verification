import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, num_groups):
        super().__init__()
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x):
        x = self.relu(x)
        x = self.group_norm(x)
        x = x / 2.0  # Dividing by 2.0
        return x


batch_size = 128
channels = 3
height, width = 32, 32
num_groups = 3


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [channels, num_groups]