import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, div_factor):
        super().__init__()
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.div_factor = div_factor

    def forward(self, x):
        x = self.adaptive_avg_pool_2d(x)
        x = self.max(x)
        x = x / self.div_factor
        return x


batch_size = 16
in_channels = 3
height, width = 64, 64
kernel_size = 3
stride = 2
padding = 1
div_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, div_factor]