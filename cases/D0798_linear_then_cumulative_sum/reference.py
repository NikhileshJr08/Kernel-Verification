import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        x = torch.cumsum(x, dim=1)
        return x


input_size = 128
output_size = 64


def get_inputs():
    return [torch.randn(16, input_size)]


def get_init_inputs():
    return [input_size, output_size]