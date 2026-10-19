import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, normalization_shape, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=eps)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.tanh(x)
        return x


def get_inputs():
    # Assuming the input is a 2D tensor [batch_size, input_size]
    input_size = 128
    batch_size = 64
    x = torch.randn(batch_size, input_size)
    return [x]


def get_init_inputs():
    # Assuming the input is a 2D tensor [batch_size, input_size]
    input_size = 128
    eps = 1e-05
    return [input_size, input_size, eps]