import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = torch.logsumexp(x, dim=1)
        x = torch.tanh(x)
        return x


batch_size = 32
num_classes = 10
height = 32
width = 32


def get_inputs():
    x = torch.randn(batch_size, num_classes, height, width)
    return [x]


def get_init_inputs():
    return []