import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        x = torch.min(x)
        return x


def get_inputs():
    return [torch.randn(10, 5)]


def get_init_inputs():
    return []