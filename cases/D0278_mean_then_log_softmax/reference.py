import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = torch.log_softmax(x, dim=0)
        return x


def get_inputs():
    return [torch.randn(3, 5, 10)]


def get_init_inputs():
    return []