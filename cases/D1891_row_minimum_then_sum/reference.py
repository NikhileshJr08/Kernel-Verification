import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x):
        x = torch.min(x, dim=1)[0]
        x = torch.sum(x, dim=0)
        return x


def get_inputs():
    return [torch.randn(16, 32)]


def get_init_inputs():
    return [32, 16]