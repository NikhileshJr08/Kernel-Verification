import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, target_size, reduction="mean"):
        super().__init__()
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-6, swap=False, reduction=reduction)
        self.mean = torch.mean
        self.input_size = input_size
        self.target_size = target_size

    def forward(self, anchor, positive, negative):
        loss = self.criterion(anchor, positive, negative)
        return self.mean(loss)


def get_inputs():
    anchor = torch.randn(16, 5)
    positive = torch.randn(16, 5)
    negative = torch.randn(16, 5)
    return [anchor, positive, negative]


def get_init_inputs():
    return [5, 5, "mean"]