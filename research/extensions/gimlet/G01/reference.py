"""Runnable reconstruction of Gimlet's published clamped-attention semantics."""
import math

import torch
from torch import nn


class Model(nn.Module):
    def forward(self, q, k, v):
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        probabilities = logits.clamp(min=-5.0, max=5.0).softmax(dim=-1)
        return (probabilities @ v).clamp(min=-5.0, max=5.0)
