"""Reconstruction of the blog's intentionally incorrect SDPA replacement.

The intermediate score clamp is missing. This is not recovered original code.
"""
import math

import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def forward(self, q, k, v):
        scale = 1.0 / math.sqrt(q.size(-1))
        result = F.scaled_dot_product_attention(
            q, k, v, scale=scale, dropout_p=0.0, is_causal=False
        )
        return result.clamp(min=-5.0, max=5.0)
