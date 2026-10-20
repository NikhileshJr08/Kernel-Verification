import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@triton.jit
def triplet_margin_loss_serial_kernel(
    anchor_ptr, pos_ptr, neg_ptr, out_ptr,
    D, eps, margin,
    BLOCK_D: tl.constexpr,
    N_TILES: tl.constexpr,
    N_ROWS: tl.constexpr,
):
    # Single program handles all rows; loops are statically unrolled.
    # Accumulate squared distances in fp32 to improve numerical stability.
    for rb in tl.static_range(N_ROWS):
        s_pos = tl.zeros((), dtype=tl.float32)
        s_neg = tl.zeros((), dtype=tl.float32)

        row_start = rb * D

        # Tiled over feature dimension
        for t in tl.static_range(N_TILES):
            offs = t * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs < D
            idx = row_start + offs

            a = tl.load(anchor_ptr + idx, mask=mask, other=0.0)
            p = tl.load(pos_ptr    + idx, mask=mask, other=0.0)
            n = tl.load(neg_ptr    + idx, mask=mask, other=0.0)

            diff_pos = (a - p).to(tl.float32)
            diff_neg = (a - n).to(tl.float32)

            s_pos += tl.sum(diff_pos * diff_pos, axis=0)
            s_neg += tl.sum(diff_neg * diff_neg, axis=0)

        # Compute distances and hinge loss
        pos_dist = tl.sqrt(s_pos + eps)
        neg_dist = tl.sqrt(s_neg + eps)
        loss_val = tl.maximum(0.0, pos_dist - neg_dist + margin)

        # Store one loss value per row
        tl.store(out_ptr + rb, loss_val)


def _choose_launch_params(D, B):
    # Choose a small power-of-two tile close to D to reduce mask overhead.
    # Since we process all rows in a single program, keep num_warps/stages small to minimize overhead.
    if D <= 16:
        BLOCK_D = 16
    elif D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    else:
        BLOCK_D = 128
    N_TILES = (D + BLOCK_D - 1) // BLOCK_D
    N_ROWS = B  # compile-time specialization per batch
    num_warps = 1
    num_stages = 1
    return BLOCK_D, N_TILES, N_ROWS, num_warps, num_stages


def _triton_triplet_margin_loss(anchor, positive, negative, margin=1.0, eps=1e-6):
    """
    Triton forward-only computation of per-sample triplet margin loss (swap=False, p=2).
    Returns a tensor of shape [B] with per-row losses (dtype matches input).
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Triton kernel requires CUDA tensors."
    assert anchor.shape == positive.shape == negative.shape, "Inputs must have the same shape."
    assert anchor.ndim == 2, "Expected 2D inputs [B, D]."

    # Make inputs contiguous to simplify addressing and ensure coalesced loads
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    B, D = anchor.shape

    # Output buffer: dtype matches input
    out = torch.empty((B,), device=anchor.device, dtype=anchor.dtype)

    BLOCK_D, N_TILES, N_ROWS, num_warps, num_stages = _choose_launch_params(D, B)

    # Single-program launch; loops over rows and tiles are statically unrolled
    grid = (1,)
    triplet_margin_loss_serial_kernel[grid](
        anchor, positive, negative, out,
        D, float(eps), float(margin),
        BLOCK_D=BLOCK_D,
        N_TILES=N_TILES,
        N_ROWS=N_ROWS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, target_size, reduction="mean"):
        super().__init__()
        # Mirror parameters from the original TripletMarginLoss configuration
        self.margin = 1.0
        self.p = 2
        self.eps = 1e-6
        self.swap = False
        self.reduction = reduction
        self.mean = torch.mean
        self.input_size = input_size
        self.target_size = target_size

    def forward(self, anchor, positive, negative):
        # Always use Triton when available and on CUDA, except when autograd is required.
        needs_grad = anchor.requires_grad or positive.requires_grad or negative.requires_grad
        is_cuda = anchor.is_cuda and positive.is_cuda and negative.is_cuda
        use_triton = (
            _TRITON_AVAILABLE
            and is_cuda
            and (self.p == 2) and (not self.swap)
            and (not needs_grad)
        )

        if use_triton:
            loss_vec = _triton_triplet_margin_loss(anchor, positive, negative, margin=self.margin, eps=self.eps)
            # Apply reduction to mirror nn.TripletMarginLoss, then the extra mean as in the original code
            if self.reduction == "none":
                reduced = loss_vec
            elif self.reduction == "sum":
                reduced = loss_vec.sum()
            else:  # "mean" or default
                reduced = loss_vec.mean()
            return self.mean(reduced)
        else:
            # Fallback to PyTorch functional for correctness and autograd
            loss = F.triplet_margin_loss(
                anchor, positive, negative,
                margin=self.margin, p=self.p, eps=self.eps,
                swap=self.swap, reduction=self.reduction
            )
            return self.mean(loss)


def get_inputs():
    anchor = torch.randn(16, 5)
    positive = torch.randn(16, 5)
    negative = torch.randn(16, 5)
    return [anchor, positive, negative]


def get_init_inputs():
    return [5, 5, "mean"]
