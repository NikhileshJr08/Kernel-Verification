import torch

from gemm_impl import linear_bf16_input

_SCRATCH_KEY = None
_SCRATCH_CAPACITY = 0
_SCRATCH = None


def _get_scratch(device: torch.device, m: int, hidden_size: int):
    global _SCRATCH_KEY, _SCRATCH_CAPACITY, _SCRATCH
    key = (device.type, device.index, hidden_size)
    if _SCRATCH_KEY != key or _SCRATCH_CAPACITY < m:
        _SCRATCH_KEY = key
        _SCRATCH_CAPACITY = m
        _SCRATCH = {
            "proj_1": torch.empty((_SCRATCH_CAPACITY, hidden_size), device=device, dtype=torch.float32),
            "proj_2": torch.empty((_SCRATCH_CAPACITY, hidden_size), device=device, dtype=torch.float32),
        }
    return _SCRATCH


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
    output: torch.Tensor,
):
    first_stream = hidden_states[0]
    bsz, seq_len, hidden_size = first_stream.shape
    m = bsz * seq_len

    scratch = _get_scratch(hidden_states.device, m, hidden_size)
    proj_1 = scratch["proj_1"][:m]
    proj_2 = scratch["proj_2"][:m]

    linear_bf16_input(hidden_states[1].reshape(m, hidden_size), unembed_proj_1, proj_1)
    linear_bf16_input(hidden_states[2].reshape(m, hidden_size), unembed_proj_2, proj_2)

    proj_1 = proj_1.view(bsz, seq_len, hidden_size)
    proj_2 = proj_2.view(bsz, seq_len, hidden_size)

    target_magnitude = torch.linalg.vector_norm(
        first_stream,
        ord=2,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
    ).mul_(hidden_size ** -0.5)
    current_magnitude_1 = torch.sqrt(torch.clamp_min(torch.mean(proj_1.square(), dim=-1, keepdim=True), epsilon))
    current_magnitude_2 = torch.sqrt(torch.clamp_min(torch.mean(proj_2.square(), dim=-1, keepdim=True), epsilon))

    proj_1.mul_(target_magnitude / current_magnitude_1)
    proj_2.mul_(target_magnitude / current_magnitude_2)

    proj_1.add_(proj_2).add_(first_stream).mul_(1.0 / 3.0)
    output.copy_(proj_1.to(torch.bfloat16))
