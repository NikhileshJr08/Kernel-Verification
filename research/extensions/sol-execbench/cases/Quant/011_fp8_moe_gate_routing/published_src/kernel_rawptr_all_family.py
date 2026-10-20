import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch

from fp8_blockwise_common import fp8_blockwise_mm_explicit_rawptr
from routing_workspace_common import route_postprocess_inplace


_E4M3_MAX = 448.0
_BLOCK_K = 128
_HIDDEN_SIZE = 7168
_NUM_EXPERTS = 256
_HIDDEN_BLOCKS = _HIDDEN_SIZE // _BLOCK_K
_EXPERT_BLOCKS = _NUM_EXPERTS // _BLOCK_K
_NUM_GROUPS = 8
_EXPERTS_PER_GROUP = _NUM_EXPERTS // _NUM_GROUPS
_TOPK_GROUPS = 4
_TOPK_EXPERTS = 8
_ROUTE_WORKSPACE_TOKENS = frozenset((768,))
_DEVICE_WORKSPACES = {}


@torch.no_grad()
def _device_key(device: torch.device) -> int:
    return device.index if device.type == "cuda" and device.index is not None else -1


@torch.no_grad()
def _get_workspace(device: torch.device, num_tokens: int):
    key = _device_key(device)
    ws = _DEVICE_WORKSPACES.get(key)
    if ws is None:
        max_tokens = num_tokens
        ws = {
            "max_tokens": max_tokens,
            "hidden_f32": torch.empty((max_tokens, _HIDDEN_BLOCKS, _BLOCK_K), dtype=torch.float32, device=device),
            "qx": torch.empty((max_tokens, _HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=device),
            "logits": torch.empty((max_tokens, _NUM_EXPERTS), dtype=torch.bfloat16, device=device),
            "weight_t_f32": torch.empty((_HIDDEN_SIZE, _NUM_EXPERTS), dtype=torch.float32, device=device),
            "qw": torch.empty((_NUM_EXPERTS, _HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=device),
            "scale_w_t": torch.empty((_EXPERT_BLOCKS, _HIDDEN_BLOCKS), dtype=torch.float32, device=device),
        }
        _DEVICE_WORKSPACES[key] = ws
    elif ws["max_tokens"] < num_tokens:
        new_max_tokens = max(num_tokens, ws["max_tokens"] * 2)
        ws["max_tokens"] = new_max_tokens
        ws["hidden_f32"] = torch.empty((new_max_tokens, _HIDDEN_BLOCKS, _BLOCK_K), dtype=torch.float32, device=device)
        ws["qx"] = torch.empty((new_max_tokens, _HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=device)
        ws["logits"] = torch.empty((new_max_tokens, _NUM_EXPERTS), dtype=torch.bfloat16, device=device)
    return ws


@torch.no_grad()
def _quantize_activation_into(
    hidden_states: torch.Tensor,
    scale_x: torch.Tensor,
    hidden_f32: torch.Tensor,
    qx_out: torch.Tensor,
):
    num_tokens = hidden_states.shape[0]
    hidden_view = hidden_f32[:num_tokens]
    hidden_view.copy_(hidden_states.view(num_tokens, _HIDDEN_BLOCKS, _BLOCK_K))
    hidden_view.div_(scale_x.unsqueeze(-1))
    hidden_view.clamp_(min=-_E4M3_MAX, max=_E4M3_MAX)
    qx_out.copy_(hidden_view.view(num_tokens, _HIDDEN_SIZE))
    return qx_out


@torch.no_grad()
def _quantize_weight_into(
    weight: torch.Tensor,
    scale_w: torch.Tensor,
    weight_t_f32: torch.Tensor,
    qw_out: torch.Tensor,
    scale_w_t_out: torch.Tensor,
):
    weight_t_f32.copy_(weight.t())
    weight_blocks = weight_t_f32.view(_HIDDEN_BLOCKS, _BLOCK_K, _EXPERT_BLOCKS, _BLOCK_K)
    weight_blocks.div_(scale_w.view(_HIDDEN_BLOCKS, 1, _EXPERT_BLOCKS, 1))
    weight_blocks.clamp_(min=-_E4M3_MAX, max=_E4M3_MAX)
    qw_out.copy_(weight_t_f32.t())
    scale_w_t_out.copy_(scale_w.t())
    return qw_out, scale_w_t_out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    routed_scaling_factor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
):
    num_tokens = hidden_states.shape[0]
    scale_x_f32 = scale_x.to(torch.float32).contiguous()
    scale_w_f32 = scale_w.to(torch.float32).contiguous()

    ws = _get_workspace(hidden_states.device, num_tokens)
    qx = _quantize_activation_into(
        hidden_states,
        scale_x_f32,
        ws["hidden_f32"],
        ws["qx"][:num_tokens],
    )
    qw, scale_w_t = _quantize_weight_into(
        weight,
        scale_w_f32,
        ws["weight_t_f32"],
        ws["qw"],
        ws["scale_w_t"],
    )

    logits = fp8_blockwise_mm_explicit_rawptr(
        qx,
        scale_x_f32,
        qw,
        scale_w_t,
        out_dtype=torch.bfloat16,
        out=ws["logits"][:num_tokens],
    )
    route_postprocess_inplace(
        logits,
        e_score_correction_bias,
        routed_scaling_factor,
        topk_idx,
        topk_weight,
        use_workspace=num_tokens in _ROUTE_WORKSPACE_TOKENS,
    )
