import os
import pathlib
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from nvfp4_gemm import Sm100BlockScaledDenseGemmKernel
from nvfp4_common import run_blockscaled_gemm as run_persistent_blockscaled_gemm

_SF_VEC_SIZE = 16
_E2M1FN_MAX = 6.0
_FLOAT8_E4M3_MAX = 448.0
_F32_MAX = torch.finfo(torch.float32).max
_N_DIM = 8192
_K_DIM = 3584
_SCALE_BLOCK_COLS = _K_DIM // _SF_VEC_SIZE
_PERSISTENT_MIN_M = 2048

_SMALL_GEMM_CACHE = {}
_WORKSPACE_CACHE = {}
_WEIGHT_SCALE_BLOCKED_CACHE = {}
_WEIGHT_SCALE_PERM_CACHE = {}
_VISION_POS_CACHE = {}

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")


def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    row_blocks = rows // 128
    col_blocks = cols // 4
    blocks = input_matrix.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().contiguous()


@torch.compile
def _pack_fp4x2_compiled(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    mag = (
        (abs_x > 0.25).to(torch.uint8)
        + (abs_x >= 0.75).to(torch.uint8)
        + (abs_x > 1.25).to(torch.uint8)
        + (abs_x >= 1.75).to(torch.uint8)
        + (abs_x > 2.5).to(torch.uint8)
        + (abs_x >= 3.5).to(torch.uint8)
        + (abs_x > 5.0).to(torch.uint8)
    )
    code = mag + ((x < 0.0).to(torch.uint8) << 3)
    return code[:, ::2] + code[:, 1::2] * 16


@torch.compile
def _quantize_nvfp4_2d_compiled(x: torch.Tensor):
    m, k = x.shape
    sf_k = k // _SF_VEC_SIZE
    blocked = x.view(m, sf_k, _SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1)
    global_amax = block_max.amax().to(torch.float32).clamp(min=1e-12)

    global_encode_scale = torch.clamp((_FLOAT8_E4M3_MAX * _E2M1FN_MAX) / global_amax, max=_F32_MAX)
    global_decode_scale = torch.reciprocal(global_encode_scale)

    decode_scale = torch.clamp(
        (block_max.to(torch.float32) / _E2M1FN_MAX) * global_encode_scale,
        min=-_FLOAT8_E4M3_MAX,
        max=_FLOAT8_E4M3_MAX,
    )
    decode_scale_fp8 = decode_scale.to(torch.float8_e4m3fn)

    encode_scale = torch.clamp(
        torch.reciprocal(decode_scale_fp8.to(torch.float32) * global_decode_scale),
        max=_F32_MAX,
    )
    clipped = torch.clamp(
        blocked.to(torch.float32) * encode_scale.unsqueeze(-1),
        min=-_E2M1FN_MAX,
        max=_E2M1FN_MAX,
    ).reshape(m, k)
    packed_u8 = _pack_fp4x2_compiled(clipped)
    return packed_u8, decode_scale_fp8, global_decode_scale


def _quantize_nvfp4_2d(x: torch.Tensor):
    packed_u8, decode_scale_fp8, global_decode_scale = _quantize_nvfp4_2d_compiled(x)
    return packed_u8.view(torch.float4_e2m1fn_x2), decode_scale_fp8, global_decode_scale


@torch.compile
def _apply_scale_bias_inplace(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    x.mul_(scale)
    x.add_(bias)


def _make_small_gemm_args(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out):
    return (
        make_ptr(cutlass.Float4E2M1FN, act_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, weight_fp4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, act_scale_blocked.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, weight_scale_blocked.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
    )


def _get_small_gemm(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out):
    key = (str(out.device), act_fp4.shape[0])
    compiled = _SMALL_GEMM_CACHE.get(key)
    if compiled is None:
        gemm = Sm100BlockScaledDenseGemmKernel()
        compiled = cute.compile(
            gemm,
            *_make_small_gemm_args(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out),
            (act_fp4.shape[0], _N_DIM, _K_DIM, 1),
            options="--gpu-arch sm_100a --opt-level 2",
        )
        _SMALL_GEMM_CACHE[key] = compiled
    return compiled


def _run_small_gemm(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out, m):
    compiled = _get_small_gemm(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out)
    compiled(
        *_make_small_gemm_args(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out),
        (m, _N_DIM, _K_DIM, 1),
    )


def _make_dummy_fp4(rows: int, packed_cols: int, device: torch.device):
    return torch.zeros((rows, packed_cols), dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)


def _make_blocked_perm(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    if rows % 128 != 0:
        raise ValueError(f"Blocked permutation requires rows multiple of 128, got {rows}")
    base = torch.arange(rows * cols, dtype=torch.int64, device=device).view(rows, cols)
    return (
        base.view(rows // 128, 128, cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


def _to_blocked_out(scale_2d: torch.Tensor, perm: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    torch.index_select(scale_2d.view(-1), 0, perm, out=out)
    return out


def _get_workspace(m: int, device: torch.device):
    device_index = device.index if device.index is not None else -1
    key = (device_index, m)
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = {
            "act_perm": _make_blocked_perm(m, _SCALE_BLOCK_COLS, device),
            "act_scale_blocked": torch.empty(m * _SCALE_BLOCK_COLS, dtype=torch.float8_e4m3fn, device=device),
            "vision_embeds_flat": torch.empty((m, _N_DIM), dtype=torch.bfloat16, device=device),
        }
        _WORKSPACE_CACHE[key] = workspace
    return workspace


def _get_weight_scale_blocked(weight_scale_input: torch.Tensor, weight_scale_fp8: torch.Tensor) -> torch.Tensor:
    device_key = str(weight_scale_fp8.device)
    cache_key = (device_key, id(weight_scale_input))
    version = getattr(weight_scale_input, "_version", 0)
    entry = _WEIGHT_SCALE_BLOCKED_CACHE.get(cache_key)
    if entry is None or entry[0] is not weight_scale_input or entry[1] != version:
        perm = _WEIGHT_SCALE_PERM_CACHE.get(device_key)
        if perm is None:
            perm = _make_blocked_perm(weight_scale_fp8.shape[0], weight_scale_fp8.shape[1], weight_scale_fp8.device)
            _WEIGHT_SCALE_PERM_CACHE[device_key] = perm
        blocked = torch.empty(weight_scale_fp8.numel(), dtype=weight_scale_fp8.dtype, device=weight_scale_fp8.device)
        _to_blocked_out(weight_scale_fp8, perm, blocked)
        _WEIGHT_SCALE_BLOCKED_CACHE[cache_key] = (weight_scale_input, version, blocked)
        return blocked
    return entry[2]


def _get_vision_positions(vision_token_mask: torch.Tensor) -> torch.Tensor:
    cache_key = id(vision_token_mask)
    version = getattr(vision_token_mask, "_version", 0)
    entry = _VISION_POS_CACHE.get(cache_key)
    if entry is None or entry[0] is not vision_token_mask or entry[1] != version:
        positions = vision_token_mask.reshape(-1).nonzero(as_tuple=False).flatten().contiguous()
        _VISION_POS_CACHE[cache_key] = (vision_token_mask, version, positions)
        return positions
    return entry[2]


def _warmup_small_gemm_compiles():
    try:
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda")
        packed_k = _K_DIM // 2
        sf_k = _SCALE_BLOCK_COLS
        weight_fp4 = _make_dummy_fp4(_N_DIM, packed_k, device)
        weight_scale = torch.zeros((_N_DIM, sf_k), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        weight_scale_blocked = _to_blocked(weight_scale)
        for m in (128, 256, 512, 768, 1024, 1536):
            act_fp4 = _make_dummy_fp4(m, packed_k, device)
            act_scale = torch.zeros((m, sf_k), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
            act_scale_blocked = _to_blocked(act_scale)
            out = torch.empty((m, _N_DIM), dtype=torch.bfloat16, device=device)
            _get_small_gemm(act_fp4, weight_fp4, act_scale_blocked, weight_scale_blocked, out)
        torch.cuda.synchronize()
    except Exception:
        pass


_warmup_small_gemm_compiles()


@torch.no_grad()
def run(
    text_input_ids,
    text_embedding_weight,
    vision_features,
    vision_proj_weight_fp4,
    vision_proj_weight_scale,
    vision_proj_weight_global_scale,
    vision_proj_bias,
    vision_token_mask,
    merged_embeddings,
):
    device = vision_features.device
    batch_size, _ = text_input_ids.shape
    num_vision_tokens = vision_features.shape[1]
    hidden_size = text_embedding_weight.shape[1]
    vision_hidden_size = vision_features.shape[2]
    m = batch_size * num_vision_tokens
    k = vision_hidden_size

    flat_vision = vision_features.reshape(m, k).contiguous()
    act_fp4, act_scale_fp8, act_global_scale = _quantize_nvfp4_2d(flat_vision)

    workspace = _get_workspace(m, device)
    weight_fp4 = vision_proj_weight_fp4.squeeze(0).contiguous()
    weight_scale_fp8 = vision_proj_weight_scale.squeeze(0).contiguous()
    vision_embeds_flat = workspace["vision_embeds_flat"]

    if m >= _PERSISTENT_MIN_M:
        run_persistent_blockscaled_gemm(
            act_fp4,
            act_scale_fp8,
            weight_fp4,
            weight_scale_fp8,
            vision_embeds_flat,
        )
    else:
        weight_scale_blocked = _get_weight_scale_blocked(vision_proj_weight_scale, weight_scale_fp8)
        act_scale_blocked = _to_blocked_out(
            act_scale_fp8,
            workspace["act_perm"],
            workspace["act_scale_blocked"],
        )
        _run_small_gemm(
            act_fp4,
            weight_fp4,
            act_scale_blocked,
            weight_scale_blocked,
            vision_embeds_flat,
            m,
        )

    global_scale = act_global_scale * vision_proj_weight_global_scale
    if m >= _PERSISTENT_MIN_M:
        _apply_scale_bias_inplace(vision_embeds_flat, global_scale, vision_proj_bias)
    else:
        vision_embeds_flat.mul_(global_scale)
        vision_embeds_flat.add_(vision_proj_bias)

    merged_flat = merged_embeddings.view(-1, hidden_size)
    torch.index_select(text_embedding_weight, 0, text_input_ids.reshape(-1), out=merged_flat)
    vision_positions = _get_vision_positions(vision_token_mask)
    merged_flat.index_copy_(0, vision_positions, vision_embeds_flat)
