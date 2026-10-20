import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import torch._dynamo


torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 512)
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 512)

import kernel_mid_family as mid_family
import kernel_mid_plain_family as mid_plain_family
import kernel_rawptr_all_family as rawptr_all_family
import kernel_qdq12_family as qdq12_family
import kernel_qdq_family as qdq_small_family
import kernel_v20_family as v20_family

# Exact-shape hybrid with the qdq12 official-speed backbone and the lower-latency
# small-token carveouts:
# - 256: exact-QDQ small path
# - 640/1024/2560: rawptr hybrid mid family
#   - keep the route-workspace policy from the small-WS lane
#   - use the route-mask broadcast path from iteration_26
# - 1280: exact compiled-route v20 family
# - 768/1536: all-rawptr workspace-reuse family, with route workspaces only on 768
# - 2816 and above: qdq12 compiled-post family
_SMALL_QDQ_TOKENS = frozenset((256,))
_MID_RAWPTR_TOKENS = frozenset((640, 1024, 2560))
_RAWPTR_ALL_TOKENS = frozenset((768, 1536))
_V20_TOKENS = frozenset((1280,))


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
    num_tokens = int(hidden_states.shape[0])
    if hidden_states.device.type != "cuda" and num_tokens in _V20_TOKENS:
        mid_plain_family.run(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
            topk_idx,
            topk_weight,
        )
        return

    if num_tokens in _SMALL_QDQ_TOKENS:
        qdq_small_family.run(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
            topk_idx,
            topk_weight,
        )
    elif num_tokens in _RAWPTR_ALL_TOKENS:
        rawptr_all_family.run(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
            topk_idx,
            topk_weight,
        )
    elif num_tokens in _MID_RAWPTR_TOKENS:
        mid_family.run(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
            topk_idx,
            topk_weight,
        )
    elif num_tokens in _V20_TOKENS:
        v20_family.run(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
            topk_idx,
            topk_weight,
        )
    else:
        qdq12_family.run(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
            topk_idx,
            topk_weight,
        )
