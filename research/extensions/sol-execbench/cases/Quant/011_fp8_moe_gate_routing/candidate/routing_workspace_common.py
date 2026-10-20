import torch


_NUM_EXPERTS = 256
_TOPK_EXPERTS = 8
_NUM_GROUPS = 8
_TOPK_GROUPS = 4
_EXPERTS_PER_GROUP = _NUM_EXPERTS // _NUM_GROUPS
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
            "bias_f32": torch.empty((_NUM_EXPERTS,), dtype=torch.float32, device=device),
            "scores": torch.empty((max_tokens, _NUM_EXPERTS), dtype=torch.float32, device=device),
            "tmp_scores": torch.empty((max_tokens, _NUM_EXPERTS), dtype=torch.float32, device=device),
            "group_top2_vals": torch.empty((max_tokens, _NUM_GROUPS, 2), dtype=torch.float32, device=device),
            "group_top2_idx": torch.empty((max_tokens, _NUM_GROUPS, 2), dtype=torch.int64, device=device),
            "group_scores": torch.empty((max_tokens, _NUM_GROUPS), dtype=torch.float32, device=device),
            "group_topk_vals": torch.empty((max_tokens, _TOPK_GROUPS), dtype=torch.float32, device=device),
            "group_idx": torch.empty((max_tokens, _TOPK_GROUPS), dtype=torch.int64, device=device),
            "group_mask": torch.empty((max_tokens, _NUM_GROUPS), dtype=torch.bool, device=device),
            "selected_vals": torch.empty((max_tokens, _TOPK_EXPERTS), dtype=torch.float32, device=device),
            "denom": torch.empty((max_tokens, 1), dtype=torch.float32, device=device),
        }
        _DEVICE_WORKSPACES[key] = ws
    elif ws["max_tokens"] < num_tokens:
        new_max_tokens = max(num_tokens, ws["max_tokens"] * 2)
        ws["max_tokens"] = new_max_tokens
        ws["scores"] = torch.empty((new_max_tokens, _NUM_EXPERTS), dtype=torch.float32, device=device)
        ws["tmp_scores"] = torch.empty((new_max_tokens, _NUM_EXPERTS), dtype=torch.float32, device=device)
        ws["group_top2_vals"] = torch.empty((new_max_tokens, _NUM_GROUPS, 2), dtype=torch.float32, device=device)
        ws["group_top2_idx"] = torch.empty((new_max_tokens, _NUM_GROUPS, 2), dtype=torch.int64, device=device)
        ws["group_scores"] = torch.empty((new_max_tokens, _NUM_GROUPS), dtype=torch.float32, device=device)
        ws["group_topk_vals"] = torch.empty((new_max_tokens, _TOPK_GROUPS), dtype=torch.float32, device=device)
        ws["group_idx"] = torch.empty((new_max_tokens, _TOPK_GROUPS), dtype=torch.int64, device=device)
        ws["group_mask"] = torch.empty((new_max_tokens, _NUM_GROUPS), dtype=torch.bool, device=device)
        ws["selected_vals"] = torch.empty((new_max_tokens, _TOPK_EXPERTS), dtype=torch.float32, device=device)
        ws["denom"] = torch.empty((new_max_tokens, 1), dtype=torch.float32, device=device)
    return ws


@torch.no_grad()
def route_postprocess_inplace(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    *,
    use_workspace: bool,
):
    num_tokens = logits.shape[0]
    if use_workspace and logits.device.type == "cuda":
        ws = _get_workspace(logits.device, num_tokens)
        scores = ws["scores"][:num_tokens]
        tmp_scores = ws["tmp_scores"][:num_tokens]
        group_top2_vals = ws["group_top2_vals"][:num_tokens]
        group_top2_idx = ws["group_top2_idx"][:num_tokens]
        group_scores = ws["group_scores"][:num_tokens]
        group_topk_vals = ws["group_topk_vals"][:num_tokens]
        group_idx = ws["group_idx"][:num_tokens]
        group_mask = ws["group_mask"][:num_tokens]
        selected_vals = ws["selected_vals"][:num_tokens]
        denom = ws["denom"][:num_tokens]

        scores.copy_(logits.to(torch.float32))
        scores.sigmoid_()
        ws["bias_f32"].copy_(e_score_correction_bias)
        tmp_scores.copy_(scores)
        tmp_scores.add_(ws["bias_f32"].unsqueeze(0))

        group_scores_view = tmp_scores.view(num_tokens, _NUM_GROUPS, _EXPERTS_PER_GROUP)
        torch.topk(group_scores_view, k=2, dim=-1, out=(group_top2_vals, group_top2_idx))
        torch.sum(group_top2_vals, dim=-1, out=group_scores)
        torch.topk(group_scores, k=_TOPK_GROUPS, dim=-1, sorted=False, out=(group_topk_vals, group_idx))

        group_mask.zero_()
        group_mask.scatter_(1, group_idx, True)
        tmp_scores.view(num_tokens, _NUM_GROUPS, _EXPERTS_PER_GROUP).masked_fill_(
            ~group_mask.unsqueeze(-1),
            float("-inf"),
        )

        torch.topk(tmp_scores, k=_TOPK_EXPERTS, dim=-1, sorted=False, out=(selected_vals, topk_idx))
        torch.gather(scores, 1, topk_idx, out=topk_weight)
        torch.sum(topk_weight, dim=-1, keepdim=True, out=denom)
        denom.add_(1e-20)
        topk_weight.div_(denom)
        topk_weight.mul_(routed_scaling_factor)
        return

    scores = torch.sigmoid(logits.to(torch.float32))
    scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)
    group_scores = scores_for_choice.view(num_tokens, _NUM_GROUPS, _EXPERTS_PER_GROUP).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=_TOPK_GROUPS, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    masked_scores = scores_for_choice.view(num_tokens, _NUM_GROUPS, _EXPERTS_PER_GROUP).masked_fill(
        ~group_mask.unsqueeze(-1).bool(),
        float("-inf"),
    ).reshape(num_tokens, _NUM_EXPERTS)
    selected_idx = torch.topk(masked_scores, k=_TOPK_EXPERTS, dim=-1, sorted=False)[1]
    selected_weight = scores.gather(1, selected_idx)
    selected_weight = selected_weight / (selected_weight.sum(dim=-1, keepdim=True) + 1e-20)
    selected_weight = selected_weight * routed_scaling_factor
    topk_idx.copy_(selected_idx)
    topk_weight.copy_(selected_weight)


def _warmup_route_workspace():
    try:
        if not torch.cuda.is_available():
            return
        _get_workspace(torch.device("cuda"), 2560)
        torch.cuda.synchronize()
    except Exception:
        pass


_warmup_route_workspace()
