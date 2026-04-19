"""SPGM manager shell.

B1 split SPGM into update-policy vs state-management layers.
B2 made manager diagnostics scene-aware.
B3 now supports a first real population-participation controller: besides the
legacy xyz-only grad scale probe, it can prepare a deterministic pseudo-render
participation mask for the next iteration.
"""

from __future__ import annotations

import math

import torch

from .policy import build_spgm_grad_weights

_CLUSTER_NAMES = ('near', 'mid', 'far')


def _cluster_counts(cluster_id: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    counts = {name: 0 for name in _CLUSTER_NAMES}
    if cluster_id is None or mask is None or not bool(mask.any()):
        return counts
    for idx, name in enumerate(_CLUSTER_NAMES):
        counts[name] = int(((cluster_id == idx) & mask).sum().item())
    return counts


def _cluster_mask(cluster_id: torch.Tensor | None, active_mask: torch.Tensor | None, cluster_index: int) -> torch.Tensor | None:
    if cluster_id is None or active_mask is None:
        return None
    return active_mask & (cluster_id == int(cluster_index))


def _masked_mean(values: torch.Tensor | None, mask: torch.Tensor | None) -> float:
    if values is None or mask is None or not bool(mask.any()):
        return 0.0
    return float(values[mask].mean().item())


def _masked_p50(values: torch.Tensor | None, mask: torch.Tensor | None) -> float:
    if values is None or mask is None or not bool(mask.any()):
        return 0.0
    return float(values[mask].median().item())


def _masked_ratio(mask: torch.Tensor | None, ref_mask: torch.Tensor | None) -> float:
    if mask is None or ref_mask is None or not bool(ref_mask.any()):
        return 0.0
    return float(mask[ref_mask].float().mean().item())


def _apply_scale_(param, scale: torch.Tensor) -> tuple[float, float]:
    if param is None or getattr(param, 'grad', None) is None:
        return 0.0, 0.0
    grad = param.grad
    pre = float(torch.norm(grad).detach().cpu().item())
    mask = scale.to(device=grad.device, dtype=grad.dtype)
    while mask.ndim < grad.ndim:
        mask = mask.unsqueeze(-1)
    grad.mul_(mask)
    post = float(torch.norm(grad).detach().cpu().item())
    return pre, post


def build_spgm_update_policy(*args, **kwargs) -> dict:
    """Thin wrapper around gradient-weight policy."""
    policy = build_spgm_grad_weights(*args, **kwargs)
    wrapped = dict(policy)
    wrapped['update_policy_mode_effective'] = policy.get('policy_mode_effective')
    return wrapped


def _build_xyz_state_scale(
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    state_score: torch.Tensor | None,
    base_scales: dict[str, float],
) -> torch.Tensor:
    device = active_mask.device
    dtype = state_score.dtype if state_score is not None else torch.float32
    xyz_state_scale = torch.ones(active_mask.shape[0], dtype=dtype, device=device)
    if state_score is None or not bool(active_mask.any()):
        return xyz_state_scale

    for idx, name in enumerate(_CLUSTER_NAMES):
        mask = _cluster_mask(cluster_id, active_mask, idx)
        if mask is None or not bool(mask.any()):
            continue
        base = float(base_scales[name])
        mild_state_factor = 0.85 + 0.15 * state_score[mask]
        xyz_state_scale[mask] = torch.clamp(base * mild_state_factor, min=0.75, max=1.0)
    return xyz_state_scale


def _build_state_candidate_mask(
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    state_score: torch.Tensor | None,
    state_candidate_quantile: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    candidate_mask = torch.zeros_like(active_mask, dtype=torch.bool)
    candidate_counts = {name: 0 for name in _CLUSTER_NAMES}
    if state_score is None or not bool(active_mask.any()):
        return candidate_mask, candidate_counts

    q = float(min(max(state_candidate_quantile, 0.0), 1.0))
    for idx, name in enumerate(_CLUSTER_NAMES):
        mask = _cluster_mask(cluster_id, active_mask, idx)
        if mask is None or not bool(mask.any()):
            continue
        cluster_values = state_score[mask]
        threshold = float(torch.quantile(cluster_values, q=q).item())
        cur_mask = mask.clone()
        cur_mask[mask] = cluster_values <= threshold
        candidate_mask |= cur_mask
        candidate_counts[name] = int(cur_mask.sum().item())
    return candidate_mask, candidate_counts


def _build_participation_render_mask(
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    state_score: torch.Tensor | None,
    state_candidate_quantile: float,
    keep_ratios: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, int], dict[str, float]]:
    render_mask = torch.ones_like(active_mask, dtype=torch.bool)
    candidate_mask, candidate_counts = _build_state_candidate_mask(
        cluster_id=cluster_id,
        active_mask=active_mask,
        state_score=state_score,
        state_candidate_quantile=state_candidate_quantile,
    )
    drop_counts = {name: 0 for name in _CLUSTER_NAMES}
    part_ratios = {name: 1.0 for name in _CLUSTER_NAMES}
    if state_score is None or not bool(candidate_mask.any()):
        return render_mask, candidate_mask, candidate_counts, drop_counts, part_ratios

    for idx, name in enumerate(_CLUSTER_NAMES):
        cluster_candidate = candidate_mask & (cluster_id == idx)
        if not bool(cluster_candidate.any()):
            mask = _cluster_mask(cluster_id, active_mask, idx)
            part_ratios[name] = _masked_ratio(render_mask, mask)
            continue

        keep_ratio = min(max(float(keep_ratios[name]), 0.0), 1.0)
        cluster_indices = torch.where(cluster_candidate)[0]
        count = int(cluster_indices.numel())
        if keep_ratio <= 0.0:
            keep_indices = cluster_indices[:0]
        elif keep_ratio >= 1.0:
            keep_indices = cluster_indices
        else:
            keep_count = int(math.ceil(count * keep_ratio))
            keep_count = min(max(keep_count, 0), count)
            if keep_count <= 0:
                keep_indices = cluster_indices[:0]
            elif keep_count >= count:
                keep_indices = cluster_indices
            else:
                cluster_scores = state_score[cluster_indices]
                topk = torch.topk(cluster_scores, k=keep_count, largest=True, sorted=False)
                keep_indices = cluster_indices[topk.indices]
        dropped = torch.ones(count, dtype=torch.bool, device=cluster_indices.device)
        if keep_indices.numel() > 0:
            keep_set = set(keep_indices.tolist())
            dropped = torch.tensor([idx_val.item() not in keep_set for idx_val in cluster_indices], dtype=torch.bool, device=cluster_indices.device)
        drop_indices = cluster_indices[dropped]
        if drop_indices.numel() > 0:
            render_mask[drop_indices] = False
        drop_counts[name] = int(drop_indices.numel())
        mask = _cluster_mask(cluster_id, active_mask, idx)
        part_ratios[name] = _masked_ratio(render_mask, mask)

    return render_mask, candidate_mask, candidate_counts, drop_counts, part_ratios


def apply_spgm_state_management(
    gaussians,
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    update_policy: dict,
    manager_mode: str = 'summary_only',
    state_score: torch.Tensor | None = None,
    population_support_count: torch.Tensor | None = None,
    state_candidate_quantile: float = 0.5,
    state_base_scale_near: float = 1.0,
    state_base_scale_mid: float = 0.95,
    state_base_scale_far: float = 0.90,
    state_participation_keep_near: float = 1.0,
    state_participation_keep_mid: float = 0.9,
    state_participation_keep_far: float = 0.75,
) -> dict:
    """Return/apply deterministic state-management diagnostics.

    Modes:
    - summary_only: diagnostics only, no state action
    - xyz_lr_scale: apply a mild per-Gaussian xyz grad scale before optimizer.step
    - deterministic_participation: prepare a deterministic pseudo-render participation mask
      for the next iteration (pre-render / in-render control instead of post-backward scaling)
    """
    manager_mode_effective = str(manager_mode or 'summary_only').strip().lower()
    if manager_mode_effective not in {'summary_only', 'xyz_lr_scale', 'deterministic_participation', 'off', 'disabled'}:
        raise ValueError(
            f"Unsupported SPGM manager_mode={manager_mode_effective!r}; supported values are 'summary_only', 'xyz_lr_scale', 'deterministic_participation', and 'off'"
        )
    if manager_mode_effective in {'off', 'disabled'}:
        manager_mode_effective = 'disabled'

    state_candidate_quantile = float(min(max(state_candidate_quantile, 0.0), 1.0))
    base_scales = {
        'near': float(state_base_scale_near),
        'mid': float(state_base_scale_mid),
        'far': float(state_base_scale_far),
    }
    keep_ratios = {
        'near': float(state_participation_keep_near),
        'mid': float(state_participation_keep_mid),
        'far': float(state_participation_keep_far),
    }

    active_counts = _cluster_counts(cluster_id, active_mask)
    cluster_state_mean = {}
    cluster_state_p50 = {}

    candidate_mask, candidate_counts = _build_state_candidate_mask(
        cluster_id=cluster_id,
        active_mask=active_mask,
        state_score=state_score,
        state_candidate_quantile=state_candidate_quantile,
    )

    for idx, name in enumerate(_CLUSTER_NAMES):
        mask = _cluster_mask(cluster_id, active_mask, idx)
        cluster_state_mean[name] = _masked_mean(state_score, mask)
        cluster_state_p50[name] = _masked_p50(state_score, mask)
        if candidate_counts[name] <= 0:
            candidate_counts[name] = active_counts[name] if state_score is None else 0

    xyz_state_scale = _build_xyz_state_scale(
        cluster_id=cluster_id,
        active_mask=active_mask,
        state_score=state_score,
        base_scales=base_scales,
    )

    xyz_scale_mean = _masked_mean(xyz_state_scale, active_mask)
    xyz_scale_mean_near = _masked_mean(xyz_state_scale, _cluster_mask(cluster_id, active_mask, 0))
    xyz_scale_mean_mid = _masked_mean(xyz_state_scale, _cluster_mask(cluster_id, active_mask, 1))
    xyz_scale_mean_far = _masked_mean(xyz_state_scale, _cluster_mask(cluster_id, active_mask, 2))

    grad_pre_state_xyz = 0.0
    grad_post_state_xyz = 0.0
    participation_render_mask = None
    participation_drop_counts = {name: 0 for name in _CLUSTER_NAMES}
    participation_ratios = {name: 1.0 for name in _CLUSTER_NAMES}
    state_action_applied = False

    if manager_mode_effective == 'xyz_lr_scale':
        grad_pre_state_xyz, grad_post_state_xyz = _apply_scale_(getattr(gaussians, '_xyz', None), xyz_state_scale)
        state_action_applied = grad_pre_state_xyz > 0.0
    elif manager_mode_effective == 'deterministic_participation':
        participation_render_mask, candidate_mask, candidate_counts, participation_drop_counts, participation_ratios = _build_participation_render_mask(
            cluster_id=cluster_id,
            active_mask=active_mask,
            state_score=state_score,
            state_candidate_quantile=state_candidate_quantile,
            keep_ratios=keep_ratios,
        )
        state_action_applied = any(v > 0 for v in participation_drop_counts.values())

    state_participation_ratio = _masked_ratio(participation_render_mask, active_mask) if participation_render_mask is not None else 1.0

    return {
        'manager_mode_effective': manager_mode_effective,
        'state_action_applied': state_action_applied,
        'state_lr_scale_near': xyz_scale_mean_near,
        'state_lr_scale_mid': xyz_scale_mean_mid,
        'state_lr_scale_far': xyz_scale_mean_far,
        'state_opacity_decay_near': 0.0,
        'state_opacity_decay_mid': 0.0,
        'state_opacity_decay_far': 0.0,
        'state_candidate_count_near': candidate_counts['near'],
        'state_candidate_count_mid': candidate_counts['mid'],
        'state_candidate_count_far': candidate_counts['far'],
        'state_score_mean': _masked_mean(state_score, active_mask),
        'state_score_p50': _masked_p50(state_score, active_mask),
        'population_support_mean': _masked_mean(population_support_count, active_mask),
        'state_score_mean_near': cluster_state_mean['near'],
        'state_score_mean_mid': cluster_state_mean['mid'],
        'state_score_mean_far': cluster_state_mean['far'],
        'state_score_p50_near': cluster_state_p50['near'],
        'state_score_p50_mid': cluster_state_p50['mid'],
        'state_score_p50_far': cluster_state_p50['far'],
        'state_xyz_scale_mean': xyz_scale_mean,
        'state_xyz_scale_mean_near': xyz_scale_mean_near,
        'state_xyz_scale_mean_mid': xyz_scale_mean_mid,
        'state_xyz_scale_mean_far': xyz_scale_mean_far,
        'state_grad_norm_xyz_pre_state': grad_pre_state_xyz,
        'state_grad_norm_xyz_post_state': grad_post_state_xyz,
        'state_participation_ratio': state_participation_ratio,
        'state_participation_ratio_near': participation_ratios['near'],
        'state_participation_ratio_mid': participation_ratios['mid'],
        'state_participation_ratio_far': participation_ratios['far'],
        'state_participation_drop_count_near': participation_drop_counts['near'],
        'state_participation_drop_count_mid': participation_drop_counts['mid'],
        'state_participation_drop_count_far': participation_drop_counts['far'],
        'participation_render_mask': participation_render_mask,
    }
