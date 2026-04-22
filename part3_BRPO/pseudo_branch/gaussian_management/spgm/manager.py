"""SPGM manager shell.

B1 split SPGM into update-policy vs state-management layers.
B2 made manager diagnostics scene-aware.
B3 supports deterministic participation and opacity participation modes.

G-BRPO-0 adds stochastic_bernoulli_opacity mode for direct BRPO alignment:
- p_i^{drop} = r * w_cluster(i) * S_i
- m_i ~ Bernoulli(1 - p_i^{drop})
- alpha_i^{eff} = alpha_i * m_i
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


def _cluster_mask(cluster_id: torch.Tensor | None, control_mask: torch.Tensor | None, cluster_index: int) -> torch.Tensor | None:
    if cluster_id is None or control_mask is None:
        return None
    return control_mask & (cluster_id == int(cluster_index))


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
    control_mask: torch.Tensor,
    state_score: torch.Tensor | None,
    base_scales: dict[str, float],
) -> torch.Tensor:
    device = control_mask.device
    dtype = state_score.dtype if state_score is not None else torch.float32
    xyz_state_scale = torch.ones(control_mask.shape[0], dtype=dtype, device=device)
    if state_score is None or not bool(control_mask.any()):
        return xyz_state_scale

    for idx, name in enumerate(_CLUSTER_NAMES):
        mask = _cluster_mask(cluster_id, control_mask, idx)
        if mask is None or not bool(mask.any()):
            continue
        base = float(base_scales[name])
        mild_state_factor = 0.85 + 0.15 * state_score[mask]
        xyz_state_scale[mask] = torch.clamp(base * mild_state_factor, min=0.75, max=1.0)
    return xyz_state_scale


def _build_candidate_mask(
    cluster_id: torch.Tensor,
    control_mask: torch.Tensor,
    candidate_score: torch.Tensor | None,
    candidate_quantile: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    candidate_mask = torch.zeros_like(control_mask, dtype=torch.bool)
    candidate_counts = {name: 0 for name in _CLUSTER_NAMES}
    if candidate_score is None or not bool(control_mask.any()):
        return candidate_mask, candidate_counts

    q = float(min(max(candidate_quantile, 0.0), 1.0))
    for idx, name in enumerate(_CLUSTER_NAMES):
        mask = _cluster_mask(cluster_id, control_mask, idx)
        if mask is None or not bool(mask.any()):
            continue
        cluster_values = candidate_score[mask]
        threshold = float(torch.quantile(cluster_values, q=q).item())
        cur_mask = mask.clone()
        cur_mask[mask] = cluster_values <= threshold
        candidate_mask |= cur_mask
        candidate_counts[name] = int(cur_mask.sum().item())
    return candidate_mask, candidate_counts


def _build_state_candidate_mask(
    cluster_id: torch.Tensor,
    control_mask: torch.Tensor,
    state_score: torch.Tensor | None,
    state_candidate_quantile: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    return _build_candidate_mask(
        cluster_id=cluster_id,
        control_mask=control_mask,
        candidate_score=state_score,
        candidate_quantile=state_candidate_quantile,
    )


def _build_participation_render_mask(
    cluster_id: torch.Tensor,
    control_mask: torch.Tensor,
    state_score: torch.Tensor | None,
    state_candidate_quantile: float,
    keep_ratios: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, int], dict[str, float]]:
    render_mask = torch.ones_like(control_mask, dtype=torch.bool)
    candidate_mask, candidate_counts = _build_state_candidate_mask(
        cluster_id=cluster_id,
        control_mask=control_mask,
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
            mask = _cluster_mask(cluster_id, control_mask, idx)
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
        mask = _cluster_mask(cluster_id, control_mask, idx)
        part_ratios[name] = _masked_ratio(render_mask, mask)

    return render_mask, candidate_mask, candidate_counts, drop_counts, part_ratios


def _build_opacity_participation_scale(
    cluster_id: torch.Tensor,
    control_mask: torch.Tensor,
    participation_score: torch.Tensor | None,
    state_candidate_quantile: float,
    opacity_floors: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, float]]:
    device = control_mask.device
    dtype = participation_score.dtype if participation_score is not None else torch.float32
    participation_scale = torch.ones(control_mask.shape[0], dtype=dtype, device=device)
    candidate_mask, candidate_counts = _build_candidate_mask(
        cluster_id=cluster_id,
        control_mask=control_mask,
        candidate_score=participation_score,
        candidate_quantile=state_candidate_quantile,
    )
    cluster_scale_means = {name: 1.0 for name in _CLUSTER_NAMES}
    if participation_score is None or not bool(candidate_mask.any()):
        return participation_scale, candidate_mask, candidate_counts, cluster_scale_means

    for idx, name in enumerate(_CLUSTER_NAMES):
        cluster_candidate = candidate_mask & (cluster_id == idx)
        if bool(cluster_candidate.any()):
            floor = min(max(float(opacity_floors[name]), 0.0), 1.0)
            cluster_scores = torch.clamp(participation_score[cluster_candidate], 0.0, 1.0)
            participation_scale[cluster_candidate] = floor + (1.0 - floor) * cluster_scores
        mask = _cluster_mask(cluster_id, control_mask, idx)
        cluster_scale_means[name] = _masked_mean(participation_scale, mask)

    return participation_scale, candidate_mask, candidate_counts, cluster_scale_means


def _build_stochastic_bernoulli_opacity_action(
    cluster_id: torch.Tensor,
    control_mask: torch.Tensor,
    unified_score: torch.Tensor | None,
    drop_rate_global: float,
    cluster_weights: dict[str, float],
    sample_seed: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, float]]:
    """Build stochastic Bernoulli opacity mask for direct BRPO.

    BRPO formula:
        p_i^{drop} = r * w_cluster(i) * S_i
        m_i ~ Bernoulli(1 - p_i^{drop})
        alpha_i^{eff} = alpha_i * m_i

    Args:
        cluster_id: Per-Gaussian cluster assignment (0=near, 1=mid, 2=far)
        control_mask: Primary control domain mask (population_active for direct BRPO)
        unified_score: BRPO unified importance score S_i
        drop_rate_global: Global drop rate r
        cluster_weights: Per-cluster weight multipliers w_cluster
        sample_seed: Seed for Bernoulli sampling (per_iter mode)
        generator: Optional torch.Generator for reproducibility

    Returns:
        participation_keep_mask: Boolean mask (True = keep, False = drop)
        participation_opacity_scale: Float scale (1.0 for keep, 0.0 for drop)
        cluster_keep_ratio: Per-cluster keep ratio statistics
        cluster_drop_prob_mean: Per-cluster mean drop probability
    """
    device = control_mask.device
    dtype = unified_score.dtype if unified_score is not None else torch.float32
    N = control_mask.shape[0]

    # Initialize: all keep by default
    participation_keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    participation_opacity_scale = torch.ones(N, dtype=dtype, device=device)
    cluster_keep_ratio = {name: 1.0 for name in _CLUSTER_NAMES}
    cluster_drop_prob_mean = {name: 0.0 for name in _CLUSTER_NAMES}

    if unified_score is None or not bool(control_mask.any()):
        return participation_keep_mask, participation_opacity_scale, cluster_keep_ratio, cluster_drop_prob_mean

    # Build drop probability: p_i^{drop} = r * w_cluster * S_i
    drop_prob = torch.zeros(N, dtype=dtype, device=device)
    r = float(min(max(drop_rate_global, 0.0), 1.0))

    for idx, name in enumerate(_CLUSTER_NAMES):
        cluster_mask = control_mask & (cluster_id == idx)
        if not bool(cluster_mask.any()):
            continue

        w_cluster = float(cluster_weights.get(name, 1.0))
        cluster_scores = unified_score[cluster_mask]

        # p_i^{drop} = r * w_cluster * S_i, then clamp to [0, 1]
        cluster_drop_prob = torch.clamp(r * w_cluster * cluster_scores, 0.0, 1.0)
        drop_prob[cluster_mask] = cluster_drop_prob

        # Bernoulli sampling: m_i ~ Bernoulli(1 - p_i^{drop})
        # Sample keep mask (True = keep)
        if generator is not None:
            cluster_keep = torch.bernoulli(1.0 - cluster_drop_prob, generator=generator).bool()
        elif sample_seed is not None:
            # Create a local generator with the seed
            local_gen = torch.Generator(device=device)
            local_gen.manual_seed(sample_seed + idx)  # Different seed per cluster
            cluster_keep = torch.bernoulli(1.0 - cluster_drop_prob, generator=local_gen).bool()
        else:
            # Default: use global random state
            cluster_keep = torch.bernoulli(1.0 - cluster_drop_prob).bool()

        participation_keep_mask[cluster_mask] = cluster_keep
        participation_opacity_scale[cluster_mask] = cluster_keep.float()

        # Statistics
        cluster_keep_ratio[name] = float(cluster_keep.float().mean().item())
        cluster_drop_prob_mean[name] = float(cluster_drop_prob.mean().item())

    # Non-control universe elements: always keep (m_i = 1)
    participation_keep_mask[~control_mask] = True
    participation_opacity_scale[~control_mask] = 1.0

    return participation_keep_mask, participation_opacity_scale, cluster_keep_ratio, cluster_drop_prob_mean


def apply_spgm_state_management(
    gaussians,
    cluster_id: torch.Tensor,
    control_mask: torch.Tensor,
    update_policy: dict,
    manager_mode: str = 'summary_only',
    state_score: torch.Tensor | None = None,
    participation_score: torch.Tensor | None = None,
    unified_score: torch.Tensor | None = None,
    population_support_count: torch.Tensor | None = None,
    state_candidate_quantile: float = 0.5,
    state_base_scale_near: float = 1.0,
    state_base_scale_mid: float = 0.95,
    state_base_scale_far: float = 0.90,
    state_participation_keep_near: float = 1.0,
    state_participation_keep_mid: float = 0.9,
    state_participation_keep_far: float = 0.75,
    state_opacity_floor_near: float = 1.0,
    state_opacity_floor_mid: float = 0.98,
    state_opacity_floor_far: float = 0.92,
    # Direct BRPO parameters
    drop_rate_global: float = 0.05,
    cluster_weight_near: float = 1.0,
    cluster_weight_mid: float = 1.0,
    cluster_weight_far: float = 1.0,
    sample_seed: int | None = None,
    generator: torch.Generator | None = None,
    active_mask: torch.Tensor | None = None,
) -> dict:
    """Return/apply deterministic or stochastic state-management diagnostics.

    Modes:
    - summary_only: diagnostics only, no state action
    - xyz_lr_scale: apply a mild per-Gaussian xyz grad scale before optimizer.step
    - deterministic_participation: prepare a deterministic pseudo-render participation mask
    - deterministic_opacity_participation: prepare a deterministic per-Gaussian opacity scale
    - stochastic_bernoulli_opacity: direct BRPO Bernoulli opacity masking

    Args:
        control_mask: Primary control domain (active or population_active)
        unified_score: BRPO unified score (required for stochastic_bernoulli_opacity)
        drop_rate_global: Global drop rate r for BRPO formula
        cluster_weight_*: Per-cluster weight multipliers
        sample_seed: Seed for Bernoulli sampling (per_iter mode)
        active_mask: Legacy active_mask for backward compatibility (optional)
    """
    # Backward compatibility: if active_mask provided but not control_mask, use active_mask
    if active_mask is not None and control_mask is None:
        control_mask = active_mask

    manager_mode_effective = str(manager_mode or 'summary_only').strip().lower()
    valid_modes = {
        'summary_only', 'xyz_lr_scale', 'deterministic_participation',
        'deterministic_opacity_participation', 'stochastic_bernoulli_opacity',
        'off', 'disabled'
    }
    if manager_mode_effective not in valid_modes:
        raise ValueError(
            f"Unsupported SPGM manager_mode={manager_mode_effective!r}; "
            f"supported values are {sorted(valid_modes - {'off', 'disabled'})}"
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
    opacity_floors = {
        'near': float(state_opacity_floor_near),
        'mid': float(state_opacity_floor_mid),
        'far': float(state_opacity_floor_far),
    }
    cluster_weights = {
        'near': float(cluster_weight_near),
        'mid': float(cluster_weight_mid),
        'far': float(cluster_weight_far),
    }

    control_counts = _cluster_counts(cluster_id, control_mask)
    cluster_state_mean = {}
    cluster_state_p50 = {}
    cluster_participation_mean = {}
    cluster_participation_p50 = {}
    cluster_unified_mean = {}
    cluster_unified_p50 = {}

    candidate_mask, candidate_counts = _build_state_candidate_mask(
        cluster_id=cluster_id,
        control_mask=control_mask,
        state_score=state_score,
        state_candidate_quantile=state_candidate_quantile,
    )

    for idx, name in enumerate(_CLUSTER_NAMES):
        mask = _cluster_mask(cluster_id, control_mask, idx)
        cluster_state_mean[name] = _masked_mean(state_score, mask)
        cluster_state_p50[name] = _masked_p50(state_score, mask)
        cluster_participation_mean[name] = _masked_mean(participation_score, mask)
        cluster_participation_p50[name] = _masked_p50(participation_score, mask)
        cluster_unified_mean[name] = _masked_mean(unified_score, mask)
        cluster_unified_p50[name] = _masked_p50(unified_score, mask)
        if candidate_counts[name] <= 0:
            candidate_counts[name] = control_counts[name] if state_score is None else 0

    xyz_state_scale = _build_xyz_state_scale(
        cluster_id=cluster_id,
        control_mask=control_mask,
        state_score=state_score,
        base_scales=base_scales,
    )

    xyz_scale_mean = _masked_mean(xyz_state_scale, control_mask)
    xyz_scale_mean_near = _masked_mean(xyz_state_scale, _cluster_mask(cluster_id, control_mask, 0))
    xyz_scale_mean_mid = _masked_mean(xyz_state_scale, _cluster_mask(cluster_id, control_mask, 1))
    xyz_scale_mean_far = _masked_mean(xyz_state_scale, _cluster_mask(cluster_id, control_mask, 2))

    grad_pre_state_xyz = 0.0
    grad_post_state_xyz = 0.0
    participation_render_mask = None
    participation_opacity_scale = None
    participation_drop_counts = {name: 0 for name in _CLUSTER_NAMES}
    participation_ratios = {name: 1.0 for name in _CLUSTER_NAMES}
    opacity_scale_means = {name: 1.0 for name in _CLUSTER_NAMES}
    cluster_keep_ratio = {name: 1.0 for name in _CLUSTER_NAMES}
    cluster_drop_prob_mean = {name: 0.0 for name in _CLUSTER_NAMES}
    drop_prob_mean = 0.0
    drop_prob_p50 = 0.0
    sampled_keep_ratio = 1.0
    state_action_applied = False

    if manager_mode_effective == 'xyz_lr_scale':
        grad_pre_state_xyz, grad_post_state_xyz = _apply_scale_(getattr(gaussians, '_xyz', None), xyz_state_scale)
        state_action_applied = grad_pre_state_xyz > 0.0
    elif manager_mode_effective == 'deterministic_participation':
        participation_render_mask, candidate_mask, candidate_counts, participation_drop_counts, participation_ratios = _build_participation_render_mask(
            cluster_id=cluster_id,
            control_mask=control_mask,
            state_score=state_score,
            state_candidate_quantile=state_candidate_quantile,
            keep_ratios=keep_ratios,
        )
        state_action_applied = any(v > 0 for v in participation_drop_counts.values())
    elif manager_mode_effective == 'deterministic_opacity_participation':
        participation_opacity_scale, candidate_mask, candidate_counts, opacity_scale_means = _build_opacity_participation_scale(
            cluster_id=cluster_id,
            control_mask=control_mask,
            participation_score=participation_score,
            state_candidate_quantile=state_candidate_quantile,
            opacity_floors=opacity_floors,
        )
        state_action_applied = bool(((participation_opacity_scale < 0.999999) & control_mask).any())
    elif manager_mode_effective == 'stochastic_bernoulli_opacity':
        participation_keep_mask, participation_opacity_scale, cluster_keep_ratio, cluster_drop_prob_mean = _build_stochastic_bernoulli_opacity_action(
            cluster_id=cluster_id,
            control_mask=control_mask,
            unified_score=unified_score,
            drop_rate_global=drop_rate_global,
            cluster_weights=cluster_weights,
            sample_seed=sample_seed,
            generator=generator,
        )
        # For stochastic mode, render_mask is derived from opacity_scale > 0.5
        participation_render_mask = participation_opacity_scale > 0.5
        state_action_applied = bool((participation_opacity_scale < 0.999999).any())

        # Compute drop probability statistics
        if unified_score is not None and control_mask.any():
            # Re-compute drop_prob for statistics
            r = float(min(max(drop_rate_global, 0.0), 1.0))
            drop_prob_tensor = torch.zeros_like(unified_score)
            for idx, name in enumerate(_CLUSTER_NAMES):
                cluster_mask = control_mask & (cluster_id == idx)
                if cluster_mask.any():
                    w_cluster = cluster_weights[name]
                    drop_prob_tensor[cluster_mask] = torch.clamp(r * w_cluster * unified_score[cluster_mask], 0.0, 1.0)
            drop_prob_mean = float(drop_prob_tensor[control_mask].mean().item())
            drop_prob_p50 = float(drop_prob_tensor[control_mask].median().item())
            sampled_keep_ratio = float(participation_keep_mask[control_mask].float().mean().item())

    state_participation_ratio = _masked_ratio(participation_render_mask, control_mask) if participation_render_mask is not None else 1.0
    state_opacity_scale_mean = _masked_mean(participation_opacity_scale, control_mask) if participation_opacity_scale is not None else 1.0
    state_opacity_scale_mean_near = opacity_scale_means['near']
    state_opacity_scale_mean_mid = opacity_scale_means['mid']
    state_opacity_scale_mean_far = opacity_scale_means['far']

    return {
        'manager_mode_effective': manager_mode_effective,
        'state_action_applied': state_action_applied,
        'state_lr_scale_near': xyz_scale_mean_near,
        'state_lr_scale_mid': xyz_scale_mean_mid,
        'state_lr_scale_far': xyz_scale_mean_far,
        'state_opacity_decay_near': max(0.0, 1.0 - state_opacity_scale_mean_near),
        'state_opacity_decay_mid': max(0.0, 1.0 - state_opacity_scale_mean_mid),
        'state_opacity_decay_far': max(0.0, 1.0 - state_opacity_scale_mean_far),
        'state_opacity_scale_mean': state_opacity_scale_mean,
        'state_opacity_scale_mean_near': state_opacity_scale_mean_near,
        'state_opacity_scale_mean_mid': state_opacity_scale_mean_mid,
        'state_opacity_scale_mean_far': state_opacity_scale_mean_far,
        'state_candidate_count_near': candidate_counts['near'],
        'state_candidate_count_mid': candidate_counts['mid'],
        'state_candidate_count_far': candidate_counts['far'],
        'state_score_mean': _masked_mean(state_score, control_mask),
        'state_score_p50': _masked_p50(state_score, control_mask),
        'participation_score_mean': _masked_mean(participation_score, control_mask),
        'participation_score_p50': _masked_p50(participation_score, control_mask),
        'population_support_mean': _masked_mean(population_support_count, control_mask),
        'state_score_mean_near': cluster_state_mean['near'],
        'state_score_mean_mid': cluster_state_mean['mid'],
        'state_score_mean_far': cluster_state_mean['far'],
        'state_score_p50_near': cluster_state_p50['near'],
        'state_score_p50_mid': cluster_state_p50['mid'],
        'state_score_p50_far': cluster_state_p50['far'],
        'participation_score_mean_near': cluster_participation_mean['near'],
        'participation_score_mean_mid': cluster_participation_mean['mid'],
        'participation_score_mean_far': cluster_participation_mean['far'],
        'participation_score_p50_near': cluster_participation_p50['near'],
        'participation_score_p50_mid': cluster_participation_p50['mid'],
        'participation_score_p50_far': cluster_participation_p50['far'],
        'unified_score_mean_near': cluster_unified_mean['near'],
        'unified_score_mean_mid': cluster_unified_mean['mid'],
        'unified_score_mean_far': cluster_unified_mean['far'],
        'unified_score_p50_near': cluster_unified_p50['near'],
        'unified_score_p50_mid': cluster_unified_p50['mid'],
        'unified_score_p50_far': cluster_unified_p50['far'],
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
        'spgm_drop_rate_global': float(min(max(drop_rate_global, 0.0), 1.0)),
        'spgm_drop_prob_mean': drop_prob_mean,
        'spgm_drop_prob_p50': drop_prob_p50,
        'spgm_drop_prob_mean_near': cluster_drop_prob_mean['near'],
        'spgm_drop_prob_mean_mid': cluster_drop_prob_mean['mid'],
        'spgm_drop_prob_mean_far': cluster_drop_prob_mean['far'],
        'spgm_sampled_keep_ratio': sampled_keep_ratio,
        'spgm_sampled_keep_ratio_near': cluster_keep_ratio['near'],
        'spgm_sampled_keep_ratio_mid': cluster_keep_ratio['mid'],
        'spgm_sampled_keep_ratio_far': cluster_keep_ratio['far'],
        'participation_render_mask': participation_render_mask,
        'participation_opacity_scale': participation_opacity_scale,
        'participation_keep_mask': participation_keep_mask if manager_mode_effective == 'stochastic_bernoulli_opacity' else None,
    }