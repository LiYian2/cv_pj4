"""SPGM weighting / ranking / state score builder.

Phase B2 keeps selector behavior backward compatible while introducing:
- explicit `weight_score`
- explicit `ranking_score`
- explicit `state_score`
- explicit `participation_score`
- `struct_density` as an optional density-side proxy

G-BRPO-0 adds direct BRPO unified score path:
- score_semantics='brpo_unified_v1': S_i = alpha * hat_s_i^{(z)} + (1-alpha) * hat_s_i^{(rho)}
- unified_score: the paper's unified importance score for Bernoulli drop probability
"""

from __future__ import annotations

import math

import torch


def _normalize_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values)
    if not mask.any():
        return out
    active = values[mask]
    v_min = float(active.min().item())
    v_max = float(active.max().item())
    if v_max <= v_min + 1e-8:
        out[mask] = 0.5
        return out
    out[mask] = (active - v_min) / (v_max - v_min)
    return out


def build_depth_partition(
    depth_value: torch.Tensor,
    control_mask: torch.Tensor,
    num_clusters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition control universe Gaussians by depth quantiles.

    Args:
        control_mask: The primary control domain mask (active or population_active)
    """
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype

    cluster_id = torch.full((N,), -1, dtype=torch.long, device=device)
    depth_score = torch.zeros(N, dtype=dtype, device=device)
    if not control_mask.any():
        return cluster_id, depth_score

    control_depths = depth_value[control_mask]
    z_min = float(control_depths.min().item())
    z_max = float(control_depths.max().item())
    if z_max <= z_min + 1e-8:
        depth_score[control_mask] = 1.0
        cluster_id[control_mask] = 0
        return cluster_id, depth_score

    # Depth score: closer (smaller z) -> higher score (1.0), farther -> lower (0.0)
    depth_score[control_mask] = 1.0 - (control_depths - z_min) / (z_max - z_min)
    sorted_indices = torch.argsort(control_depths)
    num_control = int(control_depths.shape[0])
    effective_clusters = min(num_clusters, num_control)
    if effective_clusters < 1:
        return cluster_id, depth_score

    bin_sizes = num_control // effective_clusters
    remainder = num_control % effective_clusters
    cluster_assignments = torch.zeros(num_control, dtype=torch.long, device=device)

    start_idx = 0
    for k in range(effective_clusters):
        bin_size = bin_sizes + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        end_idx = min(start_idx + bin_size, num_control)
        cluster_assignments[sorted_indices[start_idx:end_idx]] = k
        start_idx = end_idx

    control_indices = torch.where(control_mask)[0]
    cluster_id[control_indices] = cluster_assignments
    return cluster_id, depth_score


def compute_density_entropy(
    density_values: torch.Tensor,
    control_mask: torch.Tensor,
    entropy_bins: int = 32,
) -> float:
    """Compute normalized Shannon entropy of control universe scalar distribution."""
    if not control_mask.any():
        return 0.0

    control_values = density_values[control_mask]
    min_val = float(control_values.min().item())
    max_val = float(control_values.max().item())
    if max_val <= min_val + 1e-8:
        return 0.0

    bins = max(int(entropy_bins), 2)
    normalized = (control_values - min_val) / (max_val - min_val)
    bin_indices = torch.clamp((normalized * (bins - 1)).long(), min=0, max=bins - 1)
    hist = torch.bincount(bin_indices, minlength=bins).to(dtype=torch.float32, device=density_values.device)

    total = hist.sum()
    if total <= 0:
        return 0.0

    probs = hist / total
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + eps)).item()
    max_entropy = math.log(float(bins))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return min(max(normalized_entropy, 0.0), 1.0)


def build_spgm_brpo_unified_score(
    depth_value: torch.Tensor,
    density_proxy: torch.Tensor,
    support_count: torch.Tensor,
    control_mask: torch.Tensor,
    num_clusters: int = 3,
    alpha: float = 0.5,
    entropy_bins: int = 32,
    density_mode: str = 'opacity_support',
    struct_density_proxy: torch.Tensor | None = None,
    population_support_count: torch.Tensor | None = None,
) -> dict:
    """Build BRPO unified score S_i from paper formula.

    Paper formula: S_i = alpha * hat_s_i^{(z)} + (1-alpha) * hat_s_i^{(rho)}
    where:
    - hat_s_i^{(z)}: normalized depth score (closer -> higher importance)
    - hat_s_i^{(rho)}: normalized density/support score

    Args:
        control_mask: Primary control domain (population_active for direct BRPO)
        alpha: Weight for depth vs density in unified score
    """
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype

    cluster_id, depth_score = build_depth_partition(depth_value, control_mask, num_clusters)

    population_support_count = support_count if population_support_count is None else population_support_count
    struct_density_proxy = density_proxy if struct_density_proxy is None else struct_density_proxy

    density_mode = str(density_mode or 'opacity_support').strip().lower()
    if density_mode == 'opacity_support':
        density_values = density_proxy
    elif density_mode in {'support', 'support_only'}:
        density_values = support_count
    elif density_mode == 'struct_density':
        density_values = struct_density_proxy
    else:
        raise ValueError(
            f"Unsupported SPGM density_mode={density_mode!r}; supported values are 'opacity_support', 'support', and 'struct_density'"
        )

    density_entropy = compute_density_entropy(density_values, control_mask, entropy_bins)

    # Normalize depth and density scores over control universe
    normalized_depth = _normalize_masked(depth_score, control_mask)
    normalized_density = _normalize_masked(density_values, control_mask)

    # BRPO unified score: S_i = alpha * hat_s_z + (1-alpha) * hat_s_rho
    unified_score = alpha * normalized_depth + (1.0 - alpha) * normalized_density
    unified_score = torch.clamp(unified_score, 0.0, 1.0)

    # Compute drop probability (pre-clip): p_i^{drop} = r * w_cluster * S_i
    # Note: actual drop_prob computation happens in manager with r and cluster weights
    # Here we just prepare the unified score

    cluster_counts = {
        'near': int((cluster_id == 0).sum().item()),
        'mid': int((cluster_id == 1).sum().item()),
        'far': int((cluster_id == 2).sum().item()),
    }

    if control_mask.any():
        control_unified = unified_score[control_mask]
        control_depth = normalized_depth[control_mask]
        control_density = normalized_density[control_mask]
        unified_mean = float(control_unified.mean().item())
        unified_p50 = float(control_unified.median().item())
        depth_mean = float(control_depth.mean().item())
        density_mean = float(control_density.mean().item())

        # Per-cluster stats
        unified_mean_near = float(unified_score[control_mask & (cluster_id == 0)].mean().item()) if cluster_counts['near'] > 0 else 0.0
        unified_mean_mid = float(unified_score[control_mask & (cluster_id == 1)].mean().item()) if cluster_counts['mid'] > 0 else 0.0
        unified_mean_far = float(unified_score[control_mask & (cluster_id == 2)].mean().item()) if cluster_counts['far'] > 0 else 0.0
        unified_p50_near = float(unified_score[control_mask & (cluster_id == 0)].median().item()) if cluster_counts['near'] > 0 else 0.0
        unified_p50_mid = float(unified_score[control_mask & (cluster_id == 1)].median().item()) if cluster_counts['mid'] > 0 else 0.0
        unified_p50_far = float(unified_score[control_mask & (cluster_id == 2)].median().item()) if cluster_counts['far'] > 0 else 0.0
    else:
        unified_mean = 0.0
        unified_p50 = 0.0
        depth_mean = 0.0
        density_mean = 0.0
        unified_mean_near = 0.0
        unified_mean_mid = 0.0
        unified_mean_far = 0.0
        unified_p50_near = 0.0
        unified_p50_mid = 0.0
        unified_p50_far = 0.0

    return {
        'cluster_id': cluster_id,
        'depth_score': normalized_depth,
        'density_entropy': density_entropy,
        'density_score': normalized_density,
        'unified_score': unified_score,
        'cluster_count_near': cluster_counts['near'],
        'cluster_count_mid': cluster_counts['mid'],
        'cluster_count_far': cluster_counts['far'],
        'unified_score_mean': unified_mean,
        'unified_score_p50': unified_p50,
        'depth_score_mean': depth_mean,
        'density_score_mean': density_mean,
        'unified_score_mean_near': unified_mean_near,
        'unified_score_mean_mid': unified_mean_mid,
        'unified_score_mean_far': unified_mean_far,
        'unified_score_p50_near': unified_p50_near,
        'unified_score_p50_mid': unified_p50_mid,
        'unified_score_p50_far': unified_p50_far,
        'density_mode_effective': density_mode,
        'score_semantics': 'brpo_unified_v1',
        'brpo_alpha_effective': alpha,
    }


def build_spgm_importance_score(
    depth_value: torch.Tensor,
    density_proxy: torch.Tensor,
    support_count: torch.Tensor,
    active_mask: torch.Tensor,
    num_clusters: int = 3,
    alpha_depth: float = 0.5,
    beta_entropy: float = 0.5,
    gamma_entropy: float = 0.5,
    support_eta: float = 0.5,
    entropy_bins: int = 32,
    density_mode: str = 'opacity_support',
    ranking_mode: str = 'v1',
    lambda_support_rank: float = 0.0,
    population_support_count: torch.Tensor | None = None,
    struct_density_proxy: torch.Tensor | None = None,
    score_semantics: str = 'legacy_v1',
    control_mask: torch.Tensor | None = None,
    spgm_brpo_alpha: float = 0.5,
) -> dict:
    """Build weighting / ranking / state / participation scores from SPGM statistics.

    Args:
        score_semantics: 'legacy_v1' uses old multi-score bundle; 'brpo_unified_v1'
            uses paper unified S_i score for direct BRPO path.
        control_mask: Primary control domain mask. If None, defaults to active_mask.
        spgm_brpo_alpha: Alpha for BRPO unified score (depth vs density weight).
    """
    # Determine control universe
    if control_mask is None:
        control_mask = active_mask

    score_semantics_effective = str(score_semantics or 'legacy_v1').strip().lower()

    # Direct BRPO path: use unified score builder
    if score_semantics_effective == 'brpo_unified_v1':
        brpo_result = build_spgm_brpo_unified_score(
            depth_value=depth_value,
            density_proxy=density_proxy,
            support_count=support_count,
            control_mask=control_mask,
            num_clusters=num_clusters,
            alpha=spgm_brpo_alpha,
            entropy_bins=entropy_bins,
            density_mode=density_mode,
            struct_density_proxy=struct_density_proxy,
            population_support_count=population_support_count,
        )
        # Add legacy-compatible fields for diagnostics (but these should not be used for action)
        N = depth_value.shape[0]
        device = depth_value.device
        dtype = depth_value.dtype

        # Legacy fields (zero-filled, only for backward compatibility in history)
        legacy_zeros = torch.zeros(N, dtype=dtype, device=device)
        support_norm = _normalize_masked(support_count, control_mask)
        pop_support = population_support_count if population_support_count is not None else support_count
        population_support_norm = _normalize_masked(pop_support, control_mask)
        struct_density_proxy = density_proxy if struct_density_proxy is None else struct_density_proxy

        # Legacy-compatible summary scalars
        if control_mask.any():
            importance_mean = float(brpo_result['unified_score'][control_mask].mean().item())
            importance_p50 = float(brpo_result['unified_score'][control_mask].median().item())
            ranking_score_mean = importance_mean
            ranking_score_p50 = importance_p50
            state_score_mean = 0.0
            state_score_p50 = 0.0
            participation_score_mean = importance_mean
            participation_score_p50 = importance_p50
            support_norm_mean = float(support_norm[control_mask].mean().item())
            population_support_mean = float(pop_support[control_mask].mean().item())
            struct_density_mean = float(struct_density_proxy[control_mask].mean().item())
        else:
            importance_mean = 0.0
            importance_p50 = 0.0
            ranking_score_mean = 0.0
            ranking_score_p50 = 0.0
            state_score_mean = 0.0
            state_score_p50 = 0.0
            participation_score_mean = 0.0
            participation_score_p50 = 0.0
            support_norm_mean = 0.0
            population_support_mean = 0.0
            struct_density_mean = 0.0

        brpo_result.update({
            'importance_raw': brpo_result['unified_score'],
            'support_norm': support_norm,
            'population_support_norm': population_support_norm,
            'importance_score': brpo_result['unified_score'],
            'weight_score': brpo_result['unified_score'],
            'ranking_score': brpo_result['unified_score'],
            'state_score': legacy_zeros,
            'participation_score': brpo_result['unified_score'],
            'ranking_mode_effective': 'brpo_unified_v1',
            'depth_entropy': 0.0,
            'importance_mean': importance_mean,
            'importance_p50': importance_p50,
            'ranking_score_mean': ranking_score_mean,
            'ranking_score_p50': ranking_score_p50,
            'state_score_mean': state_score_mean,
            'state_score_p50': state_score_p50,
            'participation_score_mean': participation_score_mean,
            'participation_score_p50': participation_score_p50,
            'support_norm_mean': support_norm_mean,
            'population_support_mean': population_support_mean,
            'struct_density_mean': struct_density_mean,
        })
        return brpo_result

    # Legacy path: original multi-score builder
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype

    cluster_id, depth_score = build_depth_partition(depth_value, control_mask, num_clusters)

    population_support_count = support_count if population_support_count is None else population_support_count
    struct_density_proxy = density_proxy if struct_density_proxy is None else struct_density_proxy

    density_mode = str(density_mode or 'opacity_support').strip().lower()
    if density_mode == 'opacity_support':
        density_values = density_proxy
    elif density_mode in {'support', 'support_only'}:
        density_values = support_count
    elif density_mode == 'struct_density':
        density_values = struct_density_proxy
    else:
        raise ValueError(
            f"Unsupported SPGM density_mode={density_mode!r}; supported values are 'opacity_support', 'support', and 'struct_density'"
        )

    density_entropy = compute_density_entropy(density_values, control_mask, entropy_bins)
    density_score = torch.zeros(N, dtype=dtype, device=device)
    if control_mask.any():
        normalized_density = _normalize_masked(density_values, control_mask)
        entropy_factor = 1.0 - beta_entropy * density_entropy
        density_score[control_mask] = normalized_density[control_mask] * entropy_factor + gamma_entropy * density_entropy

    importance_raw = alpha_depth * depth_score + (1.0 - alpha_depth) * density_score

    support_norm = torch.zeros(N, dtype=dtype, device=device)
    population_support_norm = torch.zeros(N, dtype=dtype, device=device)
    weight_score = torch.zeros(N, dtype=dtype, device=device)
    ranking_score = torch.zeros(N, dtype=dtype, device=device)
    state_score = torch.zeros(N, dtype=dtype, device=device)
    participation_score = torch.zeros(N, dtype=dtype, device=device)
    ranking_mode_effective = str(ranking_mode or 'v1').strip().lower()
    lambda_support_rank = min(max(float(lambda_support_rank), 0.0), 1.0)

    if control_mask.any():
        support_norm = _normalize_masked(support_count, control_mask)
        population_support_norm = _normalize_masked(population_support_count, control_mask)
        state_density_norm = _normalize_masked(struct_density_proxy, control_mask)

        support_modifier = support_norm[control_mask].pow(support_eta)
        weight_score[control_mask] = importance_raw[control_mask] * support_modifier

        if ranking_mode_effective == 'v1':
            ranking_score[control_mask] = weight_score[control_mask]
        elif ranking_mode_effective == 'support_blend':
            ranking_score[control_mask] = (
                (1.0 - lambda_support_rank) * importance_raw[control_mask]
                + lambda_support_rank * support_norm[control_mask]
            )
        else:
            raise ValueError(
                f"Unsupported SPGM ranking_mode={ranking_mode_effective!r}; supported values are 'v1' and 'support_blend'"
            )

        state_score[control_mask] = (
            0.45 * state_density_norm[control_mask]
            + 0.35 * population_support_norm[control_mask]
            + 0.20 * depth_score[control_mask]
        )
        participation_score[control_mask] = (
            0.80 * importance_raw[control_mask]
            + 0.20 * population_support_norm[control_mask]
        )

        weight_score = torch.clamp(weight_score, 0.0, 1.0)
        ranking_score = torch.clamp(ranking_score, 0.0, 1.0)
        state_score = torch.clamp(state_score, 0.0, 1.0)
        participation_score = torch.clamp(participation_score, 0.0, 1.0)

    cluster_counts = {
        'near': int((cluster_id == 0).sum().item()),
        'mid': int((cluster_id == 1).sum().item()),
        'far': int((cluster_id == 2).sum().item()),
    }

    if control_mask.any():
        active_weight = weight_score[control_mask]
        active_ranking = ranking_score[control_mask]
        active_state = state_score[control_mask]
        active_participation = participation_score[control_mask]
        active_support_norm = support_norm[control_mask]
        active_population_support = population_support_count[control_mask]
        active_struct_density = struct_density_proxy[control_mask]
        importance_mean = float(active_weight.mean().item())
        importance_p50 = float(active_weight.median().item())
        ranking_score_mean = float(active_ranking.mean().item())
        ranking_score_p50 = float(active_ranking.median().item())
        state_score_mean = float(active_state.mean().item())
        state_score_p50 = float(active_state.median().item())
        participation_score_mean = float(active_participation.mean().item())
        participation_score_p50 = float(active_participation.median().item())
        support_norm_mean = float(active_support_norm.mean().item())
        population_support_mean = float(active_population_support.mean().item())
        struct_density_mean = float(active_struct_density.mean().item())
    else:
        importance_mean = 0.0
        importance_p50 = 0.0
        ranking_score_mean = 0.0
        ranking_score_p50 = 0.0
        state_score_mean = 0.0
        state_score_p50 = 0.0
        participation_score_mean = 0.0
        participation_score_p50 = 0.0
        support_norm_mean = 0.0
        population_support_mean = 0.0
        struct_density_mean = 0.0

    return {
        'cluster_id': cluster_id,
        'depth_score': depth_score,
        'density_entropy': density_entropy,
        'density_score': density_score,
        'importance_raw': importance_raw,
        'support_norm': support_norm,
        'population_support_norm': population_support_norm,
        'importance_score': weight_score,
        'weight_score': weight_score,
        'ranking_score': ranking_score,
        'state_score': state_score,
        'participation_score': participation_score,
        'unified_score': torch.zeros_like(weight_score),  # Legacy path: no unified score
        'cluster_count_near': cluster_counts['near'],
        'cluster_count_mid': cluster_counts['mid'],
        'cluster_count_far': cluster_counts['far'],
        'importance_mean': importance_mean,
        'importance_p50': importance_p50,
        'ranking_score_mean': ranking_score_mean,
        'ranking_score_p50': ranking_score_p50,
        'state_score_mean': state_score_mean,
        'state_score_p50': state_score_p50,
        'participation_score_mean': participation_score_mean,
        'participation_score_p50': participation_score_p50,
        'support_norm_mean': support_norm_mean,
        'population_support_mean': population_support_mean,
        'struct_density_mean': struct_density_mean,
        'ranking_mode_effective': ranking_mode_effective,
        'depth_entropy': 0.0,
        'density_mode_effective': density_mode,
        'score_semantics': score_semantics_effective,
        'unified_score_mean': 0.0,
        'unified_score_p50': 0.0,
    }