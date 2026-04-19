"""SPGM weighting / ranking / state score builder.

Phase B2 keeps selector behavior backward compatible while introducing:
- explicit `weight_score`
- explicit `ranking_score`
- explicit `state_score`
- `struct_density` as an optional density-side proxy
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
    active_mask: torch.Tensor,
    num_clusters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition active Gaussians by depth quantiles."""
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype

    cluster_id = torch.full((N,), -1, dtype=torch.long, device=device)
    depth_score = torch.zeros(N, dtype=dtype, device=device)
    if not active_mask.any():
        return cluster_id, depth_score

    active_depths = depth_value[active_mask]
    z_min = float(active_depths.min().item())
    z_max = float(active_depths.max().item())
    if z_max <= z_min + 1e-8:
        depth_score[active_mask] = 1.0
        cluster_id[active_mask] = 0
        return cluster_id, depth_score

    depth_score[active_mask] = 1.0 - (active_depths - z_min) / (z_max - z_min)
    sorted_indices = torch.argsort(active_depths)
    num_active = int(active_depths.shape[0])
    effective_clusters = min(num_clusters, num_active)
    if effective_clusters < 1:
        return cluster_id, depth_score

    bin_sizes = num_active // effective_clusters
    remainder = num_active % effective_clusters
    cluster_assignments = torch.zeros(num_active, dtype=torch.long, device=device)

    start_idx = 0
    for k in range(effective_clusters):
        bin_size = bin_sizes + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        end_idx = min(start_idx + bin_size, num_active)
        cluster_assignments[sorted_indices[start_idx:end_idx]] = k
        start_idx = end_idx

    active_indices = torch.where(active_mask)[0]
    cluster_id[active_indices] = cluster_assignments
    return cluster_id, depth_score


def compute_density_entropy(
    density_values: torch.Tensor,
    active_mask: torch.Tensor,
    entropy_bins: int = 32,
) -> float:
    """Compute normalized Shannon entropy of an active Gaussian scalar distribution."""
    if not active_mask.any():
        return 0.0

    active_values = density_values[active_mask]
    min_val = float(active_values.min().item())
    max_val = float(active_values.max().item())
    if max_val <= min_val + 1e-8:
        return 0.0

    bins = max(int(entropy_bins), 2)
    normalized = (active_values - min_val) / (max_val - min_val)
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
) -> dict:
    """Build weighting / ranking / state scores from SPGM statistics."""
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype

    cluster_id, depth_score = build_depth_partition(depth_value, active_mask, num_clusters)

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

    density_entropy = compute_density_entropy(density_values, active_mask, entropy_bins)
    density_score = torch.zeros(N, dtype=dtype, device=device)
    if active_mask.any():
        normalized_density = _normalize_masked(density_values, active_mask)
        entropy_factor = 1.0 - beta_entropy * density_entropy
        density_score[active_mask] = normalized_density[active_mask] * entropy_factor + gamma_entropy * density_entropy

    importance_raw = alpha_depth * depth_score + (1.0 - alpha_depth) * density_score

    support_norm = torch.zeros(N, dtype=dtype, device=device)
    population_support_norm = torch.zeros(N, dtype=dtype, device=device)
    weight_score = torch.zeros(N, dtype=dtype, device=device)
    ranking_score = torch.zeros(N, dtype=dtype, device=device)
    state_score = torch.zeros(N, dtype=dtype, device=device)
    ranking_mode_effective = str(ranking_mode or 'v1').strip().lower()
    lambda_support_rank = min(max(float(lambda_support_rank), 0.0), 1.0)

    if active_mask.any():
        support_norm = _normalize_masked(support_count, active_mask)
        population_support_norm = _normalize_masked(population_support_count, active_mask)
        state_density_norm = _normalize_masked(struct_density_proxy, active_mask)

        support_modifier = support_norm[active_mask].pow(support_eta)
        weight_score[active_mask] = importance_raw[active_mask] * support_modifier

        if ranking_mode_effective == 'v1':
            ranking_score[active_mask] = weight_score[active_mask]
        elif ranking_mode_effective == 'support_blend':
            ranking_score[active_mask] = (
                (1.0 - lambda_support_rank) * importance_raw[active_mask]
                + lambda_support_rank * support_norm[active_mask]
            )
        else:
            raise ValueError(
                f"Unsupported SPGM ranking_mode={ranking_mode_effective!r}; supported values are 'v1' and 'support_blend'"
            )

        state_score[active_mask] = (
            0.45 * state_density_norm[active_mask]
            + 0.35 * population_support_norm[active_mask]
            + 0.20 * depth_score[active_mask]
        )

        weight_score = torch.clamp(weight_score, 0.0, 1.0)
        ranking_score = torch.clamp(ranking_score, 0.0, 1.0)
        state_score = torch.clamp(state_score, 0.0, 1.0)

    cluster_counts = {
        'near': int((cluster_id == 0).sum().item()),
        'mid': int((cluster_id == 1).sum().item()),
        'far': int((cluster_id == 2).sum().item()),
    }

    if active_mask.any():
        active_weight = weight_score[active_mask]
        active_ranking = ranking_score[active_mask]
        active_state = state_score[active_mask]
        active_support_norm = support_norm[active_mask]
        active_population_support = population_support_count[active_mask]
        active_struct_density = struct_density_proxy[active_mask]
        importance_mean = float(active_weight.mean().item())
        importance_p50 = float(active_weight.median().item())
        ranking_score_mean = float(active_ranking.mean().item())
        ranking_score_p50 = float(active_ranking.median().item())
        state_score_mean = float(active_state.mean().item())
        state_score_p50 = float(active_state.median().item())
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
        'cluster_count_near': cluster_counts['near'],
        'cluster_count_mid': cluster_counts['mid'],
        'cluster_count_far': cluster_counts['far'],
        'importance_mean': importance_mean,
        'importance_p50': importance_p50,
        'ranking_score_mean': ranking_score_mean,
        'ranking_score_p50': ranking_score_p50,
        'state_score_mean': state_score_mean,
        'state_score_p50': state_score_p50,
        'support_norm_mean': support_norm_mean,
        'population_support_mean': population_support_mean,
        'struct_density_mean': struct_density_mean,
        'ranking_mode_effective': ranking_mode_effective,
        'depth_entropy': 0.0,
        'density_mode_effective': density_mode,
    }
