"""SPGM importance / ranking score builder.

Build per-Gaussian scores from statistics:
- Depth partition (K=3 quantile split: near/mid/far)
- depth_score: 1 - (z - z_min) / (z_max - z_min + eps)
- density_entropy: histogram entropy of density_proxy
- density_score: rho_norm * (1 - beta*Hbar) + gamma*Hbar
- importance_score: weighting score after support modifier
- ranking_score: optional selector-specific score, decoupled from weighting
"""

from __future__ import annotations

import math

import torch


def build_depth_partition(
    depth_value: torch.Tensor,
    active_mask: torch.Tensor,
    num_clusters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition active Gaussians by depth quantiles.
    
    Args:
        depth_value: Tensor[N] of depths
        active_mask: BoolTensor[N] of active Gaussians
        num_clusters: Number of quantile clusters (default 3: near/mid/far)
        
    Returns:
        cluster_id: Tensor[N] with cluster indices (-1 for inactive)
        depth_score: Tensor[N] normalized depth scores (higher = closer)
    """
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype
    
    cluster_id = torch.full((N,), -1, dtype=torch.long, device=device)
    depth_score = torch.zeros(N, dtype=dtype, device=device)
    
    if not active_mask.any():
        return cluster_id, depth_score
    
    # Get active depths
    active_depths = depth_value[active_mask]
    
    # Handle degenerate case: all same depth
    z_min = float(active_depths.min().item())
    z_max = float(active_depths.max().item())
    
    if z_max <= z_min + 1e-8:
        # All same depth: score = 1 for active, cluster = 0
        depth_score[active_mask] = 1.0
        cluster_id[active_mask] = 0
        return cluster_id, depth_score
    
    # Depth score: closer = higher score
    # s_z = 1 - (z - z_min) / (z_max - z_min)
    depth_score[active_mask] = 1.0 - (active_depths - z_min) / (z_max - z_min)
    
    # Quantile partition
    # Sort active depths and split into K bins
    sorted_indices = torch.argsort(active_depths)
    num_active = int(active_depths.shape[0])
    
    # Adaptive cluster count if too few active Gaussians
    effective_clusters = min(num_clusters, num_active)
    if effective_clusters < 1:
        return cluster_id, depth_score
    
    # Compute quantile boundaries
    bin_sizes = num_active // effective_clusters
    remainder = num_active % effective_clusters
    
    # Assign cluster IDs based on sorted order
    cluster_assignments = torch.zeros(num_active, dtype=torch.long, device=device)
    
    start_idx = 0
    for k in range(effective_clusters):
        # This bin gets base size + 1 extra if remainder > 0
        bin_size = bin_sizes + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        
        end_idx = min(start_idx + bin_size, num_active)
        cluster_assignments[sorted_indices[start_idx:end_idx]] = k
        start_idx = end_idx
    
    # Map back to full tensor
    active_indices = torch.where(active_mask)[0]
    cluster_id[active_indices] = cluster_assignments
    
    return cluster_id, depth_score


def compute_density_entropy(
    density_values: torch.Tensor,
    active_mask: torch.Tensor,
    entropy_bins: int = 32,
) -> float:
    """Compute normalized Shannon entropy of an active Gaussian scalar distribution.

    Normalization follows H_bar = H / log(B), where B is the histogram bin count.
    """
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
) -> dict:
    """Build per-Gaussian weighting and ranking scores from statistics.
    
    Args:
        depth_value: Tensor[N] of camera-space depths
        density_proxy: Tensor[N] of density proxies
        support_count: Tensor[N] of support counts
        active_mask: BoolTensor[N] of active Gaussians
        num_clusters: Number of depth quantile clusters (default 3)
        alpha_depth: Weight for depth score (default 0.5)
        beta_entropy: Entropy penalty weight (default 0.5)
        gamma_entropy: Entropy boost weight (default 0.5)
        support_eta: Support modifier exponent for weighting score (default 0.5)
        entropy_bins: Number of histogram bins for entropy (default 32)
        density_mode: Which scalar to use for density-side scoring/entropy.
            Supported: 'opacity_support' (default), 'support'.
        ranking_mode: Selector ranking mode. Supported: 'v1', 'support_blend'.
        lambda_support_rank: Support blend coefficient for ranking_mode='support_blend'.
        
    Returns:
        dict with tensors and summary stats:
            - cluster_id: Tensor[N] (-1 for inactive, 0/1/2 for near/mid/far)
            - depth_score: Tensor[N] in [0, 1] (higher = closer)
            - density_entropy: float in [0, 1]
            - density_score: Tensor[N] in [0, 1]
            - importance_raw: Tensor[N] before support modifier
            - support_norm: Tensor[N] normalized support in [0, 1]
            - importance_score: Tensor[N] weighting score in [0, 1]
            - ranking_score: Tensor[N] selector ranking score in [0, 1]
            - cluster_count_near/mid/far: int
            - importance_mean/p50: float
            - ranking_score_mean/p50: float
            - support_norm_mean: float
            - depth_entropy: float (placeholder for future)
    """
    N = depth_value.shape[0]
    device = depth_value.device
    dtype = depth_value.dtype
    
    # Depth partition and score
    cluster_id, depth_score = build_depth_partition(
        depth_value, active_mask, num_clusters
    )
    
    # Select density-side scalar based on configured mode.
    density_mode = str(density_mode or 'opacity_support').strip().lower()
    if density_mode == 'opacity_support':
        density_values = density_proxy
    elif density_mode in {'support', 'support_only'}:
        density_values = support_count
    else:
        raise ValueError(
            f"Unsupported SPGM density_mode={density_mode!r}; supported values are 'opacity_support' and 'support'"
        )

    # Density entropy
    density_entropy = compute_density_entropy(
        density_values, active_mask, entropy_bins
    )
    
    # Normalize density-side scalar for active Gaussians
    density_score = torch.zeros(N, dtype=dtype, device=device)
    if active_mask.any():
        active_density_values = density_values[active_mask]
        d_min = float(active_density_values.min().item())
        d_max = float(active_density_values.max().item())
        
        if d_max > d_min + 1e-8:
            normalized_density = (active_density_values - d_min) / (d_max - d_min)
        else:
            normalized_density = torch.full_like(active_density_values, 0.5)
        
        # Entropy-aware density score
        entropy_factor = 1.0 - beta_entropy * density_entropy
        density_score[active_mask] = normalized_density * entropy_factor + gamma_entropy * density_entropy
    
    # Raw importance: alpha * depth + (1-alpha) * density
    importance_raw = alpha_depth * depth_score + (1.0 - alpha_depth) * density_score

    support_norm = torch.zeros(N, dtype=dtype, device=device)
    importance_score = torch.zeros(N, dtype=dtype, device=device)
    ranking_score = torch.zeros(N, dtype=dtype, device=device)
    ranking_mode_effective = str(ranking_mode or 'v1').strip().lower()
    lambda_support_rank = min(max(float(lambda_support_rank), 0.0), 1.0)

    if active_mask.any():
        active_support = support_count[active_mask]
        support_max = float(active_support.max().item()) + 1e-8
        support_norm[active_mask] = active_support / support_max
        support_modifier = support_norm[active_mask].pow(support_eta)
        importance_score[active_mask] = importance_raw[active_mask] * support_modifier

        if ranking_mode_effective == 'v1':
            ranking_score[active_mask] = importance_score[active_mask]
        elif ranking_mode_effective == 'support_blend':
            ranking_score[active_mask] = (
                (1.0 - lambda_support_rank) * importance_raw[active_mask]
                + lambda_support_rank * support_norm[active_mask]
            )
        else:
            raise ValueError(
                f"Unsupported SPGM ranking_mode={ranking_mode_effective!r}; supported values are 'v1' and 'support_blend'"
            )

        importance_score = torch.clamp(importance_score, 0.0, 1.0)
        ranking_score = torch.clamp(ranking_score, 0.0, 1.0)
    
    # Summary stats
    cluster_counts = {
        'near': int((cluster_id == 0).sum().item()),
        'mid': int((cluster_id == 1).sum().item()),
        'far': int((cluster_id == 2).sum().item()),
    }
    
    if active_mask.any():
        active_importance = importance_score[active_mask]
        active_ranking = ranking_score[active_mask]
        active_support_norm = support_norm[active_mask]
        importance_mean = float(active_importance.mean().item())
        importance_p50 = float(active_importance.median().item())
        ranking_score_mean = float(active_ranking.mean().item())
        ranking_score_p50 = float(active_ranking.median().item())
        support_norm_mean = float(active_support_norm.mean().item())
    else:
        importance_mean = 0.0
        importance_p50 = 0.0
        ranking_score_mean = 0.0
        ranking_score_p50 = 0.0
        support_norm_mean = 0.0
    
    return {
        'cluster_id': cluster_id,
        'depth_score': depth_score,
        'density_entropy': density_entropy,
        'density_score': density_score,
        'importance_raw': importance_raw,
        'support_norm': support_norm,
        'importance_score': importance_score,
        'ranking_score': ranking_score,
        'cluster_count_near': cluster_counts['near'],
        'cluster_count_mid': cluster_counts['mid'],
        'cluster_count_far': cluster_counts['far'],
        'importance_mean': importance_mean,
        'importance_p50': importance_p50,
        'ranking_score_mean': ranking_score_mean,
        'ranking_score_p50': ranking_score_p50,
        'support_norm_mean': support_norm_mean,
        'ranking_mode_effective': ranking_mode_effective,
        'depth_entropy': 0.0,  # Placeholder for future extension
        'density_mode_effective': density_mode,
    }