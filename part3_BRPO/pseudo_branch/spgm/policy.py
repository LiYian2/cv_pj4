"""SPGM gradient weight policy - Phase 3 implementation.

Build per-Gaussian gradient weights from importance score:
- Deterministic soft keep: w_keep = weight_floor + (1 - weight_floor) * importance
- Cluster-aware weighting: w = w_keep * cluster_keep[cluster_id]
- Inactive -> 0
"""

from __future__ import annotations

import torch


def build_spgm_grad_weights(
    importance_score: torch.Tensor,
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    weight_floor: float = 0.05,
    cluster_keep: tuple = (1.0, 0.8, 0.6),
) -> dict:
    """Build per-Gaussian gradient weights from importance score.
    
    Args:
        importance_score: Tensor[N] of importance scores in [0, 1]
        cluster_id: Tensor[N] of cluster IDs (-1 for inactive, 0/1/2 for near/mid/far)
        active_mask: BoolTensor[N] of active Gaussians
        weight_floor: Minimum weight floor (default 0.05)
        cluster_keep: Tuple of (near, mid, far) cluster keep weights
        
    Returns:
        dict with tensors and summary stats:
            - weights: Tensor[N] in [0, 1]
            - weight_mean: float
            - weight_p10: float
            - weight_p50: float
            - weight_p90: float
            - active_ratio: float
    """
    N = importance_score.shape[0]
    device = importance_score.device
    dtype = importance_score.dtype
    
    # Initialize weights to 0
    weights = torch.zeros(N, dtype=dtype, device=device)
    
    if not active_mask.any():
        return {
            'weights': weights,
            'weight_mean': 0.0,
            'weight_p10': 0.0,
            'weight_p50': 0.0,
            'weight_p90': 0.0,
            'active_ratio': 0.0,
        }
    
    # Build cluster keep tensor
    # cluster_keep[0] = near (1.0), cluster_keep[1] = mid (0.8), cluster_keep[2] = far (0.6)
    cluster_weights = torch.tensor(cluster_keep, dtype=dtype, device=device)
    
    # For active Gaussians:
    # 1. Base keep weight from importance
    base_keep = weight_floor + (1.0 - weight_floor) * importance_score
    
    # 2. Apply cluster modifier
    # Map cluster_id to cluster_weights: valid clusters are 0, 1, 2
    # cluster_id = -1 means inactive (already filtered by active_mask)
    
    active_cluster_id = cluster_id[active_mask].clamp(0, 2)  # Clamp to valid range [0, 2]
    cluster_factor = cluster_weights[active_cluster_id]
    
    # Final weight
    weights[active_mask] = base_keep[active_mask] * cluster_factor
    
    # Compute summary stats
    active_ratio = float(active_mask.float().mean().item())
    
    if active_mask.any():
        active_weights = weights[active_mask]
        weight_mean = float(active_weights.mean().item())
        
        # Percentiles: p10, p50, p90
        sorted_weights = torch.sort(active_weights).values
        n_active = int(active_weights.shape[0])
        
        p10_idx = max(0, int(n_active * 0.1))
        p50_idx = max(0, int(n_active * 0.5))
        p90_idx = max(0, min(n_active - 1, int(n_active * 0.9)))
        
        weight_p10 = float(sorted_weights[p10_idx].item())
        weight_p50 = float(sorted_weights[p50_idx].item())
        weight_p90 = float(sorted_weights[p90_idx].item())
    else:
        weight_mean = 0.0
        weight_p10 = 0.0
        weight_p50 = 0.0
        weight_p90 = 0.0
    
    return {
        'weights': weights,
        'weight_mean': weight_mean,
        'weight_p10': weight_p10,
        'weight_p50': weight_p50,
        'weight_p90': weight_p90,
        'active_ratio': active_ratio,
    }