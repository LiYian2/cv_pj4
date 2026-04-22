"""SPGM gradient weight policy.

Build per-Gaussian gradient weights from importance score.

Supported policy modes:
- dense_keep: deterministic soft keep on the whole active set
- selector_quantile: cluster-wise selector-first keep, then deterministic soft keep
"""

from __future__ import annotations

import math

import torch


def _build_cluster_tensor(values, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    raw = list(values) if values is not None else [1.0]
    if not raw:
        raw = [1.0]
    return torch.tensor([float(v) for v in raw], dtype=dtype, device=device)


def _lookup_cluster_values(cluster_id: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    max_idx = int(table.shape[0] - 1)
    safe_cluster_id = cluster_id.clamp(0, max_idx)
    return table[safe_cluster_id]


def _cluster_counts(cluster_id: torch.Tensor, mask: torch.Tensor) -> dict:
    counts = {'near': 0, 'mid': 0, 'far': 0}
    if not mask.any():
        return counts
    counts['near'] = int(((cluster_id == 0) & mask).sum().item())
    counts['mid'] = int(((cluster_id == 1) & mask).sum().item())
    counts['far'] = int(((cluster_id == 2) & mask).sum().item())
    return counts


def _select_by_cluster_quantile(
    ranking_score: torch.Tensor,
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    selector_keep_ratio: tuple,
    selector_min_keep: int,
) -> torch.Tensor:
    selected_mask = torch.zeros_like(active_mask, dtype=torch.bool)
    if not active_mask.any():
        return selected_mask

    keep_ratio_table = [float(v) for v in (selector_keep_ratio or (1.0, 1.0, 1.0))]
    if not keep_ratio_table:
        keep_ratio_table = [1.0]

    min_keep = max(0, int(selector_min_keep))
    active_cluster_ids = torch.unique(cluster_id[active_mask])
    for raw_cluster in active_cluster_ids.tolist():
        cid = int(raw_cluster)
        cluster_mask = active_mask & (cluster_id == cid)
        cluster_indices = torch.where(cluster_mask)[0]
        count = int(cluster_indices.numel())
        if count <= 0:
            continue

        keep_ratio = keep_ratio_table[min(max(cid, 0), len(keep_ratio_table) - 1)]
        keep_ratio = min(max(float(keep_ratio), 0.0), 1.0)
        keep_count = int(math.ceil(count * keep_ratio))
        keep_count = max(min_keep, keep_count)
        keep_count = min(count, keep_count)
        if keep_count <= 0:
            continue
        if keep_count >= count:
            selected_mask[cluster_indices] = True
            continue

        cluster_scores = ranking_score[cluster_indices]
        topk = torch.topk(cluster_scores, k=keep_count, largest=True, sorted=False)
        selected_indices = cluster_indices[topk.indices]
        selected_mask[selected_indices] = True

    return selected_mask


def build_spgm_grad_weights(
    weight_score: torch.Tensor,
    cluster_id: torch.Tensor,
    active_mask: torch.Tensor,
    weight_floor: float = 0.05,
    cluster_keep: tuple = (1.0, 0.8, 0.6),
    policy_mode: str = 'dense_keep',
    selector_keep_ratio: tuple = (1.0, 1.0, 1.0),
    selector_min_keep: int = 1,
    ranking_score: torch.Tensor | None = None,
) -> dict:
    """Build per-Gaussian gradient weights from decoupled weighting / ranking scores."""
    N = weight_score.shape[0]
    device = weight_score.device
    dtype = weight_score.dtype

    weights = torch.zeros(N, dtype=dtype, device=device)
    if not active_mask.any():
        return {
            'weights': weights,
            'weight_mean': 0.0,
            'weight_p10': 0.0,
            'weight_p50': 0.0,
            'weight_p90': 0.0,
            'active_ratio': 0.0,
            'selected_ratio': 0.0,
            'selected_count_near': 0,
            'selected_count_mid': 0,
            'selected_count_far': 0,
            'policy_mode_effective': str(policy_mode or 'dense_keep').strip().lower(),
        }

    cluster_weights = _build_cluster_tensor(cluster_keep, dtype=dtype, device=device)
    base_keep = weight_floor + (1.0 - weight_floor) * weight_score
    if ranking_score is None:
        ranking_score = weight_score

    policy_mode = str(policy_mode or 'dense_keep').strip().lower()
    if policy_mode == 'dense_keep':
        selected_mask = active_mask.clone()
    elif policy_mode == 'selector_quantile':
        selected_mask = _select_by_cluster_quantile(
            ranking_score=ranking_score,
            cluster_id=cluster_id,
            active_mask=active_mask,
            selector_keep_ratio=selector_keep_ratio,
            selector_min_keep=selector_min_keep,
        )
    else:
        raise ValueError(f'Unsupported SPGM policy_mode={policy_mode!r}')

    if selected_mask.any():
        selected_cluster_id = cluster_id[selected_mask]
        cluster_factor = _lookup_cluster_values(selected_cluster_id, cluster_weights)
        weights[selected_mask] = base_keep[selected_mask] * cluster_factor

    active_ratio = float(active_mask.float().mean().item())
    selected_ratio = float(selected_mask.float().mean().item())

    active_weights = weights[active_mask]
    weight_mean = float(active_weights.mean().item())
    sorted_weights = torch.sort(active_weights).values
    n_active = int(active_weights.shape[0])
    p10_idx = max(0, int(n_active * 0.1))
    p50_idx = max(0, int(n_active * 0.5))
    p90_idx = max(0, min(n_active - 1, int(n_active * 0.9)))
    weight_p10 = float(sorted_weights[p10_idx].item())
    weight_p50 = float(sorted_weights[p50_idx].item())
    weight_p90 = float(sorted_weights[p90_idx].item())

    selected_counts = _cluster_counts(cluster_id, selected_mask)

    return {
        'weights': weights,
        'weight_mean': weight_mean,
        'weight_p10': weight_p10,
        'weight_p50': weight_p50,
        'weight_p90': weight_p90,
        'active_ratio': active_ratio,
        'selected_ratio': selected_ratio,
        'selected_count_near': selected_counts['near'],
        'selected_count_mid': selected_counts['mid'],
        'selected_count_far': selected_counts['far'],
        'policy_mode_effective': policy_mode,
    }
