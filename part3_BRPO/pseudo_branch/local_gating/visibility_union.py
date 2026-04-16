from __future__ import annotations

from typing import Iterable, List

import torch


def build_visibility_weight_map(render_packages: Iterable[dict], gate_results: List[dict], num_gaussians: int, device=None) -> dict:
    if device is None:
        for pkg in render_packages:
            vis = pkg.get('visibility_filter') if pkg is not None else None
            if vis is not None and hasattr(vis, 'device'):
                device = vis.device
                break
        if device is None:
            device = torch.device('cpu')

    weights = torch.zeros(int(num_gaussians), dtype=torch.float32, device=device)
    accepted_count = 0
    for pkg, gate in zip(render_packages, gate_results):
        if pkg is None:
            continue
        view_weight = float(gate.get('weight', 0.0))
        if view_weight <= 0.0:
            continue
        vis = pkg.get('visibility_filter')
        if vis is None:
            continue
        vis = vis.detach()
        if vis.dtype != torch.bool:
            vis = vis > 0
        if vis.device != weights.device:
            vis = vis.to(weights.device)
        if vis.numel() != weights.numel():
            raise ValueError(f'Visibility size mismatch: got {vis.numel()} expected {weights.numel()}')
        accepted_count += 1
        fill_value = torch.tensor(view_weight, dtype=weights.dtype, device=weights.device)
        weights[vis] = torch.maximum(weights[vis], fill_value)

    positive = weights > 0
    return {
        'weights': weights,
        'accepted_count': int(accepted_count),
        'visible_union_ratio': float(positive.float().mean().item()) if weights.numel() > 0 else 0.0,
        'visible_union_weight_mean': float(weights.mean().item()) if weights.numel() > 0 else 0.0,
    }
