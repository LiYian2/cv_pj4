"""SPGM statistics extractor.

Phase B2 extends SPGM statistics from accepted-pseudo-only summaries to a more
scene-aware current-window summary while preserving the old pseudo-active set for
selector / weighting compatibility.

G-BRPO-0 adds control_universe parameter to switch between active_mask and
population_active_mask as the primary control domain.
"""

from __future__ import annotations

from typing import Iterable

import torch


def _camera_space_depths(xyz: torch.Tensor, vp) -> torch.Tensor:
    """Compute camera-space z-depth for each Gaussian."""
    from pseudo_branch.refine.pseudo_camera_state import current_w2c

    w2c = current_w2c(vp)
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    xyz_cam = xyz @ R.T + T.unsqueeze(0)
    return xyz_cam[:, 2]


def _normalize_active(x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range, only for active Gaussians."""
    if not active_mask.any():
        return torch.zeros_like(x)

    active_x = x[active_mask]
    x_min = float(active_x.min().item())
    x_max = float(active_x.max().item())

    normalized = torch.zeros_like(x)
    if x_max <= x_min + 1e-8:
        normalized[active_mask] = 0.5
        return normalized

    normalized[active_mask] = (active_x - x_min) / (x_max - x_min)
    return normalized


def _normalize_control(x: torch.Tensor, control_mask: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range, only for control universe Gaussians."""
    if not control_mask.any():
        return torch.zeros_like(x)

    control_x = x[control_mask]
    x_min = float(control_x.min().item())
    x_max = float(control_x.max().item())

    normalized = torch.zeros_like(x)
    if x_max <= x_min + 1e-8:
        normalized[control_mask] = 0.5
        return normalized

    normalized[control_mask] = (control_x - x_min) / (x_max - x_min)
    return normalized


def _to_visibility_mask(vis, N: int, device: torch.device) -> torch.Tensor | None:
    if vis is None:
        return None
    vis = vis.detach()
    if vis.dtype != torch.bool:
        vis = vis > 0
    if vis.device != device:
        vis = vis.to(device)
    if vis.numel() != N:
        raise ValueError(f'Visibility size mismatch: got {vis.numel()} expected {N}')
    return vis


def _accumulate_support(
    support_tensor: torch.Tensor,
    render_packages: Iterable,
    weights: Iterable[float],
    N: int,
    device: torch.device,
) -> int:
    used = 0
    for pkg, weight in zip(render_packages, weights):
        if pkg is None:
            continue
        cur_weight = float(weight)
        if cur_weight <= 0.0:
            continue
        vis = _to_visibility_mask(pkg.get('visibility_filter'), N=N, device=device)
        if vis is None:
            continue
        support_tensor[vis] += cur_weight
        used += 1
    return used


def _weighted_median_vectorized(
    depth_stack: torch.Tensor,
    weight_stack: torch.Tensor,
    valid_stack: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted median depth across views for each Gaussian (vectorized)."""
    V, N = depth_stack.shape
    device = depth_stack.device
    dtype = depth_stack.dtype

    invalid_mask = ~valid_stack
    depth_masked = depth_stack.clone()
    depth_masked[invalid_mask] = float('inf')
    weight_masked = weight_stack.clone()
    weight_masked[invalid_mask] = 0.0

    sorted_indices = torch.argsort(depth_masked, dim=0)
    sorted_depths = torch.zeros_like(depth_masked)
    sorted_weights = torch.zeros_like(weight_masked)

    for v in range(V):
        idx = sorted_indices[v, :]
        sorted_depths[v, :] = depth_masked[idx, torch.arange(N, device=device)]
        sorted_weights[v, :] = weight_masked[idx, torch.arange(N, device=device)]

    cum_weights = torch.cumsum(sorted_weights, dim=0)
    total_weights = cum_weights[-1, :]
    half_weights = total_weights * 0.5

    result = torch.zeros(N, dtype=dtype, device=device)
    active_gaussians = total_weights > 1e-8
    if not active_gaussians.any():
        return result

    threshold_crossed = cum_weights >= half_weights.unsqueeze(0)
    median_indices = torch.argmax(threshold_crossed.long(), dim=0)
    result[active_gaussians] = sorted_depths[median_indices[active_gaussians], active_gaussians]
    return result


def _get_gaussian_volume(gaussians, device: torch.device) -> torch.Tensor:
    scaling = getattr(gaussians, 'get_scaling', None)
    if scaling is None:
        return torch.ones(int(gaussians._xyz.shape[0]), dtype=torch.float32, device=device)
    scaling = scaling.detach()
    if scaling.device != device:
        scaling = scaling.to(device)
    if scaling.ndim == 1:
        scaling = scaling.view(-1, 1)
    if scaling.shape[-1] == 1:
        scaling = scaling.repeat(1, 3)
    scaling = scaling.clamp_min(1e-8)
    return scaling.prod(dim=1)


def collect_spgm_stats(
    sampled_views: list,
    gate_results: list,
    render_packages: list,
    gaussians,
    device: torch.device,
    extra_window_views: list | None = None,
    extra_window_render_packages: list | None = None,
    extra_window_weights: list | None = None,
    control_universe: str = 'active',
) -> dict:
    """Extract SPGM stats from accepted pseudo support + current train window summary.

    Args:
        control_universe: 'active' uses accepted-pseudo active_mask; 'population_active'
            uses scene-level population_active_mask. For direct BRPO path, should be
            'population_active' to align with BRPO semantics.
    """
    N = int(gaussians._xyz.shape[0])
    xyz = gaussians._xyz.detach()

    support_count = torch.zeros(N, dtype=torch.float32, device=device)
    population_support_count = torch.zeros(N, dtype=torch.float32, device=device)
    accepted_views = []

    for view, gate, pkg in zip(sampled_views, gate_results, render_packages):
        if pkg is None:
            continue
        view_weight = float(gate.get('weight', 0.0))
        if view_weight <= 0.0:
            continue

        vis = _to_visibility_mask(pkg.get('visibility_filter'), N=N, device=device)
        if vis is None:
            continue

        accepted_views.append((view, pkg, view_weight))
        support_count[vis] += view_weight
        population_support_count[vis] += view_weight

    accepted_view_count = len(accepted_views)

    extra_window_views = list(extra_window_views or [])
    extra_window_render_packages = list(extra_window_render_packages or [])
    if extra_window_weights is None:
        extra_window_weights = [1.0] * len(extra_window_render_packages)
    else:
        extra_window_weights = list(extra_window_weights)
    if extra_window_render_packages:
        _accumulate_support(
            support_tensor=population_support_count,
            render_packages=extra_window_render_packages,
            weights=extra_window_weights,
            N=N,
            device=device,
        )
    window_view_count = accepted_view_count + len(extra_window_render_packages)

    depth_value = torch.zeros(N, dtype=torch.float32, device=device)
    if accepted_view_count > 0:
        V = accepted_view_count
        depth_stack = torch.zeros(V, N, dtype=torch.float32, device=device)
        weight_stack = torch.zeros(V, N, dtype=torch.float32, device=device)
        valid_stack = torch.zeros(V, N, dtype=torch.bool, device=device)

        for v_idx, (view, pkg, view_weight) in enumerate(accepted_views):
            vis = _to_visibility_mask(pkg.get('visibility_filter'), N=N, device=device)
            if vis is None:
                continue
            try:
                z_depths = _camera_space_depths(xyz, view['vp'])
                depth_stack[v_idx, :] = z_depths.to(device)
                weight_stack[v_idx, :] = float(view_weight)
                valid_stack[v_idx, :] = vis
            except Exception:
                continue

        depth_value = _weighted_median_vectorized(depth_stack, weight_stack, valid_stack)

    opacity = gaussians.get_opacity.detach().view(-1)
    if opacity.device != device:
        opacity = opacity.to(device)

    active_mask = support_count > 0
    population_active_mask = population_support_count > 0

    # Determine control_mask based on control_universe setting
    control_universe_effective = str(control_universe or 'active').strip().lower()
    if control_universe_effective == 'population_active':
        control_mask = population_active_mask
        control_mask_name = 'population_active_mask'
    else:
        # Default: use active_mask (accepted-pseudo only)
        control_mask = active_mask
        control_mask_name = 'active_mask'

    # Normalization uses control_mask for direct BRPO path
    opacity_normalized = _normalize_control(opacity, control_mask)
    support_normalized = _normalize_control(support_count, control_mask)
    population_support_norm = _normalize_control(population_support_count, control_mask)
    density_proxy = opacity_normalized * support_normalized

    gaussian_volume = _get_gaussian_volume(gaussians, device=device)
    struct_density_proxy = (opacity.clamp_min(1e-8) / gaussian_volume.clamp_min(1e-8)) * (1.0 + population_support_norm)

    active_ratio = float(active_mask.float().mean().item()) if N > 0 else 0.0
    population_active_ratio = float(population_active_mask.float().mean().item()) if N > 0 else 0.0
    control_ratio = float(control_mask.float().mean().item()) if N > 0 else 0.0

    if control_mask.any():
        control_support = support_count[control_mask]
        control_depth = depth_value[control_mask]
        support_mean = float(control_support.mean().item())
        support_p50 = float(control_support.median().item())
        support_max = float(control_support.max().item())
        population_support_mean = float(population_support_count[control_mask].mean().item())
        struct_density_mean = float(struct_density_proxy[control_mask].mean().item())
        depth_mean = float(control_depth.mean().item())
        depth_p50 = float(control_depth.median().item())
    else:
        support_mean = 0.0
        support_p50 = 0.0
        support_max = 0.0
        population_support_mean = 0.0
        struct_density_mean = 0.0
        depth_mean = 0.0
        depth_p50 = 0.0

    return {
        'support_count': support_count,
        'population_support_count': population_support_count,
        'depth_value': depth_value,
        'density_proxy': density_proxy,
        'struct_density_proxy': struct_density_proxy,
        'gaussian_volume': gaussian_volume,
        'active_mask': active_mask,
        'population_active_mask': population_active_mask,
        'control_mask': control_mask,
        'control_mask_name': control_mask_name,
        'control_universe_effective': control_universe_effective,
        'accepted_view_count': accepted_view_count,
        'window_view_count': int(window_view_count),
        'active_ratio': active_ratio,
        'population_active_ratio': population_active_ratio,
        'control_ratio': control_ratio,
        'support_mean': support_mean,
        'support_p50': support_p50,
        'support_max': support_max,
        'population_support_mean': population_support_mean,
        'struct_density_mean': struct_density_mean,
        'depth_mean': depth_mean,
        'depth_p50': depth_p50,
    }