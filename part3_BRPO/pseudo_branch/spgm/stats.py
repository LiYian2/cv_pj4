"""SPGM statistics extractor - Phase 1 implementation.

Extract per-Gaussian statistics from accepted pseudo views:
- support_count: per-Gaussian pseudo view support count (weighted)
- depth_value: camera-space depth (weighted median, vectorized)
- density_proxy: opacity * support normalized
- active_mask: support_count > 0
"""

from __future__ import annotations

import torch


def _camera_space_depths(xyz: torch.Tensor, vp) -> torch.Tensor:
    """Compute camera-space z-depth for each Gaussian.
    
    Args:
        xyz: Tensor[N, 3] of Gaussian centers in world space
        vp: Viewpoint with R, T, cam_rot_delta, cam_trans_delta attributes
        
    Returns:
        Tensor[N] of z-values in camera space (positive = in front of camera)
    """
    from pseudo_branch.pseudo_camera_state import current_w2c
    
    w2c = current_w2c(vp)  # (4, 4)
    R = w2c[:3, :3]  # (3, 3)
    T = w2c[:3, 3]   # (3,)
    
    xyz_cam = xyz @ R.T + T.unsqueeze(0)  # (N, 3)
    z_depth = xyz_cam[:, 2]  # (N,)
    
    return z_depth


def _normalize_active(x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range, only for active Gaussians."""
    if not active_mask.any():
        return torch.zeros_like(x)
    
    active_x = x[active_mask]
    x_min = float(active_x.min().item())
    x_max = float(active_x.max().item())
    
    if x_max <= x_min + 1e-8:
        normalized = torch.zeros_like(x)
        normalized[active_mask] = 0.5
        return normalized
    
    normalized = torch.zeros_like(x)
    normalized[active_mask] = (active_x - x_min) / (x_max - x_min)
    return normalized


def _weighted_median_vectorized(
    depth_stack: torch.Tensor, 
    weight_stack: torch.Tensor, 
    valid_stack: torch.Tensor
) -> torch.Tensor:
    """Compute weighted median depth across views for each Gaussian (vectorized).
    
    Strategy: For each column (Gaussian), sort by depth, accumulate weights,
    find where cumulative weight crosses 0.5 * total_weight.
    
    Args:
        depth_stack: Tensor[V, N] of depths per view per Gaussian
        weight_stack: Tensor[V, N] of gate weights per view per Gaussian  
        valid_stack: BoolTensor[V, N] of visibility validity
        
    Returns:
        Tensor[N] of weighted median depths (inactive = 0)
    """
    V, N = depth_stack.shape
    device = depth_stack.device
    dtype = depth_stack.dtype
    
    # Mask invalid entries: set depth to inf, weight to 0
    invalid_mask = ~valid_stack
    depth_masked = depth_stack.clone()
    depth_masked[invalid_mask] = float('inf')
    weight_masked = weight_stack.clone()
    weight_masked[invalid_mask] = 0.0
    
    # Sort depths for each Gaussian (column-wise)
    # sorted_indices: Tensor[V, N] - indices that would sort each column
    sorted_indices = torch.argsort(depth_masked, dim=0)  # (V, N)
    
    # Gather sorted depths and weights
    # Use advanced indexing: for each column, reorder by sorted_indices
    sorted_depths = torch.zeros_like(depth_masked)
    sorted_weights = torch.zeros_like(weight_masked)
    
    for v in range(V):
        idx = sorted_indices[v, :]  # (N,) indices for this view's position in sorted order
        sorted_depths[v, :] = depth_masked[idx, torch.arange(N, device=device)]
        sorted_weights[v, :] = weight_masked[idx, torch.arange(N, device=device)]
    
    # Cumulative weights per column
    cum_weights = torch.cumsum(sorted_weights, dim=0)  # (V, N)
    
    # Total weight per Gaussian
    total_weights = cum_weights[-1, :]  # (N,)
    
    # Find first position where cum_weight >= 0.5 * total_weight
    half_weights = total_weights * 0.5  # (N,)
    
    # For inactive Gaussians (total_weight=0), result stays 0
    # For active ones, find median index
    result = torch.zeros(N, dtype=dtype, device=device)
    
    active_gaussians = total_weights > 1e-8
    if not active_gaussians.any():
        return result
    
    # For each active Gaussian, find where cumulative weight crosses half
    # Create threshold matrix: cum_weights >= half_weights (broadcast)
    threshold_crossed = cum_weights >= half_weights.unsqueeze(0)  # (V, N)
    
    # Find first True position per column
    # argmax returns first index where condition is True (since False=0, True=1)
    median_indices = torch.argmax(threshold_crossed.long(), dim=0)  # (N,)
    
    # Extract median depths
    result[active_gaussians] = sorted_depths[median_indices[active_gaussians], active_gaussians]
    
    return result


def collect_spgm_stats(
    sampled_views: list,
    gate_results: list,
    render_packages: list,
    gaussians,
    device: torch.device,
) -> dict:
    """Extract per-Gaussian statistics from accepted pseudo views."""
    N = int(gaussians._xyz.shape[0])
    xyz = gaussians._xyz.detach()  # (N, 3)
    
    support_count = torch.zeros(N, dtype=torch.float32, device=device)
    
    accepted_views = []
    
    for view, gate, pkg in zip(sampled_views, gate_results, render_packages):
        if pkg is None:
            continue
        view_weight = float(gate.get('weight', 0.0))
        if view_weight <= 0.0:
            continue
        
        accepted_views.append((view, pkg, view_weight))
        
        vis = pkg.get('visibility_filter')
        if vis is None:
            continue
        vis = vis.detach()
        if vis.dtype != torch.bool:
            vis = vis > 0
        if vis.device != device:
            vis = vis.to(device)
        if vis.numel() != N:
            raise ValueError(f'Visibility size mismatch: got {vis.numel()} expected {N}')
        
        support_count[vis] += view_weight
    
    accepted_view_count = len(accepted_views)
    
    depth_value = torch.zeros(N, dtype=torch.float32, device=device)
    
    if accepted_view_count > 0:
        V = accepted_view_count
        depth_stack = torch.zeros(V, N, dtype=torch.float32, device=device)
        weight_stack = torch.zeros(V, N, dtype=torch.float32, device=device)
        valid_stack = torch.zeros(V, N, dtype=torch.bool, device=device)
        
        for v_idx, (view, pkg, view_weight) in enumerate(accepted_views):
            vp = view['vp']
            vis = pkg.get('visibility_filter')
            if vis is None:
                continue
            vis = vis.detach()
            if vis.dtype != torch.bool:
                vis = vis > 0
            if vis.device != device:
                vis = vis.to(device)
            
            try:
                z_depths = _camera_space_depths(xyz, vp)
                depth_stack[v_idx, :] = z_depths.to(device)
                weight_stack[v_idx, :] = view_weight
                valid_stack[v_idx, :] = vis
            except Exception:
                continue
        
        depth_value = _weighted_median_vectorized(depth_stack, weight_stack, valid_stack)
    
    # Density proxy
    opacity = gaussians.get_opacity.detach().view(-1)
    if opacity.device != device:
        opacity = opacity.to(device)
    
    active_mask = support_count > 0
    opacity_normalized = _normalize_active(opacity, active_mask)
    support_normalized = _normalize_active(support_count, active_mask)
    density_proxy = opacity_normalized * support_normalized
    
    # Summary stats
    active_ratio = float(active_mask.float().mean().item()) if N > 0 else 0.0
    
    if active_mask.any():
        active_support = support_count[active_mask]
        support_mean = float(active_support.mean().item())
        support_p50 = float(active_support.median().item())
        support_max = float(active_support.max().item())
    else:
        support_mean = 0.0
        support_p50 = 0.0
        support_max = 0.0
    
    return {
        'support_count': support_count,
        'depth_value': depth_value,
        'density_proxy': density_proxy,
        'active_mask': active_mask,
        'accepted_view_count': accepted_view_count,
        'active_ratio': active_ratio,
        'support_mean': support_mean,
        'support_p50': support_p50,
        'support_max': support_max,
    }