from __future__ import annotations

import torch


def _grad_norm(param) -> float:
    if param is None or getattr(param, 'grad', None) is None:
        return 0.0
    return float(torch.norm(param.grad).detach().cpu().item())


def _apply_mask_(param, weights: torch.Tensor) -> None:
    if param is None or getattr(param, 'grad', None) is None:
        return
    grad = param.grad
    mask = weights.to(device=grad.device, dtype=grad.dtype)
    while mask.ndim < grad.ndim:
        mask = mask.unsqueeze(-1)
    grad.mul_(mask)


def apply_gaussian_grad_mask(gaussians, weights: torch.Tensor, params_mode: str) -> dict:
    params_mode = (params_mode or 'xyz').strip().lower()
    positive_ratio = float((weights > 0).float().mean().item()) if weights.numel() > 0 else 0.0
    mean_weight = float(weights.mean().item()) if weights.numel() > 0 else 0.0

    xyz_pre = _grad_norm(getattr(gaussians, '_xyz', None))
    opacity_pre = _grad_norm(getattr(gaussians, '_opacity', None))

    if 'xyz' in params_mode:
        _apply_mask_(getattr(gaussians, '_xyz', None), weights)
    if 'opacity' in params_mode:
        _apply_mask_(getattr(gaussians, '_opacity', None), weights)

    xyz_post = _grad_norm(getattr(gaussians, '_xyz', None))
    opacity_post = _grad_norm(getattr(gaussians, '_opacity', None))

    return {
        'grad_keep_ratio_xyz': positive_ratio if 'xyz' in params_mode else None,
        'grad_keep_ratio_opacity': positive_ratio if 'opacity' in params_mode else None,
        'grad_weight_mean_xyz': mean_weight if 'xyz' in params_mode else None,
        'grad_weight_mean_opacity': mean_weight if 'opacity' in params_mode else None,
        'grad_norm_xyz_pre_mask': xyz_pre,
        'grad_norm_xyz_post_mask': xyz_post,
        'grad_norm_opacity_pre_mask': opacity_pre,
        'grad_norm_opacity_post_mask': opacity_post,
    }
