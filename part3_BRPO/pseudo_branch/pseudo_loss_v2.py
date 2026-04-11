from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def to_torch(x, device: str = 'cuda', dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def apply_exposure(image: torch.Tensor, viewpoint) -> torch.Tensor:
    a = torch.exp(viewpoint.exposure_a).view(1, 1, 1)
    b = viewpoint.exposure_b.view(1, 1, 1)
    return a * image + b


def masked_rgb_loss(render_rgb, target_rgb, confidence_mask, viewpoint) -> torch.Tensor:
    render_rgb = apply_exposure(render_rgb, viewpoint)
    target_rgb = to_torch(target_rgb, device=render_rgb.device)
    confidence_mask = to_torch(confidence_mask, device=render_rgb.device)
    if target_rgb.dim() == 3 and target_rgb.shape[-1] == 3:
        target_rgb = target_rgb.permute(2, 0, 1)
    if confidence_mask.dim() == 2:
        confidence_mask = confidence_mask.unsqueeze(0)
    l1 = torch.abs(render_rgb - target_rgb)
    denom = confidence_mask.sum() * render_rgb.shape[0] + 1e-8
    return (l1 * confidence_mask).sum() / denom


def masked_depth_loss(render_depth, target_depth, confidence_mask) -> torch.Tensor:
    render_depth = to_torch(render_depth, device='cuda')
    target_depth = to_torch(target_depth, device=render_depth.device)
    confidence_mask = to_torch(confidence_mask, device=render_depth.device)

    if render_depth.dim() == 2:
        render_depth = render_depth.unsqueeze(0)
    if target_depth.dim() == 2:
        target_depth = target_depth.unsqueeze(0)
    if confidence_mask.dim() == 2:
        confidence_mask = confidence_mask.unsqueeze(0)

    valid = (target_depth > 1e-4).float() * (confidence_mask > 0).float()
    weighted = valid * confidence_mask
    denom = weighted.sum() + 1e-8
    return (torch.abs(render_depth - target_depth) * weighted).sum() / denom


def pose_reg_loss(viewpoint, trans_weight: float = 1.0) -> torch.Tensor:
    return torch.norm(viewpoint.cam_rot_delta, p=2) + float(trans_weight) * torch.norm(viewpoint.cam_trans_delta, p=2)


def exposure_reg_loss(viewpoint) -> torch.Tensor:
    return viewpoint.exposure_a.abs().mean() + viewpoint.exposure_b.abs().mean()


def build_stageA_loss(render_rgb, render_depth, target_rgb, target_depth, confidence_mask, viewpoint, beta_rgb: float, lambda_pose: float, lambda_exp: float, trans_weight: float, use_depth: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_rgb = masked_rgb_loss(render_rgb, target_rgb, confidence_mask, viewpoint)
    l_depth = masked_depth_loss(render_depth, target_depth, confidence_mask) if use_depth else torch.zeros((), device=render_rgb.device)
    l_pose = pose_reg_loss(viewpoint, trans_weight)
    l_exp = exposure_reg_loss(viewpoint)
    total = float(beta_rgb) * l_rgb + (1.0 - float(beta_rgb)) * l_depth + float(lambda_pose) * l_pose + float(lambda_exp) * l_exp
    stats = {
        'loss_rgb': float(l_rgb.detach().item()),
        'loss_depth': float(l_depth.detach().item()),
        'loss_pose_reg': float(l_pose.detach().item()),
        'loss_exp_reg': float(l_exp.detach().item()),
        'loss_total': float(total.detach().item()),
    }
    return total, stats
