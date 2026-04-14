from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from pseudo_branch.pseudo_camera_state import current_w2c
from utils.pose_utils import SE3_log

SEED_SOURCE_IDS = (1, 2, 3)
DENSE_SOURCE_ID = 4
FALLBACK_SOURCE_ID = 0


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


def masked_depth_loss_by_source(render_depth, target_depth, confidence_mask, source_map, source_ids) -> torch.Tensor:
    render_depth = to_torch(render_depth, device='cuda')
    target_depth = to_torch(target_depth, device=render_depth.device)
    confidence_mask = to_torch(confidence_mask, device=render_depth.device)
    source_map = to_torch(source_map, device=render_depth.device, dtype=torch.int64)

    if render_depth.dim() == 2:
        render_depth = render_depth.unsqueeze(0)
    if target_depth.dim() == 2:
        target_depth = target_depth.unsqueeze(0)
    if confidence_mask.dim() == 2:
        confidence_mask = confidence_mask.unsqueeze(0)
    if source_map.dim() == 2:
        source_map = source_map.unsqueeze(0)

    source_mask = torch.zeros_like(confidence_mask)
    for sid in source_ids:
        source_mask = source_mask + (source_map == int(sid)).float()
    source_mask = (source_mask > 0).float()

    valid = (target_depth > 1e-4).float() * (confidence_mask > 0).float() * source_mask
    weighted = valid * confidence_mask
    denom = weighted.sum() + 1e-8
    return (torch.abs(render_depth - target_depth) * weighted).sum() / denom


def pose_reg_loss(viewpoint, trans_weight: float = 1.0) -> torch.Tensor:
    return torch.norm(viewpoint.cam_rot_delta, p=2) + float(trans_weight) * torch.norm(viewpoint.cam_trans_delta, p=2)


def exposure_reg_loss(viewpoint) -> torch.Tensor:
    return viewpoint.exposure_a.abs().mean() + viewpoint.exposure_b.abs().mean()


def _build_w2c0(viewpoint):
    w2c0 = torch.eye(4, device=viewpoint.R.device, dtype=viewpoint.R.dtype)
    w2c0[:3, :3] = viewpoint.R0
    w2c0[:3, 3] = viewpoint.T0
    return w2c0


def absolute_pose_prior_loss(viewpoint) -> torch.Tensor:
    if not hasattr(viewpoint, 'R0') or not hasattr(viewpoint, 'T0'):
        return torch.zeros((), device=viewpoint.R.device, dtype=viewpoint.R.dtype)
    delta = current_w2c(viewpoint) @ torch.linalg.inv(_build_w2c0(viewpoint))
    tau = SE3_log(delta)
    return torch.sum(tau * tau)


def compute_abs_pose_components(viewpoint, scene_scale) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=viewpoint.R.device, dtype=viewpoint.R.dtype)
    if not hasattr(viewpoint, 'R0') or not hasattr(viewpoint, 'T0'):
        return {
            'rho_vec': torch.zeros(3, device=viewpoint.R.device, dtype=viewpoint.R.dtype),
            'theta_vec': torch.zeros(3, device=viewpoint.R.device, dtype=viewpoint.R.dtype),
            'rho_norm': zero,
            'theta_norm': zero,
            'rho_scaled_norm': zero,
            'scene_scale_used': torch.ones((), device=viewpoint.R.device, dtype=viewpoint.R.dtype),
        }

    delta = current_w2c(viewpoint) @ torch.linalg.inv(_build_w2c0(viewpoint))
    tau = SE3_log(delta)
    rho_vec = tau[:3]
    theta_vec = tau[3:]

    if isinstance(scene_scale, torch.Tensor):
        scene_scale_t = scene_scale.to(device=viewpoint.R.device, dtype=viewpoint.R.dtype)
    else:
        scene_scale_t = torch.tensor(float(scene_scale), device=viewpoint.R.device, dtype=viewpoint.R.dtype)
    scene_scale_t = torch.clamp(scene_scale_t, min=1e-6)

    rho_norm = torch.norm(rho_vec, p=2)
    theta_norm = torch.norm(theta_vec, p=2)
    rho_scaled_norm = rho_norm / scene_scale_t

    return {
        'rho_vec': rho_vec,
        'theta_vec': theta_vec,
        'rho_norm': rho_norm,
        'theta_norm': theta_norm,
        'rho_scaled_norm': rho_scaled_norm,
        'scene_scale_used': scene_scale_t,
    }


def _robust_penalty(x: torch.Tensor, robust_type: str = 'charbonnier', delta: float = 1e-3) -> torch.Tensor:
    robust = robust_type.lower()
    if robust == 'l2':
        return x * x
    if robust == 'huber':
        d = float(delta)
        absx = torch.abs(x)
        return torch.where(absx <= d, 0.5 * absx * absx / d, absx - 0.5 * d)
    # default: charbonnier
    return torch.sqrt(x * x + 1e-6)


def absolute_pose_prior_loss_scaled(
    viewpoint,
    scene_scale,
    lambda_abs_t: float,
    lambda_abs_r: float,
    robust_type: str = 'charbonnier',
):
    comps = compute_abs_pose_components(viewpoint, scene_scale)
    l_abs_t = float(lambda_abs_t) * _robust_penalty(comps['rho_scaled_norm'], robust_type=robust_type)
    l_abs_r = float(lambda_abs_r) * _robust_penalty(comps['theta_norm'], robust_type=robust_type)
    return l_abs_t + l_abs_r, l_abs_t, l_abs_r, comps


def _build_stats_dict(l_rgb, l_depth, l_depth_seed, l_depth_dense, l_depth_fallback, l_pose, l_abs_pose, l_abs_t, l_abs_r, l_exp, total, comps, extra_stats=None):
    out = {
        'loss_rgb': float(l_rgb.detach().item()),
        'loss_depth': float(l_depth.detach().item()),
        'loss_depth_seed': float(l_depth_seed.detach().item()),
        'loss_depth_dense': float(l_depth_dense.detach().item()),
        'loss_depth_fallback': float(l_depth_fallback.detach().item()),
        'loss_pose_reg': float(l_pose.detach().item()),
        'loss_abs_pose_reg': float(l_abs_pose.detach().item()),
        'loss_abs_pose_trans': float(l_abs_t.detach().item()),
        'loss_abs_pose_rot': float(l_abs_r.detach().item()),
        'abs_pose_rho_norm': float(comps['rho_norm'].detach().item()),
        'abs_pose_theta_norm': float(comps['theta_norm'].detach().item()),
        'scene_scale_used': float(comps['scene_scale_used'].detach().item()),
        'loss_exp_reg': float(l_exp.detach().item()),
        'loss_total': float(total.detach().item()),
    }
    if extra_stats:
        out.update(extra_stats)
    return out


def _mask_extra_stats(rgb_confidence_mask, depth_confidence_mask, depth_source_map=None, device='cuda'):
    rgb_mask = to_torch(rgb_confidence_mask, device=device)
    depth_mask = to_torch(depth_confidence_mask, device=device)
    extras = {
        'rgb_conf_nonzero_ratio': float((rgb_mask > 0).float().mean().item()),
        'rgb_conf_mean': float(rgb_mask.mean().item()),
        'depth_conf_nonzero_ratio': float((depth_mask > 0).float().mean().item()),
        'depth_conf_mean': float(depth_mask.mean().item()),
    }
    if depth_source_map is not None:
        source_map = to_torch(depth_source_map, device=device, dtype=torch.int64)
        verified = (source_map != FALLBACK_SOURCE_ID).float()
        seed = torch.isin(source_map, torch.tensor(SEED_SOURCE_IDS, device=source_map.device, dtype=source_map.dtype)).float()
        dense = (source_map == int(DENSE_SOURCE_ID)).float()
        extras.update({
            'depth_verified_ratio': float(verified.mean().item()),
            'depth_seed_ratio': float(seed.mean().item()),
            'depth_dense_ratio': float(dense.mean().item()),
            'depth_effective_mass': float((depth_mask * verified).mean().item()),
        })
    return extras


def _build_terms_dict(l_rgb, l_depth, l_depth_seed, l_depth_dense, l_depth_fallback, l_pose, l_abs_t, l_abs_r, l_exp):
    return {
        'rgb': l_rgb,
        'depth_total': l_depth,
        'depth_seed': l_depth_seed,
        'depth_dense': l_depth_dense,
        'depth_fallback': l_depth_fallback,
        'pose_reg': l_pose,
        'abs_pose_trans': l_abs_t,
        'abs_pose_rot': l_abs_r,
        'exp_reg': l_exp,
    }


def build_stageA_loss_source_aware(
    render_rgb,
    render_depth,
    target_rgb,
    target_depth,
    confidence_mask,
    depth_source_map,
    viewpoint,
    beta_rgb: float,
    lambda_pose: float,
    lambda_exp: float,
    trans_weight: float,
    lambda_depth_seed: float = 1.0,
    lambda_depth_dense: float = 0.35,
    lambda_depth_fallback: float = 0.0,
    use_depth: bool = True,
    lambda_abs_pose: float = 0.0,
    lambda_abs_t: float = 0.0,
    lambda_abs_r: float = 0.0,
    abs_pose_robust: str = 'charbonnier',
    scene_scale: float = 1.0,
    return_terms: bool = False,
    rgb_confidence_mask=None,
    depth_confidence_mask=None,
):
    rgb_mask = confidence_mask if rgb_confidence_mask is None else rgb_confidence_mask
    depth_mask = rgb_mask if depth_confidence_mask is None else depth_confidence_mask
    l_rgb = masked_rgb_loss(render_rgb, target_rgb, rgb_mask, viewpoint)
    if use_depth:
        l_depth_seed = masked_depth_loss_by_source(render_depth, target_depth, depth_mask, depth_source_map, SEED_SOURCE_IDS)
        l_depth_dense = masked_depth_loss_by_source(render_depth, target_depth, depth_mask, depth_source_map, [DENSE_SOURCE_ID])
        l_depth_fallback = masked_depth_loss_by_source(render_depth, target_depth, depth_mask, depth_source_map, [FALLBACK_SOURCE_ID])
        l_depth = (
            float(lambda_depth_seed) * l_depth_seed
            + float(lambda_depth_dense) * l_depth_dense
            + float(lambda_depth_fallback) * l_depth_fallback
        )
    else:
        z = torch.zeros((), device=render_rgb.device)
        l_depth_seed = z
        l_depth_dense = z
        l_depth_fallback = z
        l_depth = z

    l_pose = pose_reg_loss(viewpoint, trans_weight)

    if float(lambda_abs_t) != 0.0 or float(lambda_abs_r) != 0.0:
        l_abs_pose, l_abs_t, l_abs_r, comps = absolute_pose_prior_loss_scaled(
            viewpoint=viewpoint,
            scene_scale=scene_scale,
            lambda_abs_t=lambda_abs_t,
            lambda_abs_r=lambda_abs_r,
            robust_type=abs_pose_robust,
        )
    else:
        legacy = absolute_pose_prior_loss(viewpoint)
        l_abs_pose = float(lambda_abs_pose) * legacy
        l_abs_t = l_abs_pose
        l_abs_r = torch.zeros_like(l_abs_t)
        comps = compute_abs_pose_components(viewpoint, scene_scale)

    l_exp = exposure_reg_loss(viewpoint)
    total = (
        float(beta_rgb) * l_rgb
        + (1.0 - float(beta_rgb)) * l_depth
        + float(lambda_pose) * l_pose
        + l_abs_pose
        + float(lambda_exp) * l_exp
    )

    extra_stats = _mask_extra_stats(rgb_mask, depth_mask, depth_source_map=depth_source_map, device=render_rgb.device)
    stats = _build_stats_dict(l_rgb, l_depth, l_depth_seed, l_depth_dense, l_depth_fallback, l_pose, l_abs_pose, l_abs_t, l_abs_r, l_exp, total, comps, extra_stats=extra_stats)
    if return_terms:
        terms = _build_terms_dict(l_rgb, l_depth, l_depth_seed, l_depth_dense, l_depth_fallback, l_pose, l_abs_t, l_abs_r, l_exp)
        return total, stats, terms
    return total, stats


def build_stageA_loss(
    render_rgb,
    render_depth,
    target_rgb,
    target_depth,
    confidence_mask,
    viewpoint,
    beta_rgb: float,
    lambda_pose: float,
    lambda_exp: float,
    trans_weight: float,
    use_depth: bool = True,
    lambda_abs_pose: float = 0.0,
    lambda_abs_t: float = 0.0,
    lambda_abs_r: float = 0.0,
    abs_pose_robust: str = 'charbonnier',
    scene_scale: float = 1.0,
    return_terms: bool = False,
    rgb_confidence_mask=None,
    depth_confidence_mask=None,
):
    rgb_mask = confidence_mask if rgb_confidence_mask is None else rgb_confidence_mask
    depth_mask = rgb_mask if depth_confidence_mask is None else depth_confidence_mask
    l_rgb = masked_rgb_loss(render_rgb, target_rgb, rgb_mask, viewpoint)
    l_depth = masked_depth_loss(render_depth, target_depth, confidence_mask) if use_depth else torch.zeros((), device=render_rgb.device)
    l_pose = pose_reg_loss(viewpoint, trans_weight)

    if float(lambda_abs_t) != 0.0 or float(lambda_abs_r) != 0.0:
        l_abs_pose, l_abs_t, l_abs_r, comps = absolute_pose_prior_loss_scaled(
            viewpoint=viewpoint,
            scene_scale=scene_scale,
            lambda_abs_t=lambda_abs_t,
            lambda_abs_r=lambda_abs_r,
            robust_type=abs_pose_robust,
        )
    else:
        legacy = absolute_pose_prior_loss(viewpoint)
        l_abs_pose = float(lambda_abs_pose) * legacy
        l_abs_t = l_abs_pose
        l_abs_r = torch.zeros_like(l_abs_t)
        comps = compute_abs_pose_components(viewpoint, scene_scale)

    l_exp = exposure_reg_loss(viewpoint)
    total = (
        float(beta_rgb) * l_rgb
        + (1.0 - float(beta_rgb)) * l_depth
        + float(lambda_pose) * l_pose
        + l_abs_pose
        + float(lambda_exp) * l_exp
    )
    z = torch.zeros_like(l_depth)
    extra_stats = _mask_extra_stats(rgb_mask, depth_mask, depth_source_map=None, device=render_rgb.device)
    stats = _build_stats_dict(l_rgb, l_depth, z, z, z, l_pose, l_abs_pose, l_abs_t, l_abs_r, l_exp, total, comps, extra_stats=extra_stats)

    if return_terms:
        terms = _build_terms_dict(l_rgb, l_depth, z, z, z, l_pose, l_abs_t, l_abs_r, l_exp)
        return total, stats, terms
    return total, stats

