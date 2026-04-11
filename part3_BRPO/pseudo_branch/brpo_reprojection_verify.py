# -*- coding: utf-8 -*-
"""BRPO-style single-branch reprojection verification.

Phase B/M3 prototype:
- no EDP dependency
- no refine integration
- use matcher + ref depth rendered on-demand from stage PLY
- additionally export pseudo-view sparse verified depth maps for Stage M3
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2


def get_intrinsic_matrix_from_state(state: Dict) -> np.ndarray:
    fx = float(state["fx"])
    fy = float(state["fy"])
    cx = float(state["cx"])
    cy = float(state["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def pose_c2w_from_state(state: Dict) -> np.ndarray:
    return np.asarray(state["pose_c2w"], dtype=np.float32)


def create_viewpoint_from_state(state: Dict, device: str = "cuda"):
    pose_c2w = pose_c2w_from_state(state)
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    T = pose_w2c[:3, 3]

    fx = float(state["fx"])
    fy = float(state["fy"])
    cx = float(state["cx"])
    cy = float(state["cy"])
    W = int(state["image_width"])
    H = int(state["image_height"])
    FoVx = float(state["FoVx"])
    FoVy = float(state["FoVy"])

    R_tensor = torch.from_numpy(R).float().to(device)
    T_tensor = torch.from_numpy(T).float().to(device)
    world_view_transform = getWorld2View2(R_tensor, T_tensor).transpose(0, 1)
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    ).transpose(0, 1).to(device)
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0)
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    vp = type("ReplayCamera", (), {})()
    vp.uid = int(state.get("uid", state["frame_id"]))
    vp.R = R_tensor
    vp.T = T_tensor
    vp.FoVx = FoVx
    vp.FoVy = FoVy
    vp.image_height = H
    vp.image_width = W
    vp.original_image = None
    vp.exposure_a = torch.tensor(0.0, device=device)
    vp.exposure_b = torch.tensor(0.0, device=device)
    vp.world_view_transform = world_view_transform
    vp.projection_matrix = projection_matrix
    vp.full_proj_transform = full_proj_transform
    vp.camera_center = camera_center
    vp.cam_rot_delta = torch.zeros(3, device=device)
    vp.cam_trans_delta = torch.zeros(3, device=device)
    vp.fx = fx
    vp.fy = fy
    vp.cx = cx
    vp.cy = cy
    return vp


def render_depth_from_state(gaussians, state: Dict, pipe, background, device: str = "cuda") -> np.ndarray:
    viewpoint = create_viewpoint_from_state(state, device=device)
    render_pkg = render(viewpoint, gaussians, pipe, background)
    return render_pkg["depth"].squeeze().detach().cpu().numpy().astype(np.float32)


def find_neighbor_kfs(frame_id: int, kf_indices: List[int]) -> Tuple[int, int]:
    left = max([k for k in kf_indices if k < frame_id], default=None)
    right = min([k for k in kf_indices if k > frame_id], default=None)
    return left, right


def sample_depth_at_points(depth: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.zeros((pts.shape[0],), dtype=np.float32)
    valid = np.zeros((pts.shape[0],), dtype=bool)
    valid[in_bounds] = depth[y[in_bounds], x[in_bounds]] > 1e-4
    values[in_bounds] = depth[y[in_bounds], x[in_bounds]]
    return values, valid


def backproject_ref_points_to_world(pts_ref: np.ndarray, ref_depth: np.ndarray, ref_state: Dict):
    K = get_intrinsic_matrix_from_state(ref_state)
    pose_c2w = pose_c2w_from_state(ref_state)
    d, valid_depth = sample_depth_at_points(ref_depth, pts_ref)
    uv1 = np.concatenate([pts_ref, np.ones((pts_ref.shape[0], 1), dtype=np.float32)], axis=1)
    rays = (np.linalg.inv(K) @ uv1.T).T
    pts_cam = rays * d[:, None]
    pts_world = (pose_c2w[:3, :3] @ pts_cam.T + pose_c2w[:3, 3:4]).T
    return pts_world, valid_depth, d


def project_world_to_pseudo(pts_world: np.ndarray, pseudo_state: Dict):
    K = get_intrinsic_matrix_from_state(pseudo_state)
    pose_c2w = pose_c2w_from_state(pseudo_state)
    w2c = np.linalg.inv(pose_c2w)
    pts_cam = (w2c[:3, :3] @ pts_world.T + w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / (z[:, None] + 1e-8)
    return uv, z


def verify_single_branch(
    pseudo_state: Dict,
    ref_state: Dict,
    pseudo_depth: np.ndarray,
    ref_depth: np.ndarray,
    pts_pseudo: np.ndarray,
    pts_ref: np.ndarray,
    tau_reproj_px: float = 4.0,
    tau_rel_depth: float = 0.15,
):
    h = int(pseudo_state["image_height"])
    w = int(pseudo_state["image_width"])

    pts_world, valid_ref_depth, ref_depth_samples = backproject_ref_points_to_world(pts_ref, ref_depth, ref_state)
    reproj_uv, reproj_z = project_world_to_pseudo(pts_world, pseudo_state)

    in_bounds = (
        (reproj_uv[:, 0] >= 0) & (reproj_uv[:, 0] < w) &
        (reproj_uv[:, 1] >= 0) & (reproj_uv[:, 1] < h) &
        (reproj_z > 1e-4)
    )
    reproj_err = np.linalg.norm(reproj_uv - pts_pseudo, axis=1)

    pseudo_depth_samples, valid_pseudo_depth = sample_depth_at_points(pseudo_depth, pts_pseudo)
    rel_depth_err = np.full_like(reproj_err, fill_value=np.inf, dtype=np.float32)
    depth_valid = valid_ref_depth & valid_pseudo_depth & (pseudo_depth_samples > 1e-4)
    rel_depth_err[depth_valid] = np.abs(reproj_z[depth_valid] - pseudo_depth_samples[depth_valid]) / np.maximum(pseudo_depth_samples[depth_valid], 1e-6)

    support = valid_ref_depth & in_bounds & valid_pseudo_depth & (reproj_err < tau_reproj_px) & (rel_depth_err < tau_rel_depth)

    support_mask = np.zeros((h, w), dtype=np.float32)
    reproj_error_map = np.zeros((h, w), dtype=np.float32)
    rel_depth_error_map = np.zeros((h, w), dtype=np.float32)
    match_density = np.zeros((h, w), dtype=np.int32)
    projected_depth_map = np.zeros((h, w), dtype=np.float32)
    projected_depth_valid_mask = np.zeros((h, w), dtype=np.float32)

    x = np.round(pts_pseudo[:, 0]).astype(int)
    y = np.round(pts_pseudo[:, 1]).astype(int)
    pix_valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    for i in np.where(pix_valid)[0]:
        xi, yi = x[i], y[i]
        match_density[yi, xi] += 1
        if support[i]:
            support_mask[yi, xi] = 1.0
            if projected_depth_valid_mask[yi, xi] == 0 or rel_depth_err[i] < rel_depth_error_map[yi, xi] or rel_depth_error_map[yi, xi] == 0:
                projected_depth_map[yi, xi] = reproj_z[i]
                projected_depth_valid_mask[yi, xi] = 1.0
        if reproj_error_map[yi, xi] == 0 or reproj_err[i] < reproj_error_map[yi, xi]:
            reproj_error_map[yi, xi] = reproj_err[i]
        if np.isfinite(rel_depth_err[i]) and (rel_depth_error_map[yi, xi] == 0 or rel_depth_err[i] < rel_depth_error_map[yi, xi]):
            rel_depth_error_map[yi, xi] = rel_depth_err[i]

    num_projected_depth = int((projected_depth_valid_mask > 0.5).sum())
    stats = {
        "num_matches": int(len(pts_pseudo)),
        "num_valid_ref_depth": int(valid_ref_depth.sum()),
        "num_valid_pseudo_depth": int(valid_pseudo_depth.sum()),
        "num_support": int(support.sum()),
        "num_projected_depth": num_projected_depth,
        "support_ratio_vs_matches": float(support.sum() / max(len(pts_pseudo), 1)),
        "support_ratio_vs_image": float((support_mask > 0).sum() / float(h * w)),
        "projected_depth_ratio_vs_image": float(num_projected_depth / float(h * w)),
        "mean_reproj_error": float(reproj_err[support].mean()) if support.any() else None,
        "mean_rel_depth_error": float(rel_depth_err[support].mean()) if support.any() else None,
        "tau_reproj_px": float(tau_reproj_px),
        "tau_rel_depth": float(tau_rel_depth),
    }

    return {
        "support_mask": support_mask,
        "reproj_error_map": reproj_error_map,
        "rel_depth_error_map": rel_depth_error_map,
        "match_density": match_density.astype(np.float32),
        "projected_depth_map": projected_depth_map,
        "projected_depth_valid_mask": projected_depth_valid_mask,
        "stats": stats,
    }


def save_float_map_png(arr: np.ndarray, path: str, scale: float = None):
    arr = arr.astype(np.float32)
    if scale is None:
        vmax = float(arr.max()) if arr.max() > 0 else 1.0
    else:
        vmax = float(scale)
    img = np.clip(arr / max(vmax, 1e-8), 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def save_mask_png(mask: np.ndarray, path: str):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
