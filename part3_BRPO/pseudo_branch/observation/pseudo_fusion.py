# -*- coding: utf-8 -*-
"""Pseudo RGB fusion utilities.

Primary path: BRPO-style fusion where left/right branch weights are defined by
pseudo(target) -> reference overlap confidence, not by left-right agreement.

Backward compatibility: if geometry inputs are missing, falls back to a simple
legacy RGB-only residual fusion path so older callers do not crash.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from pseudo_branch.observation.brpo_reprojection_verify import (
    get_intrinsic_matrix_from_state,
    pose_c2w_from_state,
)


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _to_float_depth(depth: np.ndarray | None) -> Optional[np.ndarray]:
    if depth is None:
        return None
    return np.asarray(depth, dtype=np.float32)


def _save_mask_png(mask: np.ndarray, path: Path):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _save_float_png(arr: np.ndarray, path: Path, vmax: Optional[float] = None):
    arr = np.asarray(arr, dtype=np.float32)
    if vmax is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-8)
    img = np.clip(arr / vmax, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def _translation_consistency_scalar(
    pseudo_state: Dict,
    ref_state: Dict,
    translation_scale_tau: float = 1.0,
) -> float:
    pose_p = pose_c2w_from_state(pseudo_state)
    pose_r = pose_c2w_from_state(ref_state)
    tp = pose_p[:3, 3]
    tr = pose_r[:3, 3]
    dist = float(np.linalg.norm(tp - tr))
    tau = max(float(translation_scale_tau), 1e-8)
    return float(np.exp(-dist / tau))


def _backproject_pseudo_depth_to_world(
    pseudo_depth: np.ndarray,
    pseudo_state: Dict,
    valid_eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = pseudo_depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    valid = np.isfinite(pseudo_depth) & (pseudo_depth > float(valid_eps))
    if not valid.any():
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, valid, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    K = get_intrinsic_matrix_from_state(pseudo_state)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    d = pseudo_depth[valid].astype(np.float32)
    u = xx[valid]
    v = yy[valid]

    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    pts_cam = np.stack([x, y, d], axis=1)

    pose = pose_c2w_from_state(pseudo_state)
    R = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32)
    pts_world = (pts_cam @ R.T) + t[None, :]
    pts_uv = np.stack([u, v], axis=1).astype(np.float32)
    return pts_world.astype(np.float32), valid, u.astype(np.float32), v.astype(np.float32), pts_uv


def _project_world_to_ref(
    pts_world: np.ndarray,
    ref_state: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    if pts_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    pose = pose_c2w_from_state(ref_state)
    R = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32)
    pts_ref = (pts_world - t[None, :]) @ R
    z = pts_ref[:, 2]

    K = get_intrinsic_matrix_from_state(ref_state)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    uv = np.zeros((pts_world.shape[0], 2), dtype=np.float32)
    valid_z = z > 1e-6
    uv[valid_z, 0] = fx * (pts_ref[valid_z, 0] / z[valid_z]) + cx
    uv[valid_z, 1] = fy * (pts_ref[valid_z, 1] / z[valid_z]) + cy
    return uv, z.astype(np.float32)


def _sample_depth_nearest(depth: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    if uv.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
    x = np.rint(uv[:, 0]).astype(np.int32)
    y = np.rint(uv[:, 1]).astype(np.int32)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    sampled = np.zeros(uv.shape[0], dtype=np.float32)
    valid = np.zeros(uv.shape[0], dtype=bool)
    if in_bounds.any():
        vals = depth[y[in_bounds], x[in_bounds]].astype(np.float32)
        sampled[in_bounds] = vals
        valid[in_bounds] = np.isfinite(vals) & (vals > 1e-4)
    return sampled, valid


def compute_overlap_confidence_map(
    pseudo_state: Dict,
    ref_state: Dict,
    pseudo_depth: np.ndarray,
    ref_depth: np.ndarray,
    depth_consistency_tau: float = 0.15,
    translation_scale_tau: float = 1.0,
    overlap_valid_eps: float = 1e-4,
) -> Dict[str, np.ndarray | float | dict]:
    pseudo_depth = np.asarray(pseudo_depth, dtype=np.float32)
    ref_depth = np.asarray(ref_depth, dtype=np.float32)
    h, w = pseudo_depth.shape

    pts_world, valid_mask, _, _, pts_uv = _backproject_pseudo_depth_to_world(
        pseudo_depth, pseudo_state, valid_eps=overlap_valid_eps
    )
    uv_ref, z_ref = _project_world_to_ref(pts_world, ref_state)
    ref_depth_sampled, ref_depth_valid = _sample_depth_nearest(ref_depth, uv_ref)

    in_bounds = np.zeros(pts_world.shape[0], dtype=bool)
    if uv_ref.shape[0] > 0:
        in_bounds = (
            (uv_ref[:, 0] >= 0)
            & (uv_ref[:, 0] < w)
            & (uv_ref[:, 1] >= 0)
            & (uv_ref[:, 1] < h)
            & (z_ref > overlap_valid_eps)
        )

    overlap = in_bounds & ref_depth_valid
    rel_depth_err = np.full(pts_world.shape[0], np.inf, dtype=np.float32)
    if overlap.any():
        denom = ((np.abs(z_ref[overlap]) + np.abs(ref_depth_sampled[overlap])) * 0.5) + 1e-6
        rel_depth_err[overlap] = np.abs(z_ref[overlap] - ref_depth_sampled[overlap]) / denom

    tau = max(float(depth_consistency_tau), 1e-8)
    depth_consistency = np.zeros(pts_world.shape[0], dtype=np.float32)
    if overlap.any():
        depth_consistency[overlap] = np.exp(-rel_depth_err[overlap] / tau).astype(np.float32)

    translation_consistency = _translation_consistency_scalar(
        pseudo_state,
        ref_state,
        translation_scale_tau=translation_scale_tau,
    )

    overlap_mask = np.zeros((h, w), dtype=np.float32)
    projected_depth_map = np.zeros((h, w), dtype=np.float32)
    sampled_ref_depth_map = np.zeros((h, w), dtype=np.float32)
    rel_depth_error_map = np.zeros((h, w), dtype=np.float32)
    depth_consistency_map = np.zeros((h, w), dtype=np.float32)
    overlap_confidence = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.where(valid_mask)
    if yy.size > 0:
        overlap_mask[yy[overlap], xx[overlap]] = 1.0
        projected_depth_map[yy[overlap], xx[overlap]] = z_ref[overlap]
        sampled_ref_depth_map[yy[overlap], xx[overlap]] = ref_depth_sampled[overlap]
        rel_depth_error_map[yy[overlap], xx[overlap]] = rel_depth_err[overlap]
        depth_consistency_map[yy[overlap], xx[overlap]] = depth_consistency[overlap]
        overlap_confidence[yy[overlap], xx[overlap]] = depth_consistency[overlap] * float(translation_consistency)

    support_ratio = float((overlap_mask > 0.5).sum() / float(h * w))
    conf_positive = overlap_confidence[overlap_confidence > 0]
    mean_conf = float(conf_positive.mean()) if conf_positive.size else 0.0
    mean_rel_err = float(rel_depth_err[overlap].mean()) if overlap.any() else None

    return {
        "overlap_mask": overlap_mask,
        "projected_depth_map": projected_depth_map,
        "sampled_ref_depth_map": sampled_ref_depth_map,
        "rel_depth_error_map": rel_depth_error_map,
        "depth_consistency_map": depth_consistency_map,
        "overlap_confidence": overlap_confidence,
        "translation_consistency": float(translation_consistency),
        "stats": {
            "support_ratio": support_ratio,
            "support_pixels": int((overlap_mask > 0.5).sum()),
            "mean_overlap_confidence": mean_conf,
            "mean_rel_depth_error": mean_rel_err,
            "translation_consistency": float(translation_consistency),
            "depth_consistency_tau": float(depth_consistency_tau),
            "translation_scale_tau": float(translation_scale_tau),
        },
    }


def normalize_branch_weights(
    overlap_conf_left: np.ndarray,
    overlap_conf_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(overlap_conf_left, dtype=np.float32)
    right = np.asarray(overlap_conf_right, dtype=np.float32)
    denom = left + right
    w_left = np.zeros_like(left, dtype=np.float32)
    w_right = np.zeros_like(right, dtype=np.float32)
    valid = denom > 1e-8
    w_left[valid] = left[valid] / denom[valid]
    w_right[valid] = right[valid] / denom[valid]
    fused_conf = np.clip(left + right, 0.0, 1.0).astype(np.float32)
    return w_left, w_right, fused_conf


def fuse_residual_targets(
    I_render: np.ndarray,
    I_L: np.ndarray,
    I_R: np.ndarray,
    W_L: np.ndarray,
    W_R: np.ndarray,
) -> np.ndarray:
    r_l = I_L.astype(np.float32) - I_render.astype(np.float32)
    r_r = I_R.astype(np.float32) - I_render.astype(np.float32)
    fused = I_render.astype(np.float32) + W_L[:, :, None] * r_l + W_R[:, :, None] * r_r
    return np.clip(fused, 0.0, 255.0).astype(np.uint8)


def _legacy_rgb_only_fusion(
    render_rgb_path: str,
    target_rgb_left_path: str,
    target_rgb_right_path: str,
    conf_left: Optional[np.ndarray],
    conf_right: Optional[np.ndarray],
    output_dir: Path,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_render = _load_rgb(render_rgb_path)
    i_left = _load_rgb(target_rgb_left_path)
    i_right = _load_rgb(target_rgb_right_path)
    h, w = i_render.shape[:2]
    conf_left = np.ones((h, w), dtype=np.float32) if conf_left is None else np.asarray(conf_left, dtype=np.float32)
    conf_right = np.ones((h, w), dtype=np.float32) if conf_right is None else np.asarray(conf_right, dtype=np.float32)

    weight_sum = conf_left + conf_right + 1e-8
    w_left = conf_left / weight_sum
    w_right = conf_right / weight_sum
    fused = fuse_residual_targets(i_render, i_left, i_right, w_left, w_right)
    c_fused = np.clip(conf_left + conf_right, 0.0, 1.0).astype(np.float32)

    Image.fromarray(fused).save(output_dir / "target_rgb_fused.png")
    np.save(output_dir / "confidence_mask_fused.npy", c_fused)
    _save_mask_png(c_fused, output_dir / "confidence_mask_fused.png")

    stats = {
        "fusion_mode": "legacy_rgb_only_weighted_average",
        "support_ratio_fused": float((c_fused > 0.01).sum()) / float(h * w),
        "support_pixels_fused": int((c_fused > 0.01).sum()),
        "mean_conf_fused": float(c_fused[c_fused > 0].mean()) if (c_fused > 0).any() else 0.0,
    }
    with open(output_dir / "fusion_meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats, i_render, i_left, i_right, w_left, w_right, c_fused, c_fused


def run_fusion_for_sample(
    render_rgb_path: str,
    target_rgb_left_path: str,
    target_rgb_right_path: str,
    depth_left: Optional[np.ndarray],
    conf_left: Optional[np.ndarray],
    depth_right: Optional[np.ndarray],
    conf_right: Optional[np.ndarray],
    render_depth: Optional[np.ndarray],
    output_dir: str,
    use_depth_agreement: bool = False,
    tau_rgb: float = 10.0,
    tau_d: float = 0.1,
    tau_ld: float = 0.05,
    tau_c: float = 1.0,
    alpha1: float = 0.7,
    alpha2: float = 0.3,
    *,
    pseudo_state: Optional[Dict] = None,
    left_ref_state: Optional[Dict] = None,
    right_ref_state: Optional[Dict] = None,
    left_ref_depth: Optional[np.ndarray] = None,
    right_ref_depth: Optional[np.ndarray] = None,
    depth_consistency_tau: float = 0.15,
    translation_scale_tau: float = 1.0,
    overlap_valid_eps: float = 1e-4,
    exact_conf_left: Optional[np.ndarray] = None,
    exact_conf_right: Optional[np.ndarray] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run fusion for one sample.

    Preferred path uses BRPO-style target->reference overlap confidence when
    pseudo/ref states and depths are provided. If they are missing, falls back
    to a simple legacy RGB-only fusion to preserve compatibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    geometry_ready = (
        pseudo_state is not None
        and left_ref_state is not None
        and right_ref_state is not None
        and render_depth is not None
        and left_ref_depth is not None
        and right_ref_depth is not None
    )
    if not geometry_ready:
        return _legacy_rgb_only_fusion(
            render_rgb_path=render_rgb_path,
            target_rgb_left_path=target_rgb_left_path,
            target_rgb_right_path=target_rgb_right_path,
            conf_left=conf_left,
            conf_right=conf_right,
            output_dir=output_dir,
        )

    i_render = _load_rgb(render_rgb_path)
    i_left = _load_rgb(target_rgb_left_path)
    i_right = _load_rgb(target_rgb_right_path)
    render_depth = _to_float_depth(render_depth)
    left_ref_depth = _to_float_depth(left_ref_depth)
    right_ref_depth = _to_float_depth(right_ref_depth)

    left_geom = compute_overlap_confidence_map(
        pseudo_state=pseudo_state,
        ref_state=left_ref_state,
        pseudo_depth=render_depth,
        ref_depth=left_ref_depth,
        depth_consistency_tau=depth_consistency_tau,
        translation_scale_tau=translation_scale_tau,
        overlap_valid_eps=overlap_valid_eps,
    )
    right_geom = compute_overlap_confidence_map(
        pseudo_state=pseudo_state,
        ref_state=right_ref_state,
        pseudo_depth=render_depth,
        ref_depth=right_ref_depth,
        depth_consistency_tau=depth_consistency_tau,
        translation_scale_tau=translation_scale_tau,
        overlap_valid_eps=overlap_valid_eps,
    )

    if exact_conf_left is not None and exact_conf_right is not None:
        exact_conf_left = np.asarray(exact_conf_left, dtype=np.float32)
        exact_conf_right = np.asarray(exact_conf_right, dtype=np.float32)
        w_left, w_right, c_fused = normalize_branch_weights(exact_conf_left, exact_conf_right)
        fusion_weight_source = "exact_backend_confidence"
    else:
        w_left, w_right, c_fused = normalize_branch_weights(
            left_geom["overlap_confidence"],
            right_geom["overlap_confidence"],
        )
        fusion_weight_source = "proxy_overlap_confidence"
    fused = fuse_residual_targets(i_render, i_left, i_right, w_left, w_right)

    Image.fromarray(fused).save(output_dir / "target_rgb_fused.png")
    np.save(output_dir / "confidence_mask_fused.npy", c_fused)
    _save_mask_png(c_fused, output_dir / "confidence_mask_fused.png")

    np.save(output_dir / "fusion_weight_left.npy", w_left)
    np.save(output_dir / "fusion_weight_right.npy", w_right)
    np.save(output_dir / "overlap_conf_left.npy", left_geom["overlap_confidence"])
    np.save(output_dir / "overlap_conf_right.npy", right_geom["overlap_confidence"])
    np.save(output_dir / "overlap_mask_left.npy", left_geom["overlap_mask"])
    np.save(output_dir / "overlap_mask_right.npy", right_geom["overlap_mask"])
    np.save(output_dir / "projected_depth_left.npy", left_geom["projected_depth_map"])
    np.save(output_dir / "projected_depth_right.npy", right_geom["projected_depth_map"])
    np.save(output_dir / "sampled_ref_depth_left.npy", left_geom["sampled_ref_depth_map"])
    np.save(output_dir / "sampled_ref_depth_right.npy", right_geom["sampled_ref_depth_map"])
    np.save(output_dir / "ref_depth_left_render.npy", left_ref_depth)
    np.save(output_dir / "ref_depth_right_render.npy", right_ref_depth)

    _save_float_png(w_left, diag_dir / "fusion_weight_left.png", vmax=1.0)
    _save_float_png(w_right, diag_dir / "fusion_weight_right.png", vmax=1.0)
    _save_float_png(left_geom["overlap_confidence"], diag_dir / "overlap_conf_left.png", vmax=1.0)
    _save_float_png(right_geom["overlap_confidence"], diag_dir / "overlap_conf_right.png", vmax=1.0)
    _save_mask_png(left_geom["overlap_mask"], diag_dir / "overlap_mask_left.png")
    _save_mask_png(right_geom["overlap_mask"], diag_dir / "overlap_mask_right.png")
    _save_float_png(left_geom["projected_depth_map"], diag_dir / "projected_depth_left.png")
    _save_float_png(right_geom["projected_depth_map"], diag_dir / "projected_depth_right.png")
    _save_float_png(left_geom["rel_depth_error_map"], diag_dir / "rel_depth_error_left.png")
    _save_float_png(right_geom["rel_depth_error_map"], diag_dir / "rel_depth_error_right.png")

    support_ratio = float((c_fused > 0.01).sum() / float(c_fused.size))
    mean_conf = float(c_fused[c_fused > 0].mean()) if (c_fused > 0).any() else 0.0
    stats = {
        "fusion_mode": "brpo_overlap_confidence_v1",
        "fusion_weight_source": fusion_weight_source,
        "support_ratio_fused": support_ratio,
        "support_pixels_fused": int((c_fused > 0.01).sum()),
        "mean_conf_fused": mean_conf,
        "left": left_geom["stats"],
        "right": right_geom["stats"],
        "depth_consistency_tau": float(depth_consistency_tau),
        "translation_scale_tau": float(translation_scale_tau),
        "overlap_valid_eps": float(overlap_valid_eps),
        "legacy_args_ignored": {
            "use_depth_agreement": bool(use_depth_agreement),
            "tau_rgb": float(tau_rgb),
            "tau_d": float(tau_d),
            "tau_ld": float(tau_ld),
            "tau_c": float(tau_c),
            "alpha1": float(alpha1),
            "alpha2": float(alpha2),
        },
    }
    with open(output_dir / "fusion_meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    # Phase T1 补充：export branch-native artifacts for exact T~ backend
    branch_native_meta = export_branch_native_artifacts(
        output_dir=output_dir,
        i_render=i_render,
        i_left=i_left,
        i_right=i_right,
        w_left=w_left,
        w_right=w_right,
        left_geom=left_geom,
        right_geom=right_geom,
        export_branch_rgb=False,  # exact backend uses pseudo RGB from cache, not fusion output
    )
    stats["branch_native_exact_exported"] = True
    stats["branch_native_meta"] = branch_native_meta
    
    return stats, i_render, i_left, i_right, w_left, w_right, left_geom["overlap_confidence"], c_fused



def export_branch_native_artifacts(
    output_dir: Path,
    i_render: np.ndarray,
    i_left: np.ndarray,
    i_right: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    left_geom: Dict,
    right_geom: Dict,
    export_branch_rgb: bool = False,
):
    """Export branch-native artifacts for exact T~ backend.
    
    This is separate from target_rgb_fused.png which is kept as debug/control.
    Exact T~ backend should consume branch-native provenance-aware artifacts.
    """    
    output_dir = Path(output_dir)
    exact_dir = output_dir / "branch_native_exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save branch-native weights and confidence (already done in run_fusion_for_sample)
    # Here we add provenance markers in metadata
    
    # Optionally export branch-native RGB
    if export_branch_rgb:
        Image.fromarray(i_left).save(exact_dir / "target_rgb_left.png")
        Image.fromarray(i_right).save(exact_dir / "target_rgb_right.png")
        Image.fromarray(i_render).save(exact_dir / "render_rgb.png")
    
    # Metadata for branch-native artifacts
    meta = {
        "fusion_output_mode": "branch_native_exact",
        "target_rgb_fused_role": "debug_control",
        "branch_native_role": "exact_upstream_truth",
        "provenance": {
            "fusion_weight_left": "branch_native_from_left_ref",
            "fusion_weight_right": "branch_native_from_right_ref",
            "overlap_conf_left": "overlap_confidence_to_left_ref",
            "overlap_conf_right": "overlap_confidence_to_right_ref",
            "projected_depth_left": "projected_from_left_ref",
            "projected_depth_right": "projected_from_right_ref",
        },
        "export_branch_rgb": export_branch_rgb,
    }
    with open(exact_dir / "branch_native_meta.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)
    
    return meta
def get_fusion_diag_images(
    I_L: np.ndarray,
    I_R: np.ndarray,
    W_L: np.ndarray,
    W_R: np.ndarray,
    A: np.ndarray,
    C_fused: np.ndarray,
    conf_left: Optional[np.ndarray],
    conf_right: Optional[np.ndarray],
):
    """Keep a compatibility helper for external diagnostic callers."""
    diag = {}
    diff = np.abs(I_L.astype(np.float32) - I_R.astype(np.float32))
    diag["rgb_disagreement"] = np.clip(np.sum(diff, axis=2), 0, 255).astype(np.uint8)
    diag["weight_left"] = (np.clip(W_L, 0, 1) * 255).astype(np.uint8)
    diag["weight_right"] = (np.clip(W_R, 0, 1) * 255).astype(np.uint8)
    diag["agreement"] = (np.clip(A, 0, 1) * 255).astype(np.uint8)
    diag["confidence_fused"] = (np.clip(C_fused, 0, 1) * 255).astype(np.uint8)
    if conf_left is not None:
        diag["support_left"] = ((np.asarray(conf_left) > 0.01).astype(np.uint8) * 255)
    if conf_right is not None:
        diag["support_right"] = ((np.asarray(conf_right) > 0.01).astype(np.uint8) * 255)
    diag["support_fused"] = ((np.asarray(C_fused) > 0.01).astype(np.uint8) * 255)
    return diag
