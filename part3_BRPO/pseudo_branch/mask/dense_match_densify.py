from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from pseudo_branch.mask.rgb_mask_inference import _accumulate_match_maps


def _resolve_blur_kernel(point_radius: int, blur_sigma: float, blur_kernel: int | None) -> int:
    if blur_kernel is not None and int(blur_kernel) > 0:
        k = int(blur_kernel)
    else:
        k = max(3, int(point_radius) * 4 + 1)
        sigma_hint = int(round(float(blur_sigma) * 2.0 + 1.0))
        k = max(k, sigma_hint)
    if k % 2 == 0:
        k += 1
    return max(k, 3)


def points_to_soft_mask(
    points_xy: np.ndarray,
    h: int,
    w: int,
    *,
    radius: int = 2,
    seed_values: np.ndarray | None = None,
    seed_mode: str = "binary",
) -> np.ndarray:
    """Rasterize sparse points into a dense seed mask before blur.

    seed_mode:
      - binary: every valid point contributes disk value 1.0
      - confidence_weighted: every valid point contributes disk value seed_values[i]

    Overlaps use max-composition, matching the peer-style "presence mask" semantics
    more closely than additive accumulation.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.size == 0:
        return mask

    pts = np.rint(points_xy).astype(np.int32)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    if not np.any(valid):
        return mask

    pts = pts[valid]
    if seed_mode == "confidence_weighted":
        if seed_values is None:
            raise ValueError("seed_values is required when seed_mode='confidence_weighted'")
        values = np.asarray(seed_values, dtype=np.float32)[valid]
        values = np.clip(values, 0.0, 1.0)
    elif seed_mode == "binary":
        values = np.ones((pts.shape[0],), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported seed_mode: {seed_mode}")

    radius = int(radius)
    for (x, y), value in zip(pts, values):
        tmp = np.zeros((h, w), dtype=np.float32)
        cv2.circle(tmp, (int(x), int(y)), radius, float(value), -1)
        mask = np.maximum(mask, tmp)
    return mask


def _normalize_soft_map(soft_map: np.ndarray, mode: str = "max") -> tuple[np.ndarray, float]:
    soft_map = np.asarray(soft_map, dtype=np.float32)
    if soft_map.size == 0:
        return soft_map, 1.0
    positive = soft_map[soft_map > 0]
    if positive.size == 0:
        return np.zeros_like(soft_map, dtype=np.float32), 1.0

    if mode == "max":
        scale = float(np.max(positive))
    elif mode == "p99":
        scale = float(np.quantile(positive, 0.99))
    elif mode == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported normalize_mode: {mode}")

    scale = max(scale, 1e-6)
    return np.clip(soft_map / scale, 0.0, 1.0).astype(np.float32), scale


def build_dense_match_maps(
    image_shape: Tuple[int, int],
    pts_fused: np.ndarray,
    conf: np.ndarray,
    *,
    point_radius: int = 2,
    blur_sigma: float = 2.0,
    blur_kernel: int | None = None,
    normalize_mode: str = "max",
    corr_threshold: float = 0.15,
    seed_mode: str = "binary",
) -> Dict[str, np.ndarray | dict]:
    """Build dense RGB-only support maps from sparse reciprocal match points.

    This is a drop-in alternative to `_accumulate_match_maps(...)` for the
    rgb_only_verification path. It preserves raw reciprocal support artifacts,
    then adds a peer-style densify -> Gaussian blur -> normalize -> threshold path.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    pts_fused = np.asarray(pts_fused, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    raw_maps = _accumulate_match_maps(
        image_shape=(h, w),
        pts_fused=pts_fused,
        conf=conf,
    )
    raw_support = np.asarray(raw_maps["support_mask"], dtype=np.float32)
    raw_conf_map = np.asarray(raw_maps["conf_map"], dtype=np.float32)
    match_density = np.asarray(raw_maps["match_density"], dtype=np.float32)

    if pts_fused.shape[0] == 0:
        summary = {
            "mode": "dense_match_v1",
            "num_input_points": 0,
            "raw_support_ratio": 0.0,
            "dense_support_ratio": 0.0,
            "raw_effective_weight": 0.0,
            "dense_effective_weight": 0.0,
            "dense_support_gain_vs_raw": 0.0,
            "kernel": 0,
            "sigma": float(blur_sigma),
            "threshold": float(corr_threshold),
            "seed_mode": str(seed_mode),
            "normalize_mode": str(normalize_mode),
            "normalize_scale": 1.0,
            "point_radius": int(point_radius),
        }
        return {
            "support_mask": raw_support,
            "conf_map": raw_conf_map,
            "match_density": match_density,
            "raw_support_mask": raw_support,
            "raw_conf_map": raw_conf_map,
            "dense_seed_mask": np.zeros((h, w), dtype=np.float32),
            "dense_soft_map": np.zeros((h, w), dtype=np.float32),
            "dense_support_mask": np.zeros((h, w), dtype=np.float32),
            "summary": summary,
        }

    seed_values = None
    if seed_mode == "confidence_weighted":
        scale = float(np.quantile(conf, 0.99)) if conf.size > 0 else 1.0
        scale = max(scale, 1e-8)
        seed_values = np.clip(conf / scale, 0.0, 1.0).astype(np.float32)

    dense_seed_mask = points_to_soft_mask(
        points_xy=pts_fused,
        h=h,
        w=w,
        radius=int(point_radius),
        seed_values=seed_values,
        seed_mode=str(seed_mode),
    )

    kernel = _resolve_blur_kernel(point_radius=int(point_radius), blur_sigma=float(blur_sigma), blur_kernel=blur_kernel)
    sigma = max(float(blur_sigma), 1e-6)
    if np.max(dense_seed_mask) > 0:
        dense_soft_map = cv2.GaussianBlur(dense_seed_mask, (kernel, kernel), sigmaX=sigma)
        dense_soft_map, norm_scale = _normalize_soft_map(dense_soft_map, mode=str(normalize_mode))
    else:
        dense_soft_map = np.zeros((h, w), dtype=np.float32)
        norm_scale = 1.0

    dense_support = (dense_soft_map >= float(corr_threshold)).astype(np.float32)

    raw_support_ratio = float(raw_support.mean())
    dense_support_ratio = float(dense_support.mean())
    raw_effective_weight = float(raw_conf_map.sum() / max(float(raw_conf_map.size), 1.0))
    dense_effective_weight = float(dense_soft_map.sum() / max(float(dense_soft_map.size), 1.0))
    summary = {
        "mode": "dense_match_v1",
        "num_input_points": int(pts_fused.shape[0]),
        "raw_support_ratio": raw_support_ratio,
        "dense_support_ratio": dense_support_ratio,
        "raw_effective_weight": raw_effective_weight,
        "dense_effective_weight": dense_effective_weight,
        "dense_support_gain_vs_raw": float(dense_support_ratio / max(raw_support_ratio, 1e-8)),
        "kernel": int(kernel),
        "sigma": float(sigma),
        "threshold": float(corr_threshold),
        "seed_mode": str(seed_mode),
        "normalize_mode": str(normalize_mode),
        "normalize_scale": float(norm_scale),
        "point_radius": int(point_radius),
    }

    return {
        "support_mask": dense_support.astype(np.float32),
        "conf_map": dense_soft_map.astype(np.float32),
        "match_density": match_density,
        "raw_support_mask": raw_support.astype(np.float32),
        "raw_conf_map": raw_conf_map.astype(np.float32),
        "dense_seed_mask": dense_seed_mask.astype(np.float32),
        "dense_soft_map": dense_soft_map.astype(np.float32),
        "dense_support_mask": dense_support.astype(np.float32),
        "summary": summary,
    }
