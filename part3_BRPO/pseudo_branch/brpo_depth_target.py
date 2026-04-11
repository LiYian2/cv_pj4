# -*- coding: utf-8 -*-
"""Utilities for building blended pseudo depth targets for Stage M3."""
from __future__ import annotations

from typing import Dict

import numpy as np


SOURCE_RENDER_FALLBACK = 0
SOURCE_BOTH_FUSED = 1
SOURCE_LEFT_ONLY = 2
SOURCE_RIGHT_ONLY = 3
SOURCE_NO_DEPTH = 255

SOURCE_LEGEND = {
    int(SOURCE_RENDER_FALLBACK): "render_fallback",
    int(SOURCE_BOTH_FUSED): "both_fused",
    int(SOURCE_LEFT_ONLY): "left_only",
    int(SOURCE_RIGHT_ONLY): "right_only",
    int(SOURCE_NO_DEPTH): "no_depth",
}


def _as_depth(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _as_valid_mask(arr: np.ndarray | None, fallback_from_depth: np.ndarray) -> np.ndarray:
    if arr is None:
        return fallback_from_depth > 1e-6
    return np.asarray(arr, dtype=np.float32) > 0.5


def build_blended_target_depth(
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    valid_left: np.ndarray | None = None,
    valid_right: np.ndarray | None = None,
    fallback_mode: str = "render_depth",
    both_mode: str = "average",
) -> Dict[str, np.ndarray | Dict]:
    render_depth = _as_depth(render_depth)
    projected_depth_left = _as_depth(projected_depth_left)
    projected_depth_right = _as_depth(projected_depth_right)

    if render_depth.shape != projected_depth_left.shape or render_depth.shape != projected_depth_right.shape:
        raise ValueError(
            f"Depth shape mismatch: render={render_depth.shape}, left={projected_depth_left.shape}, right={projected_depth_right.shape}"
        )

    valid_left_mask = _as_valid_mask(valid_left, projected_depth_left)
    valid_right_mask = _as_valid_mask(valid_right, projected_depth_right)

    both_mask = valid_left_mask & valid_right_mask
    left_only_mask = valid_left_mask & (~valid_right_mask)
    right_only_mask = valid_right_mask & (~valid_left_mask)
    verified_mask = both_mask | left_only_mask | right_only_mask

    if fallback_mode == "render_depth":
        target_depth = render_depth.copy()
        source_map = np.full(render_depth.shape, SOURCE_RENDER_FALLBACK, dtype=np.uint8)
    elif fallback_mode == "none":
        target_depth = np.zeros_like(render_depth, dtype=np.float32)
        source_map = np.full(render_depth.shape, SOURCE_NO_DEPTH, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported fallback_mode={fallback_mode}")

    if both_mode != "average":
        raise ValueError(f"Unsupported both_mode={both_mode}")

    if both_mask.any():
        target_depth[both_mask] = 0.5 * (projected_depth_left[both_mask] + projected_depth_right[both_mask])
        source_map[both_mask] = SOURCE_BOTH_FUSED
    if left_only_mask.any():
        target_depth[left_only_mask] = projected_depth_left[left_only_mask]
        source_map[left_only_mask] = SOURCE_LEFT_ONLY
    if right_only_mask.any():
        target_depth[right_only_mask] = projected_depth_right[right_only_mask]
        source_map[right_only_mask] = SOURCE_RIGHT_ONLY

    h, w = render_depth.shape
    total = float(h * w)
    summary = {
        "depth_target_mode": "blended_m3",
        "fallback_mode": fallback_mode,
        "both_mode": both_mode,
        "num_verified_pixels": int(verified_mask.sum()),
        "num_both_pixels": int(both_mask.sum()),
        "num_left_only_pixels": int(left_only_mask.sum()),
        "num_right_only_pixels": int(right_only_mask.sum()),
        "num_render_fallback_pixels": int((source_map == SOURCE_RENDER_FALLBACK).sum()),
        "num_no_depth_pixels": int((source_map == SOURCE_NO_DEPTH).sum()),
        "verified_ratio": float(verified_mask.sum() / total),
        "both_ratio": float(both_mask.sum() / total),
        "left_only_ratio": float(left_only_mask.sum() / total),
        "right_only_ratio": float(right_only_mask.sum() / total),
        "render_fallback_ratio": float((source_map == SOURCE_RENDER_FALLBACK).sum() / total),
        "no_depth_ratio": float((source_map == SOURCE_NO_DEPTH).sum() / total),
        "source_legend": SOURCE_LEGEND,
    }

    return {
        "target_depth_for_refine": target_depth.astype(np.float32),
        "target_depth_for_refine_source_map": source_map,
        "verified_depth_mask": verified_mask.astype(np.float32),
        "verified_depth_left_mask": valid_left_mask.astype(np.float32),
        "verified_depth_right_mask": valid_right_mask.astype(np.float32),
        "summary": summary,
    }
