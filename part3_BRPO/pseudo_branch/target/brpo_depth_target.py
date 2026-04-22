# -*- coding: utf-8 -*-
"""Utilities for building blended pseudo depth targets for Stage M3/M5."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .brpo_depth_densify import (
    build_depth_source_map_v2,
    build_sparse_log_depth_correction,
    densify_depth_correction_patchwise,
    reconstruct_dense_depth_from_correction,
)

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


def build_blended_target_depth_v2(
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    train_confidence_mask: np.ndarray,
    valid_left: np.ndarray | None = None,
    valid_right: np.ndarray | None = None,
    patch_size: int = 11,
    stride: int = 5,
    min_seed_count: int = 6,
    max_seed_delta_std: float = 0.08,
    continuous_confidence: np.ndarray | None = None,
    support_both_mask: np.ndarray | None = None,
    support_single_mask: np.ndarray | None = None,
    min_patch_confidence: float = 0.0,
    both_seed_count_relax: int = 0,
    single_std_tighten: float = 1.0,
) -> Dict[str, np.ndarray | Dict]:
    render_depth = _as_depth(render_depth)
    projected_depth_left = _as_depth(projected_depth_left)
    projected_depth_right = _as_depth(projected_depth_right)
    train_confidence_mask = _as_depth(train_confidence_mask)
    valid_left_mask = _as_valid_mask(valid_left, projected_depth_left)
    valid_right_mask = _as_valid_mask(valid_right, projected_depth_right)

    sparse = build_sparse_log_depth_correction(
        render_depth=render_depth,
        projected_depth_left=projected_depth_left,
        projected_depth_right=projected_depth_right,
        valid_left=valid_left_mask,
        valid_right=valid_right_mask,
    )
    dense = densify_depth_correction_patchwise(
        delta_seed_sparse=sparse["delta_seed_sparse"],
        seed_valid_mask=sparse["seed_valid_mask"],
        render_depth=render_depth,
        candidate_region=train_confidence_mask > 0,
        patch_size=patch_size,
        stride=stride,
        min_seed_count=min_seed_count,
        max_seed_delta_std=max_seed_delta_std,
        continuous_confidence=continuous_confidence,
        support_both_mask=support_both_mask,
        support_single_mask=support_single_mask,
        min_patch_confidence=min_patch_confidence,
        both_seed_count_relax=both_seed_count_relax,
        single_std_tighten=single_std_tighten,
    )
    recon = reconstruct_dense_depth_from_correction(
        render_depth=render_depth,
        delta_seed_sparse=sparse["delta_seed_sparse"],
        seed_valid_mask=sparse["seed_valid_mask"],
        depth_correction_dense=dense["depth_correction_dense"],
        depth_dense_valid_mask=dense["depth_dense_valid_mask"],
    )
    src = build_depth_source_map_v2(
        render_depth=render_depth,
        seed_source_map=sparse["seed_source_map"],
        seed_valid_mask=sparse["seed_valid_mask"],
        depth_dense_valid_mask=dense["depth_dense_valid_mask"],
    )

    total = float(render_depth.size)
    masked = train_confidence_mask > 0
    summary = {
        "depth_target_mode": "blended_m5_dense",
        "candidate_region_ratio": float(masked.sum() / total),
        "patch_size": int(patch_size),
        "stride": int(stride),
        "min_seed_count": int(min_seed_count),
        "max_seed_delta_std": float(max_seed_delta_std),
        "confidence_aware_densify": bool(continuous_confidence is not None or min_patch_confidence > 0 or both_seed_count_relax > 0 or single_std_tighten != 1.0),
        **sparse["summary"],
        **dense["summary"],
        **src["summary"],
    }

    return {
        "target_depth_for_refine_v2": recon["target_depth_dense"].astype(np.float32),
        "target_depth_dense_source_map": src["target_depth_dense_source_map"],
        "depth_correction_seed": sparse["delta_seed_sparse"].astype(np.float32),
        "depth_correction_dense": dense["depth_correction_dense"].astype(np.float32),
        "depth_seed_valid_mask": sparse["seed_valid_mask"].astype(np.float32),
        "depth_dense_valid_mask": dense["depth_dense_valid_mask"].astype(np.float32),
        "summary": summary,
    }

