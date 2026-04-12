# -*- coding: utf-8 -*-
"""Depth correction densification utilities for BRPO M5."""
from __future__ import annotations

from typing import Dict

import numpy as np

SOURCE_RENDER_FALLBACK = 0
SOURCE_SEED_BOTH = 1
SOURCE_SEED_LEFT = 2
SOURCE_SEED_RIGHT = 3
SOURCE_DENSIFIED = 4
SOURCE_NO_DEPTH = 255

SOURCE_LEGEND_V2 = {
    int(SOURCE_RENDER_FALLBACK): "render_fallback",
    int(SOURCE_SEED_BOTH): "seed_both_fused",
    int(SOURCE_SEED_LEFT): "seed_left_only",
    int(SOURCE_SEED_RIGHT): "seed_right_only",
    int(SOURCE_DENSIFIED): "densified_from_seed",
    int(SOURCE_NO_DEPTH): "no_depth",
}


def _safe_log_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    return np.log(np.clip(depth, 1e-6, None))


def build_sparse_log_depth_correction(
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    valid_left: np.ndarray,
    valid_right: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    valid_left = np.asarray(valid_left, dtype=np.float32) > 0.5
    valid_right = np.asarray(valid_right, dtype=np.float32) > 0.5

    render_valid = render_depth > 1e-6
    left = valid_left & render_valid & (projected_depth_left > 1e-6)
    right = valid_right & render_valid & (projected_depth_right > 1e-6)
    both = left & right
    left_only = left & (~right)
    right_only = right & (~left)
    seed_valid = both | left_only | right_only

    delta_seed = np.zeros_like(render_depth, dtype=np.float32)
    source_map = np.full(render_depth.shape, SOURCE_RENDER_FALLBACK, dtype=np.uint8)

    render_log = _safe_log_depth(render_depth)
    if both.any():
        delta_seed[both] = 0.5 * (_safe_log_depth(projected_depth_left[both]) + _safe_log_depth(projected_depth_right[both])) - render_log[both]
        source_map[both] = SOURCE_SEED_BOTH
    if left_only.any():
        delta_seed[left_only] = _safe_log_depth(projected_depth_left[left_only]) - render_log[left_only]
        source_map[left_only] = SOURCE_SEED_LEFT
    if right_only.any():
        delta_seed[right_only] = _safe_log_depth(projected_depth_right[right_only]) - render_log[right_only]
        source_map[right_only] = SOURCE_SEED_RIGHT

    total = float(render_depth.size)
    summary = {
        "seed_valid_ratio": float(seed_valid.sum() / total),
        "seed_both_ratio": float(both.sum() / total),
        "seed_left_ratio": float(left_only.sum() / total),
        "seed_right_ratio": float(right_only.sum() / total),
    }

    return {
        "delta_seed_sparse": delta_seed,
        "seed_valid_mask": seed_valid.astype(np.float32),
        "seed_source_map": source_map,
        "summary": summary,
    }


def densify_depth_correction_patchwise(
    delta_seed_sparse: np.ndarray,
    seed_valid_mask: np.ndarray,
    render_depth: np.ndarray,
    candidate_region: np.ndarray,
    patch_size: int = 11,
    stride: int = 5,
    min_seed_count: int = 6,
    max_seed_delta_std: float = 0.08,
) -> Dict[str, np.ndarray | Dict]:
    delta_seed_sparse = np.asarray(delta_seed_sparse, dtype=np.float32)
    seed_valid = np.asarray(seed_valid_mask, dtype=np.float32) > 0.5
    render_depth = np.asarray(render_depth, dtype=np.float32)
    candidate_region = np.asarray(candidate_region, dtype=np.float32) > 0.5

    h, w = render_depth.shape
    half = patch_size // 2
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)

    for cy in range(0, h, stride):
        y0 = max(0, cy - half)
        y1 = min(h, cy + half + 1)
        for cx in range(0, w, stride):
            x0 = max(0, cx - half)
            x1 = min(w, cx + half + 1)

            patch_seed = seed_valid[y0:y1, x0:x1]
            patch_candidate = candidate_region[y0:y1, x0:x1] & (render_depth[y0:y1, x0:x1] > 1e-6)
            if not patch_candidate.any():
                continue
            seed_values = delta_seed_sparse[y0:y1, x0:x1][patch_seed]
            if seed_values.size < min_seed_count:
                continue
            if float(seed_values.std()) > float(max_seed_delta_std):
                continue
            patch_delta = float(np.median(seed_values))
            acc[y0:y1, x0:x1][patch_candidate] += patch_delta
            cnt[y0:y1, x0:x1][patch_candidate] += 1.0

    dense_valid = cnt > 0
    delta_dense = np.zeros_like(delta_seed_sparse, dtype=np.float32)
    delta_dense[dense_valid] = acc[dense_valid] / np.clip(cnt[dense_valid], 1.0, None)

    new_dense_only = dense_valid & (~seed_valid)
    total = float(render_depth.size)
    summary = {
        "patch_size": int(patch_size),
        "stride": int(stride),
        "min_seed_count": int(min_seed_count),
        "max_seed_delta_std": float(max_seed_delta_std),
        "dense_valid_ratio": float(dense_valid.sum() / total),
        "densified_only_ratio": float(new_dense_only.sum() / total),
    }
    return {
        "depth_correction_dense": delta_dense,
        "depth_dense_valid_mask": dense_valid.astype(np.float32),
        "depth_densified_only_mask": new_dense_only.astype(np.float32),
        "summary": summary,
    }


def reconstruct_dense_depth_from_correction(
    render_depth: np.ndarray,
    delta_seed_sparse: np.ndarray,
    seed_valid_mask: np.ndarray,
    depth_correction_dense: np.ndarray,
    depth_dense_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    delta_seed_sparse = np.asarray(delta_seed_sparse, dtype=np.float32)
    seed_valid = np.asarray(seed_valid_mask, dtype=np.float32) > 0.5
    delta_dense = np.asarray(depth_correction_dense, dtype=np.float32)
    dense_valid = np.asarray(depth_dense_valid_mask, dtype=np.float32) > 0.5

    target = render_depth.copy()
    target[dense_valid] = render_depth[dense_valid] * np.exp(delta_dense[dense_valid])
    target[seed_valid] = render_depth[seed_valid] * np.exp(delta_seed_sparse[seed_valid])
    return {"target_depth_dense": target.astype(np.float32)}


def build_depth_source_map_v2(
    render_depth: np.ndarray,
    seed_source_map: np.ndarray,
    seed_valid_mask: np.ndarray,
    depth_dense_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    seed_source_map = np.asarray(seed_source_map, dtype=np.uint8)
    seed_valid = np.asarray(seed_valid_mask, dtype=np.float32) > 0.5
    dense_valid = np.asarray(depth_dense_valid_mask, dtype=np.float32) > 0.5

    source_map = np.full(render_depth.shape, SOURCE_RENDER_FALLBACK, dtype=np.uint8)
    source_map[dense_valid] = SOURCE_DENSIFIED
    source_map[seed_valid] = seed_source_map[seed_valid]
    no_depth = render_depth <= 1e-6
    source_map[no_depth] = SOURCE_NO_DEPTH

    total = float(render_depth.size)
    summary = {
        "seed_ratio": float(seed_valid.sum() / total),
        "densified_ratio": float(((source_map == SOURCE_DENSIFIED)).sum() / total),
        "render_fallback_ratio": float(((source_map == SOURCE_RENDER_FALLBACK)).sum() / total),
        "no_depth_ratio": float(((source_map == SOURCE_NO_DEPTH)).sum() / total),
        "source_legend": SOURCE_LEGEND_V2,
    }
    return {"target_depth_dense_source_map": source_map, "summary": summary}
