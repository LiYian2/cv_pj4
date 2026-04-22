from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from pseudo_branch.target.depth_supervision_v2 import (
    SOURCE_BOTH_WEIGHTED,
    SOURCE_LEFT,
    SOURCE_RIGHT,
)


JOINT_NONE = 0.0
JOINT_SINGLE = 0.5
JOINT_BOTH = 1.0


def _save_mask_png(mask: np.ndarray, path: Path):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _save_float_png(arr: np.ndarray, path: Path, vmax: float | None = None):
    arr = np.asarray(arr, dtype=np.float32)
    if vmax is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-8)
    img = np.clip(arr / vmax, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def _to_bool(arr: np.ndarray | None, ref: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.zeros_like(ref, dtype=bool)
    return np.asarray(arr, dtype=np.float32) > 0.5


def build_joint_confidence_from_rgb_and_depth(
    raw_rgb_confidence: np.ndarray,
    raw_rgb_confidence_cont: np.ndarray | None,
    depth_source_map: np.ndarray,
    projected_valid_left: np.ndarray | None = None,
    projected_valid_right: np.ndarray | None = None,
) -> Dict[str, np.ndarray | Dict]:
    raw_rgb_confidence = np.asarray(raw_rgb_confidence, dtype=np.float32)
    raw_rgb_confidence_cont = None if raw_rgb_confidence_cont is None else np.asarray(raw_rgb_confidence_cont, dtype=np.float32)
    depth_source_map = np.asarray(depth_source_map)

    geom_both = depth_source_map == SOURCE_BOTH_WEIGHTED
    geom_single = np.isin(depth_source_map, [SOURCE_LEFT, SOURCE_RIGHT])
    geom_verified = geom_both | geom_single

    geometry_tier = np.zeros_like(raw_rgb_confidence, dtype=np.float32)
    geometry_tier[geom_single] = JOINT_SINGLE
    geometry_tier[geom_both] = JOINT_BOTH

    joint_confidence = np.minimum(np.clip(raw_rgb_confidence, 0.0, 1.0), geometry_tier).astype(np.float32)

    if raw_rgb_confidence_cont is None:
        joint_confidence_cont = joint_confidence.copy()
    else:
        joint_confidence_cont = (np.clip(raw_rgb_confidence_cont, 0.0, 1.0) * geometry_tier).astype(np.float32)

    left_valid = _to_bool(projected_valid_left, raw_rgb_confidence)
    right_valid = _to_bool(projected_valid_right, raw_rgb_confidence)

    summary = {
        'joint_nonzero_ratio': float((joint_confidence > 0).sum() / joint_confidence.size),
        'joint_mean_positive': float(joint_confidence[joint_confidence > 0].mean()) if (joint_confidence > 0).any() else 0.0,
        'joint_cont_nonzero_ratio': float((joint_confidence_cont > 0).sum() / joint_confidence_cont.size),
        'joint_cont_mean_positive': float(joint_confidence_cont[joint_confidence_cont > 0].mean()) if (joint_confidence_cont > 0).any() else 0.0,
        'joint_both_ratio': float((joint_confidence >= JOINT_BOTH - 1e-6).sum() / joint_confidence.size),
        'joint_single_ratio': float(((joint_confidence > 0) & (joint_confidence < JOINT_BOTH - 1e-6)).sum() / joint_confidence.size),
        'geometry_verified_ratio': float(geom_verified.sum() / geom_verified.size),
        'geometry_both_ratio': float(geom_both.sum() / geom_both.size),
        'geometry_single_ratio': float(geom_single.sum() / geom_single.size),
        'rgb_nonzero_ratio': float((raw_rgb_confidence > 0).sum() / raw_rgb_confidence.size),
        'projected_valid_left_ratio': float(left_valid.sum() / left_valid.size) if projected_valid_left is not None else None,
        'projected_valid_right_ratio': float(right_valid.sum() / right_valid.size) if projected_valid_right is not None else None,
        'coverage_delta_vs_rgb': float(((joint_confidence > 0).sum() - (raw_rgb_confidence > 0).sum()) / raw_rgb_confidence.size),
        'policy': {
            'version': 'brpo_joint_v1',
            'discrete_rule': 'joint=min(raw_rgb_confidence_v2, geometry_tier); geometry_tier={single:0.5,both:1.0,fallback:0.0}',
            'continuous_rule': 'joint_cont=raw_rgb_confidence_cont_v2*geometry_tier',
            'joint_depth_target_rule': 'reuse_target_depth_for_refine_v2_brpo_values',
        },
    }

    return {
        'joint_confidence_v2': joint_confidence,
        'joint_confidence_cont_v2': joint_confidence_cont,
        'geometry_tier_v2': geometry_tier.astype(np.float32),
        'summary': summary,
    }


def build_joint_depth_target(
    target_depth_for_refine_v2_brpo: np.ndarray,
) -> np.ndarray:
    return np.asarray(target_depth_for_refine_v2_brpo, dtype=np.float32).copy()


def write_joint_signal_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / 'joint_confidence_v2.npy', result['joint_confidence_v2'])
    np.save(frame_out / 'joint_confidence_cont_v2.npy', result['joint_confidence_cont_v2'])
    np.save(frame_out / 'joint_depth_target_v2.npy', result['joint_depth_target_v2'])
    np.save(diag_dir / 'joint_geometry_tier_v2.npy', result['geometry_tier_v2'])

    _save_mask_png(result['joint_confidence_v2'], frame_out / 'joint_confidence_v2.png')
    _save_float_png(result['joint_confidence_cont_v2'], frame_out / 'joint_confidence_cont_v2.png', vmax=1.0)
    _save_float_png(result['joint_depth_target_v2'], frame_out / 'joint_depth_target_v2.png')
    _save_float_png(result['geometry_tier_v2'], diag_dir / 'joint_geometry_tier_v2.png', vmax=1.0)

    with open(frame_out / 'joint_meta_v2.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
