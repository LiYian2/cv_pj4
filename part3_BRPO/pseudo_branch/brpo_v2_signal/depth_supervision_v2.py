from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

SOURCE_NONE = 0
SOURCE_LEFT = 1
SOURCE_RIGHT = 2
SOURCE_BOTH_WEIGHTED = 3
SOURCE_RENDER_FALLBACK = 4


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


def _save_source_map_png(source_map: np.ndarray, path: Path):
    arr = np.asarray(source_map, dtype=np.uint8)
    vmax = max(int(arr.max()), 1)
    img = (arr.astype(np.float32) / float(vmax) * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def build_depth_supervision_v2(
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    raw_rgb_confidence: np.ndarray,
    raw_rgb_confidence_cont: np.ndarray | None = None,
    projected_valid_left: np.ndarray | None = None,
    projected_valid_right: np.ndarray | None = None,
    min_rgb_conf_for_depth: float = 0.5,
    fallback_mode: str = 'render_depth',
    both_mode: str = 'weighted_by_fusion',
    single_mode: str = 'single_branch_projected',
    use_continuous_reweight: bool = True,
) -> Dict[str, np.ndarray | Dict]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    fusion_weight_left = np.asarray(fusion_weight_left, dtype=np.float32)
    fusion_weight_right = np.asarray(fusion_weight_right, dtype=np.float32)
    raw_rgb_confidence = np.asarray(raw_rgb_confidence, dtype=np.float32)
    raw_rgb_confidence_cont = None if raw_rgb_confidence_cont is None else np.asarray(raw_rgb_confidence_cont, dtype=np.float32)

    valid_left = np.asarray(projected_valid_left, dtype=np.float32) > 0.5 if projected_valid_left is not None else projected_depth_left > 1e-6
    valid_right = np.asarray(projected_valid_right, dtype=np.float32) > 0.5 if projected_valid_right is not None else projected_depth_right > 1e-6
    rgb_active = raw_rgb_confidence >= float(min_rgb_conf_for_depth)

    both = rgb_active & valid_left & valid_right
    left_only = rgb_active & valid_left & (~valid_right)
    right_only = rgb_active & valid_right & (~valid_left)
    fallback = rgb_active & (~valid_left) & (~valid_right)

    if fallback_mode == 'render_depth':
        target = render_depth.copy()
    elif fallback_mode == 'none':
        target = np.zeros_like(render_depth, dtype=np.float32)
    else:
        raise ValueError(f'Unsupported fallback_mode={fallback_mode}')

    source_map = np.zeros_like(render_depth, dtype=np.int16)

    if both_mode != 'weighted_by_fusion':
        raise ValueError(f'Unsupported both_mode={both_mode}')
    both_w_sum = fusion_weight_left + fusion_weight_right
    both_left = np.zeros_like(render_depth, dtype=np.float32)
    both_right = np.zeros_like(render_depth, dtype=np.float32)
    valid_both_weight = both & (both_w_sum > 1e-8)
    both_left[valid_both_weight] = fusion_weight_left[valid_both_weight] / both_w_sum[valid_both_weight]
    both_right[valid_both_weight] = fusion_weight_right[valid_both_weight] / both_w_sum[valid_both_weight]
    target[valid_both_weight] = (
        both_left[valid_both_weight] * projected_depth_left[valid_both_weight]
        + both_right[valid_both_weight] * projected_depth_right[valid_both_weight]
    )
    source_map[valid_both_weight] = SOURCE_BOTH_WEIGHTED

    if single_mode != 'single_branch_projected':
        raise ValueError(f'Unsupported single_mode={single_mode}')
    target[left_only] = projected_depth_left[left_only]
    target[right_only] = projected_depth_right[right_only]
    source_map[left_only] = SOURCE_LEFT
    source_map[right_only] = SOURCE_RIGHT

    if fallback_mode == 'render_depth':
        source_map[fallback] = SOURCE_RENDER_FALLBACK
    else:
        source_map[fallback] = SOURCE_NONE
        target[fallback] = 0.0

    depth_mask = rgb_active.astype(np.float32)
    verified_mask = (source_map == SOURCE_LEFT) | (source_map == SOURCE_RIGHT) | (source_map == SOURCE_BOTH_WEIGHTED)
    if use_continuous_reweight and raw_rgb_confidence_cont is not None:
        depth_mask_cont = depth_mask * np.clip(raw_rgb_confidence_cont, 0.0, 1.0)
    else:
        depth_mask_cont = depth_mask.copy()

    rel_correction = np.zeros_like(render_depth, dtype=np.float32)
    valid_render = render_depth > 1e-6
    corr_valid = verified_mask & valid_render & (target > 1e-6)
    rel_correction[corr_valid] = np.abs(target[corr_valid] - render_depth[corr_valid]) / np.maximum(render_depth[corr_valid], 1e-6)

    summary = {
        'mask_nonzero_ratio': float((depth_mask > 0).sum() / depth_mask.size),
        'mask_cont_nonzero_ratio': float((depth_mask_cont > 0).sum() / depth_mask_cont.size),
        'verified_ratio': float(verified_mask.sum() / verified_mask.size),
        'left_only_ratio': float(left_only.sum() / left_only.size),
        'right_only_ratio': float(right_only.sum() / right_only.size),
        'both_weighted_ratio': float(valid_both_weight.sum() / valid_both_weight.size),
        'render_fallback_ratio': float((source_map == SOURCE_RENDER_FALLBACK).sum() / source_map.size),
        'mean_abs_rel_correction_verified': float(rel_correction[verified_mask].mean()) if verified_mask.any() else 0.0,
        'source_counts': {
            'none': int((source_map == SOURCE_NONE).sum()),
            'left': int((source_map == SOURCE_LEFT).sum()),
            'right': int((source_map == SOURCE_RIGHT).sum()),
            'both_weighted': int((source_map == SOURCE_BOTH_WEIGHTED).sum()),
            'render_fallback': int((source_map == SOURCE_RENDER_FALLBACK).sum()),
        },
        'policy': {
            'min_rgb_conf_for_depth': float(min_rgb_conf_for_depth),
            'fallback_mode': fallback_mode,
            'both_mode': both_mode,
            'single_mode': single_mode,
            'use_continuous_reweight': bool(use_continuous_reweight),
        },
    }

    return {
        'target_depth_for_refine_v2_brpo': target.astype(np.float32),
        'target_depth_source_map_v2_brpo': source_map.astype(np.int16),
        'depth_supervision_mask_v2_brpo': depth_mask.astype(np.float32),
        'depth_supervision_mask_cont_v2_brpo': depth_mask_cont.astype(np.float32),
        'depth_verified_mask_v2_brpo': verified_mask.astype(np.float32),
        'depth_rel_correction_v2_brpo': rel_correction.astype(np.float32),
        'summary': summary,
    }


def write_depth_supervision_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / 'target_depth_for_refine_v2_brpo.npy', result['target_depth_for_refine_v2_brpo'])
    np.save(frame_out / 'target_depth_source_map_v2_brpo.npy', result['target_depth_source_map_v2_brpo'])
    np.save(frame_out / 'depth_supervision_mask_v2_brpo.npy', result['depth_supervision_mask_v2_brpo'])
    np.save(frame_out / 'depth_supervision_mask_cont_v2_brpo.npy', result['depth_supervision_mask_cont_v2_brpo'])
    np.save(frame_out / 'depth_verified_mask_v2_brpo.npy', result['depth_verified_mask_v2_brpo'])
    np.save(frame_out / 'depth_rel_correction_v2_brpo.npy', result['depth_rel_correction_v2_brpo'])

    _save_float_png(result['target_depth_for_refine_v2_brpo'], frame_out / 'target_depth_for_refine_v2_brpo.png')
    _save_mask_png(result['depth_supervision_mask_v2_brpo'], frame_out / 'depth_supervision_mask_v2_brpo.png')
    _save_float_png(result['depth_supervision_mask_cont_v2_brpo'], diag_dir / 'depth_supervision_mask_cont_v2_brpo.png', vmax=1.0)
    _save_mask_png(result['depth_verified_mask_v2_brpo'], diag_dir / 'depth_verified_mask_v2_brpo.png')
    _save_float_png(result['depth_rel_correction_v2_brpo'], diag_dir / 'depth_rel_correction_v2_brpo.png')
    _save_source_map_png(result['target_depth_source_map_v2_brpo'], frame_out / 'target_depth_source_map_v2_brpo.png')

    with open(frame_out / 'depth_meta_v2_brpo.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
