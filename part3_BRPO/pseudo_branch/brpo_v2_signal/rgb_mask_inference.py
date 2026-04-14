from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


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


def _accumulate_match_maps(
    image_shape: Tuple[int, int],
    pts_fused: np.ndarray,
    conf: np.ndarray,
) -> Dict[str, np.ndarray]:
    h, w = image_shape
    conf = np.asarray(conf, dtype=np.float32)
    if pts_fused.shape[0] == 0:
        return {
            'support_mask': np.zeros((h, w), dtype=np.float32),
            'conf_map': np.zeros((h, w), dtype=np.float32),
            'match_density': np.zeros((h, w), dtype=np.float32),
        }

    x = np.rint(pts_fused[:, 0]).astype(np.int32)
    y = np.rint(pts_fused[:, 1]).astype(np.int32)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    support = np.zeros((h, w), dtype=np.float32)
    conf_map = np.zeros((h, w), dtype=np.float32)
    density = np.zeros((h, w), dtype=np.float32)

    if valid.any():
        xv = x[valid]
        yv = y[valid]
        cv = conf[valid]
        scale = float(np.quantile(cv, 0.99)) if cv.size > 0 else 1.0
        scale = max(scale, 1e-8)
        cv_norm = np.clip(cv / scale, 0.0, 1.0).astype(np.float32)
        for xi, yi, ci in zip(xv, yv, cv_norm):
            support[yi, xi] = 1.0
            density[yi, xi] += 1.0
            if ci > conf_map[yi, xi]:
                conf_map[yi, xi] = ci

    return {
        'support_mask': support,
        'conf_map': conf_map,
        'match_density': density,
    }


def build_rgb_mask_from_correspondences(
    fused_rgb_path: str,
    left_ref_rgb_path: str,
    right_ref_rgb_path: str,
    matcher,
    size: int = 512,
    value_both: float = 1.0,
    value_single: float = 0.5,
    value_none: float = 0.0,
) -> Dict[str, np.ndarray | Dict | float | int]:
    fused_rgb = np.asarray(Image.open(fused_rgb_path).convert('RGB'))
    h, w = fused_rgb.shape[:2]

    pts_fused_left, pts_left_ref, conf_left = matcher.match_pair(fused_rgb_path, left_ref_rgb_path, size=size)
    pts_fused_right, pts_right_ref, conf_right = matcher.match_pair(fused_rgb_path, right_ref_rgb_path, size=size)

    left_maps = _accumulate_match_maps((h, w), pts_fused_left, conf_left)
    right_maps = _accumulate_match_maps((h, w), pts_fused_right, conf_right)

    support_left = left_maps['support_mask'] > 0.5
    support_right = right_maps['support_mask'] > 0.5
    support_both = support_left & support_right
    support_single = support_left ^ support_right
    support_none = ~(support_left | support_right)

    discrete = np.full((h, w), fill_value=float(value_none), dtype=np.float32)
    discrete[support_single] = float(value_single)
    discrete[support_both] = float(value_both)

    cont_left = left_maps['conf_map'].astype(np.float32)
    cont_right = right_maps['conf_map'].astype(np.float32)
    cont_fused = np.zeros((h, w), dtype=np.float32)
    if support_both.any():
        cont_fused[support_both] = np.sqrt(np.clip(cont_left[support_both] * cont_right[support_both], 0.0, 1.0)).astype(np.float32)
    left_only = support_left & (~support_right)
    right_only = support_right & (~support_left)
    cont_fused[left_only] = cont_left[left_only]
    cont_fused[right_only] = cont_right[right_only]

    summary = {
        'num_matches_left': int(pts_fused_left.shape[0]),
        'num_matches_right': int(pts_fused_right.shape[0]),
        'support_ratio_left': float(support_left.sum() / float(h * w)),
        'support_ratio_right': float(support_right.sum() / float(h * w)),
        'support_ratio_both': float(support_both.sum() / float(h * w)),
        'support_ratio_single': float(support_single.sum() / float(h * w)),
        'raw_rgb_confidence_nonzero_ratio': float((discrete > 0).sum() / float(h * w)),
        'raw_rgb_confidence_mean_positive': float(discrete[discrete > 0].mean()) if (discrete > 0).any() else 0.0,
        'raw_rgb_confidence_cont_nonzero_ratio': float((cont_fused > 0).sum() / float(h * w)),
        'raw_rgb_confidence_cont_mean_positive': float(cont_fused[cont_fused > 0].mean()) if (cont_fused > 0).any() else 0.0,
    }

    return {
        'support_left': support_left.astype(np.float32),
        'support_right': support_right.astype(np.float32),
        'support_both': support_both.astype(np.float32),
        'support_single': support_single.astype(np.float32),
        'raw_rgb_confidence_v2': discrete,
        'raw_rgb_confidence_cont_v2': cont_fused,
        'raw_rgb_confidence_left_cont_v2': cont_left,
        'raw_rgb_confidence_right_cont_v2': cont_right,
        'match_density_left': left_maps['match_density'].astype(np.float32),
        'match_density_right': right_maps['match_density'].astype(np.float32),
        'summary': summary,
        'matcher_meta': {
            'size': int(size),
            'fused_rgb_path': str(Path(fused_rgb_path)),
            'left_ref_rgb_path': str(Path(left_ref_rgb_path)),
            'right_ref_rgb_path': str(Path(right_ref_rgb_path)),
        },
    }


def write_rgb_mask_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    save_items = {
        'rgb_support_left_v2.npy': result['support_left'],
        'rgb_support_right_v2.npy': result['support_right'],
        'rgb_support_both_v2.npy': result['support_both'],
        'rgb_support_single_v2.npy': result['support_single'],
        'raw_rgb_confidence_v2.npy': result['raw_rgb_confidence_v2'],
        'raw_rgb_confidence_cont_v2.npy': result['raw_rgb_confidence_cont_v2'],
        'raw_rgb_confidence_left_cont_v2.npy': result['raw_rgb_confidence_left_cont_v2'],
        'raw_rgb_confidence_right_cont_v2.npy': result['raw_rgb_confidence_right_cont_v2'],
        'match_density_left_v2.npy': result['match_density_left'],
        'match_density_right_v2.npy': result['match_density_right'],
    }
    for name, arr in save_items.items():
        np.save(frame_out / name, arr)

    _save_mask_png(result['support_left'], frame_out / 'rgb_support_left_v2.png')
    _save_mask_png(result['support_right'], frame_out / 'rgb_support_right_v2.png')
    _save_mask_png(result['support_both'], frame_out / 'rgb_support_both_v2.png')
    _save_mask_png(result['support_single'], frame_out / 'rgb_support_single_v2.png')
    _save_mask_png(result['raw_rgb_confidence_v2'], frame_out / 'raw_rgb_confidence_v2.png')
    _save_float_png(result['raw_rgb_confidence_cont_v2'], frame_out / 'raw_rgb_confidence_cont_v2.png', vmax=1.0)
    _save_float_png(result['raw_rgb_confidence_left_cont_v2'], diag_dir / 'raw_rgb_confidence_left_cont_v2.png', vmax=1.0)
    _save_float_png(result['raw_rgb_confidence_right_cont_v2'], diag_dir / 'raw_rgb_confidence_right_cont_v2.png', vmax=1.0)
    _save_float_png(result['match_density_left'], diag_dir / 'match_density_left_v2.png')
    _save_float_png(result['match_density_right'], diag_dir / 'match_density_right_v2.png')

    with open(frame_out / 'rgb_mask_meta_v2.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
