from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def propagate_seed_mask(
    seed_mask: np.ndarray,
    pseudo_rgb: np.ndarray,
    render_depth: np.ndarray,
    max_radius_px: int = 5,
    tau_rel_depth: float = 0.03,
    tau_rgb_l1: float = 0.08,
) -> np.ndarray:
    seed = seed_mask > 0.5
    h, w = seed.shape
    if not seed.any():
        return np.zeros_like(seed_mask, dtype=np.float32)

    if pseudo_rgb.dtype != np.float32:
        pseudo_rgb = pseudo_rgb.astype(np.float32)
    if pseudo_rgb.max() > 1.5:
        pseudo_rgb = pseudo_rgb / 255.0

    depth = render_depth.astype(np.float32)
    active = seed.copy()
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for y, x in np.argwhere(seed):
        queue.append((int(y), int(x), 0))
        visited[y, x] = True

    while queue:
        y, x, d = queue.popleft()
        if d >= max_radius_px:
            continue
        center_depth = depth[y, x]
        if center_depth <= 1e-6:
            continue
        center_rgb = pseudo_rgb[y, x]
        for ny, nx in _neighbors8(y, x, h, w):
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            cand_depth = depth[ny, nx]
            if cand_depth <= 1e-6:
                continue
            rel_depth = abs(float(cand_depth - center_depth)) / max(float(center_depth), 1e-6)
            rgb_l1 = float(np.abs(pseudo_rgb[ny, nx] - center_rgb).mean())
            if rel_depth <= tau_rel_depth and rgb_l1 <= tau_rgb_l1:
                active[ny, nx] = True
                queue.append((ny, nx, d + 1))
    return active.astype(np.float32)


def build_train_confidence_masks(
    seed_left: np.ndarray,
    seed_right: np.ndarray,
    pseudo_rgb: np.ndarray,
    render_depth: np.ndarray,
    value_both: float = 1.0,
    value_single: float = 0.5,
    value_none: float = 0.0,
    max_radius_px: int = 5,
    tau_rel_depth: float = 0.03,
    tau_rgb_l1: float = 0.08,
):
    prop_left = propagate_seed_mask(seed_left, pseudo_rgb, render_depth, max_radius_px, tau_rel_depth, tau_rgb_l1) > 0.5
    prop_right = propagate_seed_mask(seed_right, pseudo_rgb, render_depth, max_radius_px, tau_rel_depth, tau_rgb_l1) > 0.5

    prop_both = prop_left & prop_right
    prop_single = prop_left ^ prop_right
    prop_left_only = prop_left & (~prop_right)
    prop_right_only = prop_right & (~prop_left)

    conf_fused = np.full(seed_left.shape, value_none, dtype=np.float32)
    conf_fused[prop_single] = value_single
    conf_fused[prop_both] = value_both

    conf_left = np.full(seed_left.shape, value_none, dtype=np.float32)
    conf_left[prop_left_only] = value_single
    conf_left[prop_both] = value_both

    conf_right = np.full(seed_left.shape, value_none, dtype=np.float32)
    conf_right[prop_right_only] = value_single
    conf_right[prop_both] = value_both

    total = float(seed_left.shape[0] * seed_left.shape[1])
    summary = {
        'seed_support_ratio_left': float((seed_left > 0.5).sum() / total),
        'seed_support_ratio_right': float((seed_right > 0.5).sum() / total),
        'train_support_ratio_left': float(prop_left.sum() / total),
        'train_support_ratio_right': float(prop_right.sum() / total),
        'train_support_ratio_both': float(prop_both.sum() / total),
        'train_support_ratio_single': float(prop_single.sum() / total),
        'train_confidence_nonzero_ratio_fused': float((conf_fused > 0).sum() / total),
        'propagation': {
            'max_radius_px': int(max_radius_px),
            'tau_rel_depth': float(tau_rel_depth),
            'tau_rgb_l1': float(tau_rgb_l1),
        },
    }

    return {
        'train_confidence_mask_brpo_fused': conf_fused,
        'train_confidence_mask_brpo_left': conf_left,
        'train_confidence_mask_brpo_right': conf_right,
        'train_support_left': prop_left.astype(np.float32),
        'train_support_right': prop_right.astype(np.float32),
        'train_support_both': prop_both.astype(np.float32),
        'train_support_single': prop_single.astype(np.float32),
        'summary': summary,
    }
