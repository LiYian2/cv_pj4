from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pseudo_branch.common import get_shared_mast3r_pair_forward


def _optional_load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def _compute_scale_anchor(
    raw_depth: np.ndarray,
    anchor_depth: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    anchor_depth = np.asarray(anchor_depth, dtype=np.float32)
    anchor_mask = np.asarray(anchor_mask, dtype=np.float32) > 0.5
    valid = anchor_mask & np.isfinite(raw_depth) & np.isfinite(anchor_depth) & (raw_depth > 1e-6) & (anchor_depth > 1e-6)

    scale_factor = 1.0
    if valid.any():
        ratios = anchor_depth[valid] / np.maximum(raw_depth[valid], 1e-6)
        positive = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
        if positive.size:
            scale_factor = float(np.median(positive))

    scaled_depth = np.where(np.isfinite(raw_depth) & (raw_depth > 1e-6), raw_depth * scale_factor, 0.0).astype(np.float32)
    rel_err = None
    if valid.any():
        rel_err_arr = np.abs(scaled_depth[valid] - anchor_depth[valid]) / np.maximum(anchor_depth[valid], 1e-6)
        rel_err = float(np.median(rel_err_arr)) if rel_err_arr.size else None

    return scaled_depth, {
        "scale_factor": float(scale_factor),
        "anchor_count": int(valid.sum()),
        "anchor_ratio": float(valid.mean()),
        "post_scale_anchor_relerr_median": rel_err,
    }


def _build_mast3r_direct_depth_branch(
    *,
    pseudo_gt_rgb_path: str,
    ref_rgb_path: str,
    anchor_depth: np.ndarray,
    anchor_mask: np.ndarray,
    cfg: RuntimeExactBackendConfig,
    pair_forwarder=None,
) -> dict[str, Any]:
    forwarder = pair_forwarder or get_shared_mast3r_pair_forward(
        model_name=str(cfg.matcher_model_name),
        device=str(cfg.matcher_device),
    )
    bundle = forwarder.run_pair(
        img1_path=str(pseudo_gt_rgb_path),
        img2_path=str(ref_rgb_path),
        size=int(anchor_depth.shape[1]),
    )
    pts3d_1 = np.asarray(bundle.pts3d_1, dtype=np.float32)
    conf1 = np.asarray(bundle.conf1, dtype=np.float32)
    raw_depth = np.where(np.isfinite(pts3d_1[..., 2]) & (pts3d_1[..., 2] > 1e-6), pts3d_1[..., 2], 0.0).astype(np.float32)
    scaled_depth, scale_meta = _compute_scale_anchor(raw_depth, anchor_depth, anchor_mask)
    return {
        "raw_depth": raw_depth,
        "scaled_depth": scaled_depth,
        "pseudo_confidence": conf1,
        "pair_meta": dict(bundle.meta),
        "scale_meta": scale_meta,
    }


