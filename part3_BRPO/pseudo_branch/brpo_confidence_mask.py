# -*- coding: utf-8 -*-
"""Utilities for BRPO-style support fusion / confidence mask generation."""
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


DEFAULT_MASK_VALUES = {
    "both": 1.0,
    "single": 0.5,
    "none": 0.0,
}


def build_brpo_confidence_mask(
    support_left: np.ndarray,
    support_right: np.ndarray,
    value_both: float = 1.0,
    value_single: float = 0.5,
    value_none: float = 0.0,
):
    left = support_left > 0.5
    right = support_right > 0.5
    both = left & right
    left_only = left & (~right)
    right_only = right & (~left)
    single = left ^ right
    neither = ~(left | right)

    conf_fused = np.zeros_like(support_left, dtype=np.float32)
    conf_fused[both] = value_both
    conf_fused[single] = value_single
    conf_fused[neither] = value_none

    conf_left = np.zeros_like(support_left, dtype=np.float32)
    conf_left[both] = value_both
    conf_left[left_only] = value_single
    conf_left[neither] = value_none

    conf_right = np.zeros_like(support_left, dtype=np.float32)
    conf_right[both] = value_both
    conf_right[right_only] = value_single
    conf_right[neither] = value_none

    return {
        "confidence_mask_brpo": conf_fused,
        "confidence_mask_brpo_fused": conf_fused,
        "confidence_mask_brpo_left": conf_left,
        "confidence_mask_brpo_right": conf_right,
        "support_both": both.astype(np.float32),
        "support_single": single.astype(np.float32),
        "support_left": left.astype(np.float32),
        "support_right": right.astype(np.float32),
    }


def save_mask_png(mask: np.ndarray, path: str):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def summarize_brpo_mask(frame_id: int, left_stats: Dict, right_stats: Dict, fused: Dict) -> Dict:
    conf = fused["confidence_mask_brpo_fused"]
    both = fused["support_both"]
    single = fused["support_single"]
    left = fused["support_left"]
    right = fused["support_right"]
    h, w = conf.shape
    total = float(h * w)

    return {
        "frame_id": int(frame_id),
        "num_support_left": int((left > 0).sum()),
        "num_support_right": int((right > 0).sum()),
        "num_support_both": int((both > 0).sum()),
        "num_support_single": int((single > 0).sum()),
        "support_ratio_left": float((left > 0).sum() / total),
        "support_ratio_right": float((right > 0).sum() / total),
        "support_ratio_both": float((both > 0).sum() / total),
        "support_ratio_single": float((single > 0).sum() / total),
        "mean_reproj_error_left": left_stats.get("mean_reproj_error"),
        "mean_reproj_error_right": right_stats.get("mean_reproj_error"),
        "mean_rel_depth_error_left": left_stats.get("mean_rel_depth_error"),
        "mean_rel_depth_error_right": right_stats.get("mean_rel_depth_error"),
        "left": left_stats,
        "right": right_stats,
    }


def write_frame_outputs(frame_out: Path, fused: Dict, meta: Dict):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / "confidence_mask_brpo.npy", fused["confidence_mask_brpo_fused"])
    np.save(frame_out / "confidence_mask_brpo_fused.npy", fused["confidence_mask_brpo_fused"])
    np.save(frame_out / "confidence_mask_brpo_left.npy", fused["confidence_mask_brpo_left"])
    np.save(frame_out / "confidence_mask_brpo_right.npy", fused["confidence_mask_brpo_right"])
    np.save(frame_out / "support_left.npy", fused["support_left"])
    np.save(frame_out / "support_right.npy", fused["support_right"])
    np.save(frame_out / "support_both.npy", fused["support_both"])
    np.save(frame_out / "support_single.npy", fused["support_single"])

    save_mask_png(fused["confidence_mask_brpo_fused"], str(frame_out / "confidence_mask_brpo.png"))
    save_mask_png(fused["confidence_mask_brpo_fused"], str(frame_out / "confidence_mask_brpo_fused.png"))
    save_mask_png(fused["confidence_mask_brpo_left"], str(frame_out / "confidence_mask_brpo_left.png"))
    save_mask_png(fused["confidence_mask_brpo_right"], str(frame_out / "confidence_mask_brpo_right.png"))
    save_mask_png(fused["support_left"], str(frame_out / "support_left.png"))
    save_mask_png(fused["support_right"], str(frame_out / "support_right.png"))
    save_mask_png(fused["support_both"], str(frame_out / "support_both.png"))
    save_mask_png(fused["support_single"], str(frame_out / "support_single.png"))

    with open(frame_out / "verification_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
