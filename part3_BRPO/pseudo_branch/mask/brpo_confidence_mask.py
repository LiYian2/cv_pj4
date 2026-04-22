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


def _safe_exp_weight(values: np.ndarray, valid_mask: np.ndarray, tau: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if tau <= 1e-8:
        out[valid_mask] = 1.0
        return out
    if valid_mask.any():
        vv = np.clip(values[valid_mask].astype(np.float32), 0.0, None)
        out[valid_mask] = np.exp(-vv / float(tau)).astype(np.float32)
    return out


def _continuous_branch_confidence(result: Dict, support_mask: np.ndarray, tau_reproj: float, tau_depth: float) -> np.ndarray:
    support = np.asarray(support_mask, dtype=np.float32) > 0.5
    reproj = np.asarray(result["reproj_error_map"], dtype=np.float32)
    rel_depth = np.asarray(result["rel_depth_error_map"], dtype=np.float32)
    valid = support & np.isfinite(reproj) & np.isfinite(rel_depth)
    w_reproj = _safe_exp_weight(reproj, valid, tau_reproj)
    w_depth = _safe_exp_weight(rel_depth, valid, tau_depth)
    conf = np.zeros_like(reproj, dtype=np.float32)
    conf[valid] = np.sqrt(w_reproj[valid] * w_depth[valid]).astype(np.float32)
    return conf


def _agreement_from_projected_depth(
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    both_mask: np.ndarray,
    tau_agree: float,
) -> np.ndarray:
    both = np.asarray(both_mask, dtype=np.float32) > 0.5
    d_left = np.asarray(projected_depth_left, dtype=np.float32)
    d_right = np.asarray(projected_depth_right, dtype=np.float32)
    valid = both & (d_left > 1e-6) & (d_right > 1e-6) & np.isfinite(d_left) & np.isfinite(d_right)
    agree = np.zeros_like(d_left, dtype=np.float32)
    if valid.any():
        diff = np.abs(np.log(np.clip(d_left[valid], 1e-6, None)) - np.log(np.clip(d_right[valid], 1e-6, None)))
        agree[valid] = np.exp(-diff / max(float(tau_agree), 1e-8)).astype(np.float32)
    return agree


def build_brpo_confidence_mask(
    support_left: np.ndarray,
    support_right: np.ndarray,
    value_both: float = 1.0,
    value_single: float = 0.5,
    value_none: float = 0.0,
    left_result: Dict | None = None,
    right_result: Dict | None = None,
    tau_reproj: float = 4.0,
    tau_depth: float = 0.15,
    tau_agree: float = 0.10,
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

    cont_left = None
    cont_right = None
    cont_fused = None
    agreement = None
    if left_result is not None and right_result is not None:
        cont_left = _continuous_branch_confidence(left_result, support_left, tau_reproj=tau_reproj, tau_depth=tau_depth)
        cont_right = _continuous_branch_confidence(right_result, support_right, tau_reproj=tau_reproj, tau_depth=tau_depth)
        agreement = _agreement_from_projected_depth(
            left_result["projected_depth_map"],
            right_result["projected_depth_map"],
            both.astype(np.float32),
            tau_agree=tau_agree,
        )
        cont_fused = np.zeros_like(conf_fused, dtype=np.float32)
        both_conf = np.sqrt(np.clip(cont_left * cont_right, 0.0, None)).astype(np.float32) * agreement.astype(np.float32)
        cont_fused[both] = both_conf[both]
        cont_fused[left_only] = cont_left[left_only]
        cont_fused[right_only] = cont_right[right_only]
        cont_left = np.where(both, cont_left * agreement, cont_left).astype(np.float32)
        cont_right = np.where(both, cont_right * agreement, cont_right).astype(np.float32)
    else:
        agreement = np.zeros_like(conf_fused, dtype=np.float32)

    out = {
        "confidence_mask_brpo": conf_fused,
        "confidence_mask_brpo_fused": conf_fused,
        "confidence_mask_brpo_left": conf_left,
        "confidence_mask_brpo_right": conf_right,
        "support_both": both.astype(np.float32),
        "support_single": single.astype(np.float32),
        "support_left": left.astype(np.float32),
        "support_right": right.astype(np.float32),
        "confidence_mask_brpo_agreement": agreement.astype(np.float32),
    }
    if cont_fused is not None:
        out.update({
            "confidence_mask_brpo_cont_fused": cont_fused.astype(np.float32),
            "confidence_mask_brpo_cont_left": cont_left.astype(np.float32),
            "confidence_mask_brpo_cont_right": cont_right.astype(np.float32),
        })
    return out


def save_mask_png(mask: np.ndarray, path: str):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _symlink_force(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.name)


def _mask_summary(mask: np.ndarray) -> Dict[str, float]:
    positive = mask[mask > 0]
    return {
        "nonzero_ratio": float((mask > 0).sum() / mask.size),
        "mean_positive": float(positive.mean()) if positive.size > 0 else 0.0,
        "p50_positive": float(np.quantile(positive, 0.5)) if positive.size > 0 else 0.0,
        "p90_positive": float(np.quantile(positive, 0.9)) if positive.size > 0 else 0.0,
    }


def summarize_brpo_mask(frame_id: int, left_stats: Dict, right_stats: Dict, fused: Dict) -> Dict:
    conf = fused["confidence_mask_brpo_fused"]
    both = fused["support_both"]
    single = fused["support_single"]
    left = fused["support_left"]
    right = fused["support_right"]
    h, w = conf.shape
    total = float(h * w)
    out = {
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
        "discrete_confidence_summary": _mask_summary(conf),
        "left": left_stats,
        "right": right_stats,
    }
    if "confidence_mask_brpo_cont_fused" in fused:
        out["continuous_confidence_summary"] = _mask_summary(fused["confidence_mask_brpo_cont_fused"])
        out["agreement_summary"] = _mask_summary(fused["confidence_mask_brpo_agreement"])
    return out


def _save_optional_mask(frame_out: Path, fused: Dict, key: str, stem: str):
    if key not in fused:
        return
    np.save(frame_out / f"{stem}.npy", fused[key])
    save_mask_png(fused[key], str(frame_out / f"{stem}.png"))


def write_frame_outputs(frame_out: Path, fused: Dict, meta: Dict, train_masks: Dict | None = None):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    raw_discrete_fused = fused["confidence_mask_brpo_fused"]
    raw_discrete_left = fused["confidence_mask_brpo_left"]
    raw_discrete_right = fused["confidence_mask_brpo_right"]

    np.save(frame_out / "raw_confidence_mask_brpo_fused.npy", raw_discrete_fused)
    np.save(frame_out / "raw_confidence_mask_brpo_left.npy", raw_discrete_left)
    np.save(frame_out / "raw_confidence_mask_brpo_right.npy", raw_discrete_right)
    save_mask_png(raw_discrete_fused, str(frame_out / "raw_confidence_mask_brpo_fused.png"))
    save_mask_png(raw_discrete_left, str(frame_out / "raw_confidence_mask_brpo_left.png"))
    save_mask_png(raw_discrete_right, str(frame_out / "raw_confidence_mask_brpo_right.png"))

    _save_optional_mask(frame_out, fused, "confidence_mask_brpo_cont_fused", "raw_confidence_mask_brpo_cont_fused")
    _save_optional_mask(frame_out, fused, "confidence_mask_brpo_cont_left", "raw_confidence_mask_brpo_cont_left")
    _save_optional_mask(frame_out, fused, "confidence_mask_brpo_cont_right", "raw_confidence_mask_brpo_cont_right")
    _save_optional_mask(frame_out, fused, "confidence_mask_brpo_agreement", "confidence_mask_brpo_agreement")

    active_conf_fused = train_masks["train_confidence_mask_brpo_fused"] if train_masks is not None else raw_discrete_fused
    active_conf_left = train_masks["train_confidence_mask_brpo_left"] if train_masks is not None else raw_discrete_left
    active_conf_right = train_masks["train_confidence_mask_brpo_right"] if train_masks is not None else raw_discrete_right

    if train_masks is not None:
        np.save(frame_out / "train_confidence_mask_brpo_fused.npy", train_masks["train_confidence_mask_brpo_fused"])
        np.save(frame_out / "train_confidence_mask_brpo_left.npy", train_masks["train_confidence_mask_brpo_left"])
        np.save(frame_out / "train_confidence_mask_brpo_right.npy", train_masks["train_confidence_mask_brpo_right"])
        np.save(frame_out / "train_support_left.npy", train_masks["train_support_left"])
        np.save(frame_out / "train_support_right.npy", train_masks["train_support_right"])
        np.save(frame_out / "train_support_both.npy", train_masks["train_support_both"])
        np.save(frame_out / "train_support_single.npy", train_masks["train_support_single"])

        _symlink_force(frame_out / "train_confidence_mask_brpo_fused.npy", frame_out / "confidence_mask_brpo.npy")
        _symlink_force(frame_out / "train_confidence_mask_brpo_fused.npy", frame_out / "confidence_mask_brpo_fused.npy")
        _symlink_force(frame_out / "train_confidence_mask_brpo_left.npy", frame_out / "confidence_mask_brpo_left.npy")
        _symlink_force(frame_out / "train_confidence_mask_brpo_right.npy", frame_out / "confidence_mask_brpo_right.npy")
    else:
        np.save(frame_out / "confidence_mask_brpo.npy", active_conf_fused)
        np.save(frame_out / "confidence_mask_brpo_fused.npy", active_conf_fused)
        np.save(frame_out / "confidence_mask_brpo_left.npy", active_conf_left)
        np.save(frame_out / "confidence_mask_brpo_right.npy", active_conf_right)

    np.save(frame_out / "seed_support_left.npy", fused["support_left"])
    np.save(frame_out / "seed_support_right.npy", fused["support_right"])
    np.save(frame_out / "seed_support_both.npy", fused["support_both"])
    np.save(frame_out / "seed_support_single.npy", fused["support_single"])

    _symlink_force(frame_out / "seed_support_left.npy", frame_out / "support_left.npy")
    _symlink_force(frame_out / "seed_support_right.npy", frame_out / "support_right.npy")
    _symlink_force(frame_out / "seed_support_both.npy", frame_out / "support_both.npy")
    _symlink_force(frame_out / "seed_support_single.npy", frame_out / "support_single.npy")

    if train_masks is not None:
        save_mask_png(train_masks["train_confidence_mask_brpo_fused"], str(frame_out / "train_confidence_mask_brpo_fused.png"))
        save_mask_png(train_masks["train_confidence_mask_brpo_left"], str(frame_out / "train_confidence_mask_brpo_left.png"))
        save_mask_png(train_masks["train_confidence_mask_brpo_right"], str(frame_out / "train_confidence_mask_brpo_right.png"))
        save_mask_png(train_masks["train_support_left"], str(frame_out / "train_support_left.png"))
        save_mask_png(train_masks["train_support_right"], str(frame_out / "train_support_right.png"))
        save_mask_png(train_masks["train_support_both"], str(frame_out / "train_support_both.png"))
        save_mask_png(train_masks["train_support_single"], str(frame_out / "train_support_single.png"))
        _symlink_force(frame_out / "train_confidence_mask_brpo_fused.png", frame_out / "confidence_mask_brpo.png")
        _symlink_force(frame_out / "train_confidence_mask_brpo_fused.png", frame_out / "confidence_mask_brpo_fused.png")
        _symlink_force(frame_out / "train_confidence_mask_brpo_left.png", frame_out / "confidence_mask_brpo_left.png")
        _symlink_force(frame_out / "train_confidence_mask_brpo_right.png", frame_out / "confidence_mask_brpo_right.png")
    else:
        save_mask_png(active_conf_fused, str(frame_out / "confidence_mask_brpo.png"))
        save_mask_png(active_conf_fused, str(frame_out / "confidence_mask_brpo_fused.png"))
        save_mask_png(active_conf_left, str(frame_out / "confidence_mask_brpo_left.png"))
        save_mask_png(active_conf_right, str(frame_out / "confidence_mask_brpo_right.png"))

    save_mask_png(fused["support_left"], str(frame_out / "seed_support_left.png"))
    save_mask_png(fused["support_right"], str(frame_out / "seed_support_right.png"))
    save_mask_png(fused["support_both"], str(frame_out / "seed_support_both.png"))
    save_mask_png(fused["support_single"], str(frame_out / "seed_support_single.png"))
    _symlink_force(frame_out / "seed_support_left.png", frame_out / "support_left.png")
    _symlink_force(frame_out / "seed_support_right.png", frame_out / "support_right.png")
    _symlink_force(frame_out / "seed_support_both.png", frame_out / "support_both.png")
    _symlink_force(frame_out / "seed_support_single.png", frame_out / "support_single.png")

    with open(frame_out / "verification_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
