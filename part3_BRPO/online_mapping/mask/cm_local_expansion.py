# -*- coding: utf-8 -*-
"""C_m Controlled Local Expansion (r1/r2 soft) — 2026-05-08

Implements reciprocal seed + local RGB/depth-gated soft expansion:
- Preserve raw reciprocal seed as high-trust anchor
- Expand in local neighborhood (r=1 or r=2) under RGB/depth consistency gates
- Assign lower weights to expanded regions (expanded_both=0.6, expanded_single=0.25)
- Track provenance for each pixel (raw_seed, expanded_both, expanded_single, etc.)

All outputs are written as separate files to enable comparison without
overwriting raw reciprocal seed artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# Provenance labels for expansion tracking
PROVENANCE_RAW_BOTH = 0       # Raw reciprocal seed, both sides
PROVENANCE_RAW_SINGLE = 1     # Raw reciprocal seed, single side
PROVENANCE_EXPANDED_BOTH = 2  # Expanded pixel, both sides agree after expansion
PROVENANCE_RAW_PLUS_EXP_SINGLE = 3  # Raw seed + expanded from other side
PROVENANCE_EXPANDED_SINGLE = 4  # Expanded pixel, single side only
PROVENANCE_NONE = 5           # No support


def expand_branch_support_local(
    raw_support: np.ndarray,
    seed_confidence: np.ndarray,
    pseudo_rgb: np.ndarray,
    pseudo_depth: np.ndarray,
    *,
    radius: int = 1,
    expansion_weight: float = 0.5,
    tau_rgb_l1: float = 0.08,
    tau_depth_rel: float = 0.05,
    min_seed_conf: float = 0.0,
    min_expanded_conf: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Expand single branch support using local RGB/depth-gated expansion.

    For each raw seed pixel p, scan radius-r neighborhood candidates q.
    Candidate q is accepted if:
    1. seed_conf[p] >= min_seed_conf
    2. RGB continuous: mean_abs(rgb[q] - rgb[p]) <= tau_rgb_l1
    3. Depth continuous: abs(depth[q] - depth[p]) / max(depth) <= tau_depth_rel
    4. Depth valid: depth[p] > 1e-6 and depth[q] > 1e-6

    Args:
        raw_support: Binary mask of raw reciprocal seed (H, W), float32
        seed_confidence: Confidence map for seed pixels (H, W), float32
        pseudo_rgb: RGB image in [0,1] (H, W, 3), float32
        pseudo_depth: Depth map (H, W), float32
        radius: Expansion radius (1=3x3, 2=5x5)
        expansion_weight: Weight multiplier for expanded confidence
        tau_rgb_l1: Max mean RGB L1 distance for expansion
        tau_depth_rel: Max relative depth difference for expansion
        min_seed_conf: Minimum seed confidence to trigger expansion
        min_expanded_conf: Minimum confidence to accept expanded pixel

    Returns:
        Dict with:
            - raw_support: Original raw support mask
            - expanded_support: Raw + expanded mask (binary)
            - expanded_only: Expanded pixels only (binary)
            - expanded_confidence: Confidence map for expanded pixels
            - reject_reasons: Dict with reject counts by reason
    """
    raw_support = np.asarray(raw_support, dtype=np.float32)
    seed_conf = np.asarray(seed_confidence, dtype=np.float32)
    pseudo_rgb = np.asarray(pseudo_rgb, dtype=np.float32)
    pseudo_depth = np.asarray(pseudo_depth, dtype=np.float32)

    H, W = raw_support.shape
    if pseudo_rgb.shape[:2] != (H, W):
        raise ValueError(f"RGB shape {pseudo_rgb.shape[:2]} != support shape {(H, W)}")

    # Output maps
    expanded_support = raw_support.copy()
    expanded_only = np.zeros((H, W), dtype=np.float32)
    expanded_confidence = np.zeros((H, W), dtype=np.float32)

    # Reject reason tracking
    reject_reasons = {
        "low_seed_conf": 0,
        "rgb_fail": 0,
        "depth_fail": 0,
        "invalid_seed_depth": 0,
        "invalid_cand_depth": 0,
        "low_expanded_conf": 0,
        "already_expanded": 0,
    }

    # Precompute valid depth mask
    depth_valid = (pseudo_depth > 1e-6) & np.isfinite(pseudo_depth)

    # Find seed pixels
    seed_pixels = np.where(raw_support > 0.5)
    num_seeds = len(seed_pixels[0])

    # For each seed pixel, scan neighbors
    for i in range(num_seeds):
        py, px = seed_pixels[0][i], seed_pixels[1][i]
        seed_conf_val = seed_conf[py, px]

        # Check seed confidence threshold
        if seed_conf_val < min_seed_conf:
            reject_reasons["low_seed_conf"] += 1
            continue

        # Check seed depth validity
        if not depth_valid[py, px]:
            reject_reasons["invalid_seed_depth"] += 1
            continue

        seed_depth = pseudo_depth[py, px]
        seed_rgb = pseudo_rgb[py, px]

        # Scan radius-r neighborhood
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue  # Skip seed itself

                qy, qx = py + dy, px + dx
                # Boundary check
                if qy < 0 or qy >= H or qx < 0 or qx >= W:
                    continue

                # Skip if already in raw support
                if raw_support[qy, qx] > 0.5:
                    continue

                # If already expanded by another seed, still evaluate this candidate and
                # keep the maximum confidence.  Do not make expansion order-dependent.
                already_expanded = bool(expanded_only[qy, qx] > 0.5)
                if already_expanded:
                    reject_reasons["already_expanded"] += 1

                # Check candidate depth validity
                if not depth_valid[qy, qx]:
                    reject_reasons["invalid_cand_depth"] += 1
                    continue

                cand_depth = pseudo_depth[qy, qx]
                cand_rgb = pseudo_rgb[qy, qx]

                # RGB gate: mean L1 distance
                rgb_l1 = np.mean(np.abs(cand_rgb - seed_rgb))
                if rgb_l1 > tau_rgb_l1:
                    reject_reasons["rgb_fail"] += 1
                    continue

                # Depth gate: relative difference
                rel_depth = abs(cand_depth - seed_depth) / max(seed_depth, cand_depth, 1e-6)
                if rel_depth > tau_depth_rel:
                    reject_reasons["depth_fail"] += 1
                    continue

                # Passed all gates - compute expanded confidence
                spatial_dist = np.sqrt(dy**2 + dx**2)
                spatial_score = np.exp(-spatial_dist / max(radius, 1))
                rgb_score = max(0.0, 1.0 - rgb_l1 / tau_rgb_l1)
                depth_score = max(0.0, 1.0 - rel_depth / tau_depth_rel)

                exp_conf = seed_conf_val * spatial_score * rgb_score * depth_score * expansion_weight

                # Only accept if above threshold
                if exp_conf < min_expanded_conf:
                    reject_reasons["low_expanded_conf"] += 1
                    continue

                # Mark as expanded.  Multiple seeds may reach the same q; keep max confidence.
                expanded_only[qy, qx] = 1.0
                expanded_support[qy, qx] = 1.0
                expanded_confidence[qy, qx] = max(float(expanded_confidence[qy, qx]), float(exp_conf))

    # Compute summary stats
    raw_ratio = float(raw_support.sum() / (H * W))
    expanded_only_ratio = float(expanded_only.sum() / (H * W))
    total_expanded_ratio = float(expanded_support.sum() / (H * W))

    summary = {
        "raw_support_ratio": raw_ratio,
        "expanded_only_ratio": expanded_only_ratio,
        "total_expanded_ratio": total_expanded_ratio,
        "expansion_gain_ratio": expanded_only_ratio,
        "num_raw_seeds": int(raw_support.sum()),
        "num_expanded_only": int(expanded_only.sum()),
        "mean_expanded_conf": float(expanded_confidence[expanded_only > 0.5].mean()) if (expanded_only > 0.5).any() else 0.0,
        "reject_reasons": reject_reasons,
        "parameters": {
            "radius": radius,
            "expansion_weight": expansion_weight,
            "tau_rgb_l1": tau_rgb_l1,
            "tau_depth_rel": tau_depth_rel,
            "min_seed_conf": min_seed_conf,
            "min_expanded_conf": min_expanded_conf,
        },
    }

    return {
        "raw_support": raw_support,
        "expanded_support": expanded_support.astype(np.float32),
        "expanded_only": expanded_only.astype(np.float32),
        "expanded_confidence": expanded_confidence.astype(np.float32),
        "reject_reasons": reject_reasons,
        "summary": summary,
    }


def compose_soft_cm_from_expanded_branches(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    expanded_left: np.ndarray,
    expanded_right: np.ndarray,
    *,
    raw_both_weight: float = 1.0,
    raw_single_weight: float = 0.5,
    expanded_both_weight: float = 0.6,
    raw_exp_agree_weight: float = 0.5,
    expanded_single_weight: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Compose soft C_m from raw and expanded branch supports.

    Weight assignment rules:
    - raw_both (both raw seeds): 1.0 (paper-aligned)
    - raw_single (xor raw seeds): 0.5 (paper-aligned)
    - expanded_both: both expanded but not raw both -> 0.6
    - raw_plus_exp_single: raw seed + expanded from other side -> 0.5
    - expanded_single: expanded but single side -> 0.25 (weak supervision)

    Args:
        raw_left/right: Binary mask of raw reciprocal seed (H, W)
        expanded_left/right: Binary mask of raw + expanded (H, W)
        raw_both_weight: Weight for raw both pixels
        raw_single_weight: Weight for raw single pixels
        expanded_both_weight: Weight for expanded both pixels
        raw_exp_agree_weight: Weight for raw+exp agree pixels
        expanded_single_weight: Weight for expanded single pixels

    Returns:
        Dict with:
            - confidence_cm: Soft C_m weights (H, W), float32
            - provenance_map: Provenance labels (H, W), int32
            - support_left_raw: Raw left support
            - support_right_raw: Raw right support
            - support_left_expanded: Expanded left support
            - support_right_expanded: Expanded right support
            - summary: Statistics dict
    """
    raw_left = np.asarray(raw_left, dtype=np.float32) > 0.5
    raw_right = np.asarray(raw_right, dtype=np.float32) > 0.5
    exp_left = np.asarray(expanded_left, dtype=np.float32) > 0.5
    exp_right = np.asarray(expanded_right, dtype=np.float32) > 0.5

    H, W = raw_left.shape

    # Derive new expanded regions (excluding raw)
    new_left = exp_left & ~raw_left
    new_right = exp_right & ~raw_right

    # Categorize pixels by provenance
    raw_both = raw_left & raw_right
    raw_single = raw_left ^ raw_right  # xor

    # Expanded-only both: both branches expanded into q, and neither side was raw.
    # Keep this category mutually exclusive from raw_plus_exp_single for truthful stats.
    expanded_both = new_left & new_right

    # Raw + expanded agreement: one side raw, other side expanded into the same q.
    raw_plus_exp_single = (raw_left & new_right) | (raw_right & new_left)

    # Expanded single: only one side expanded, neither side raw.
    expanded_single = (new_left ^ new_right) & ~(raw_left | raw_right)

    # Build provenance map
    provenance_map = np.full((H, W), PROVENANCE_NONE, dtype=np.int32)
    provenance_map[raw_both] = PROVENANCE_RAW_BOTH
    provenance_map[raw_single & ~raw_both] = PROVENANCE_RAW_SINGLE
    provenance_map[expanded_both] = PROVENANCE_EXPANDED_BOTH
    provenance_map[raw_plus_exp_single] = PROVENANCE_RAW_PLUS_EXP_SINGLE
    provenance_map[expanded_single] = PROVENANCE_EXPANDED_SINGLE

    # Build soft C_m weights
    confidence_cm = np.zeros((H, W), dtype=np.float32)
    confidence_cm[raw_both] = raw_both_weight
    confidence_cm[raw_single & ~raw_both] = raw_single_weight
    confidence_cm[expanded_both] = np.maximum(confidence_cm[expanded_both], expanded_both_weight)
    confidence_cm[raw_plus_exp_single] = np.maximum(confidence_cm[raw_plus_exp_single], raw_exp_agree_weight)
    confidence_cm[expanded_single] = np.maximum(confidence_cm[expanded_single], expanded_single_weight)

    # Compute statistics
    total_pixels = H * W

    raw_both_ratio = float(raw_both.sum() / total_pixels)
    raw_single_ratio = float(raw_single.sum() / total_pixels)
    raw_union_ratio = float((raw_left | raw_right).sum() / total_pixels)

    expanded_both_ratio = float(expanded_both.sum() / total_pixels)
    raw_exp_agree_ratio = float(raw_plus_exp_single.sum() / total_pixels)
    expanded_single_ratio = float(expanded_single.sum() / total_pixels)

    expanded_union_ratio = float((exp_left | exp_right).sum() / total_pixels)
    nonzero_cm_ratio = float((confidence_cm > 0).sum() / total_pixels)

    # Effective mask weight = sum(C_m) / num_pixels
    # This measures the total "supervision mass" per pixel
    effective_weight_raw = raw_both_weight * raw_both.sum() + raw_single_weight * raw_single.sum()
    effective_weight_raw /= total_pixels

    effective_weight_expanded = confidence_cm.sum() / total_pixels
    weight_gain_ratio = effective_weight_expanded / max(effective_weight_raw, 1e-8)

    # Mean positive C_m before and after
    mean_positive_raw = (raw_both_weight * raw_both.sum() + raw_single_weight * raw_single.sum()) / max((raw_left | raw_right).sum(), 1)
    mean_positive_expanded = float(confidence_cm[confidence_cm > 0].mean()) if (confidence_cm > 0).any() else 0.0

    # Provenance distribution
    provenance_dist = {
        "raw_both": int(raw_both.sum()),
        "raw_single": int(raw_single.sum()),
        "expanded_both": int(expanded_both.sum()),
        "raw_plus_exp_single": int(raw_plus_exp_single.sum()),
        "expanded_single": int(expanded_single.sum()),
        "none": int((provenance_map == PROVENANCE_NONE).sum()),
    }

    summary = {
        "raw_cm_union_ratio": raw_union_ratio,
        "raw_cm_both_ratio": raw_both_ratio,
        "raw_cm_single_ratio": raw_single_ratio,
        "expanded_cm_nonzero_ratio": nonzero_cm_ratio,
        "expanded_union_ratio": expanded_union_ratio,
        "expanded_both_ratio": expanded_both_ratio,
        "raw_exp_agree_ratio": raw_exp_agree_ratio,
        "expanded_single_ratio": expanded_single_ratio,
        "effective_mask_weight_raw": float(effective_weight_raw),
        "effective_mask_weight_expanded": float(effective_weight_expanded),
        "weight_gain_ratio": float(weight_gain_ratio),
        "mean_positive_cm_raw": float(mean_positive_raw),
        "mean_positive_cm_expanded": float(mean_positive_expanded),
        "provenance_distribution": provenance_dist,
        "parameters": {
            "raw_both_weight": raw_both_weight,
            "raw_single_weight": raw_single_weight,
            "expanded_both_weight": expanded_both_weight,
            "raw_exp_agree_weight": raw_exp_agree_weight,
            "expanded_single_weight": expanded_single_weight,
        },
    }

    return {
        "confidence_cm": confidence_cm,
        "provenance_map": provenance_map,
        "support_left_raw": raw_left.astype(np.float32),
        "support_right_raw": raw_right.astype(np.float32),
        "support_left_expanded": exp_left.astype(np.float32),
        "support_right_expanded": exp_right.astype(np.float32),
        "support_left_expanded_only": new_left.astype(np.float32),
        "support_right_expanded_only": new_right.astype(np.float32),
        "summary": summary,
    }


def apply_cm_local_expansion(
    raw_support_left: np.ndarray,
    raw_support_right: np.ndarray,
    confidence_left: np.ndarray,
    confidence_right: np.ndarray,
    pseudo_rgb: np.ndarray,
    pseudo_depth: np.ndarray,
    *,
    radius: int = 1,
    expansion_weight: float = 0.5,
    tau_rgb_l1: float = 0.08,
    tau_depth_rel: float = 0.05,
    min_seed_conf: float = 0.0,
    min_expanded_conf: float = 0.05,
    raw_both_weight: float = 1.0,
    raw_single_weight: float = 0.5,
    expanded_both_weight: float = 0.6,
    raw_exp_agree_weight: float = 0.5,
    expanded_single_weight: float = 0.25,
) -> Dict:
    """End-to-end C_m local expansion for both branches.

    Steps:
    1. Expand left branch support
    2. Expand right branch support
    3. Compose soft C_m from expanded branches

    Args:
        raw_support_left/right: Raw reciprocal support masks (H, W)
        confidence_left/right: Raw confidence maps (H, W)
        pseudo_rgb: RGB image in [0,1] (H, W, 3)
        pseudo_depth: Depth map (H, W)
        radius: Expansion radius
        expansion_weight: Expansion weight multiplier
        tau_rgb_l1: RGB L1 threshold
        tau_depth_rel: Relative depth threshold
        min_seed_conf: Minimum seed confidence
        min_expanded_conf: Minimum expanded confidence
        raw_both/single_weight: Weights for raw seed pixels
        expanded_both/single_weight: Weights for expanded pixels

    Returns:
        Dict with all expansion results and final C_m
    """
    # Expand left branch
    left_exp = expand_branch_support_local(
        raw_support=raw_support_left,
        seed_confidence=confidence_left,
        pseudo_rgb=pseudo_rgb,
        pseudo_depth=pseudo_depth,
        radius=radius,
        expansion_weight=expansion_weight,
        tau_rgb_l1=tau_rgb_l1,
        tau_depth_rel=tau_depth_rel,
        min_seed_conf=min_seed_conf,
        min_expanded_conf=min_expanded_conf,
    )

    # Expand right branch
    right_exp = expand_branch_support_local(
        raw_support=raw_support_right,
        seed_confidence=confidence_right,
        pseudo_rgb=pseudo_rgb,
        pseudo_depth=pseudo_depth,
        radius=radius,
        expansion_weight=expansion_weight,
        tau_rgb_l1=tau_rgb_l1,
        tau_depth_rel=tau_depth_rel,
        min_seed_conf=min_seed_conf,
        min_expanded_conf=min_expanded_conf,
    )

    # Compose soft C_m
    cm_result = compose_soft_cm_from_expanded_branches(
        raw_left=raw_support_left,
        raw_right=raw_support_right,
        expanded_left=left_exp["expanded_support"],
        expanded_right=right_exp["expanded_support"],
        raw_both_weight=raw_both_weight,
        raw_single_weight=raw_single_weight,
        expanded_both_weight=expanded_both_weight,
        raw_exp_agree_weight=raw_exp_agree_weight,
        expanded_single_weight=expanded_single_weight,
    )

    # Combine summaries
    full_summary = {
        "left_expansion": left_exp["summary"],
        "right_expansion": right_exp["summary"],
        "cm_composition": cm_result["summary"],
        "total_parameters": {
            "radius": radius,
            "expansion_weight": expansion_weight,
            "tau_rgb_l1": tau_rgb_l1,
            "tau_depth_rel": tau_depth_rel,
            "min_seed_conf": min_seed_conf,
            "min_expanded_conf": min_expanded_conf,
            "raw_both_weight": raw_both_weight,
            "raw_single_weight": raw_single_weight,
            "expanded_both_weight": expanded_both_weight,
            "raw_exp_agree_weight": raw_exp_agree_weight,
            "expanded_single_weight": expanded_single_weight,
        },
    }

    return {
        "left_expansion": left_exp,
        "right_expansion": right_exp,
        "cm_composition": cm_result,
        "summary": full_summary,
    }


def write_cm_expansion_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
) -> None:
    """Write C_m expansion artifacts to disk.

    Writes to cm_local_expand_r{radius}_v{version}/ subdirectory.

    Files:
        - cm_raw.npy: Raw C_m (from raw support only)
        - cm_expanded_soft.npy: Expanded soft C_m
        - support_left_raw.npy, support_right_raw.npy
        - support_left_expanded.npy, support_right_expanded.npy
        - support_left_expanded_only.npy, support_right_expanded_only.npy
        - expansion_provenance.npy
        - summary.json
    """
    frame_out.mkdir(parents=True, exist_ok=True)

    cm_comp = result["cm_composition"]

    np.save(frame_out / "cm_raw.npy", _build_raw_cm(cm_comp["support_left_raw"], cm_comp["support_right_raw"]))
    np.save(frame_out / "cm_expanded_soft.npy", cm_comp["confidence_cm"])
    np.save(frame_out / "support_left_raw.npy", cm_comp["support_left_raw"])
    np.save(frame_out / "support_right_raw.npy", cm_comp["support_right_raw"])
    np.save(frame_out / "support_left_expanded.npy", cm_comp["support_left_expanded"])
    np.save(frame_out / "support_right_expanded.npy", cm_comp["support_right_expanded"])
    np.save(frame_out / "support_left_expanded_only.npy", cm_comp["support_left_expanded_only"])
    np.save(frame_out / "support_right_expanded_only.npy", cm_comp["support_right_expanded_only"])
    np.save(frame_out / "expansion_provenance.npy", cm_comp["provenance_map"])

    # PNG visualizations
    _save_mask_png(cm_comp["confidence_cm"], frame_out / "cm_expanded_soft.png")

    with open(frame_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _build_raw_cm(support_left: np.ndarray, support_right: np.ndarray) -> np.ndarray:
    """Build raw C_m from raw support (both=1.0, xor=0.5)."""
    left = support_left > 0.5
    right = support_right > 0.5
    both = left & right
    single = left ^ right
    cm = np.zeros_like(support_left, dtype=np.float32)
    cm[both] = 1.0
    cm[single] = 0.5
    return cm


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save float mask as PNG (0-255)."""
    from PIL import Image
    img = np.clip(mask * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# Unit test smoke
def _smoke_test() -> None:
    """Basic smoke test for expansion logic."""
    H, W = 64, 64

    # Create synthetic raw support: scattered seeds
    raw_left = np.zeros((H, W), dtype=np.float32)
    raw_right = np.zeros((H, W), dtype=np.float32)

    # Add some seed points
    raw_left[20, 20] = 1.0
    raw_left[20, 21] = 1.0
    raw_right[20, 20] = 1.0  # Both
    raw_right[30, 30] = 1.0  # Single

    # Confidence
    conf_left = np.where(raw_left > 0, 0.8, 0.0).astype(np.float32)
    conf_right = np.where(raw_right > 0, 0.9, 0.0).astype(np.float32)

    # RGB: uniform (should pass gate)
    rgb = np.ones((H, W, 3), dtype=np.float32) * 0.5

    # Depth: uniform (should pass gate)
    depth = np.ones((H, W), dtype=np.float32) * 5.0

    result = apply_cm_local_expansion(
        raw_support_left=raw_left,
        raw_support_right=raw_right,
        confidence_left=conf_left,
        confidence_right=conf_right,
        pseudo_rgb=rgb,
        pseudo_depth=depth,
        radius=1,
    )

    s = result["summary"]["cm_composition"]
    print(f"Raw union: {s['raw_cm_union_ratio']:.4f}")
    print(f"Expanded nonzero: {s['expanded_cm_nonzero_ratio']:.4f}")
    print(f"Weight gain: {s['weight_gain_ratio']:.4f}x")
    print(f"Expanded both: {s['expanded_both_ratio']:.4f}")
    print("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
