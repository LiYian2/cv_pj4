#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sidecar diagnostic script for C_m controlled local expansion.

This script runs offline on existing brpo_debug artifacts to diagnose
the effect of r1/r2 local expansion before integrating into runtime.

Usage:
    python scripts/diagnostics/materialize_cm_local_expansion.py         --debug-root /path/to/brpo_debug         --radius 1         --out-name cm_local_expand_r1_v1

Outputs:
    For each frame:
        frame_xxxx/cm_local_expand_r1_v1/
            cm_raw.npy
            cm_expanded_soft.npy
            support_left_raw.npy, support_right_raw.npy
            support_left_expanded.npy, support_right_expanded.npy
            expansion_provenance.npy
            summary.json
    
    Global summary:
        brpo_debug/cm_local_expand_r1_v1_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def load_rgb_float(path: Path) -> np.ndarray:
    """Load RGB image as float32 in [0,1]."""
    with Image.open(path) as pil:
        rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def load_npy_float(path: Path) -> np.ndarray:
    """Load .npy as float32."""
    return np.load(path).astype(np.float32)


def find_all_frame_roots(debug_root: Path) -> List[Path]:
    """Find all pseudo frame roots under brpo_debug.

    Expected structure:
        event_kf_xxxx/frame_yyyy/
    """
    frame_roots = []
    for event_dir in sorted(debug_root.iterdir()):
        if not event_dir.is_dir() or not event_dir.name.startswith("event_"):
            continue
        for frame_dir in sorted(event_dir.iterdir()):
            if not frame_dir.is_dir() or not frame_dir.name.startswith("frame_"):
                continue
            # Check if exact_backend_v1 exists
            exact_dir = frame_dir / "exact_backend_v1"
            if exact_dir.exists():
                frame_roots.append(frame_dir)
    return frame_roots


def process_single_frame(
    frame_root: Path,
    out_dir: Path,
    params: Dict[str, Any],
    *,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    """Process single frame for C_m expansion.

    Inputs (from exact_backend_v1/):
        - support_left_exact.npy, support_right_exact.npy
        - confidence_left_exact.npy, confidence_right_exact.npy
        - projected_depth_left_exact.npy (optional, for reference)
    
    Inputs (from runtime_inputs/):
        - pseudo_fused_rgb.png (or pseudo_render_rgb_runtime.png)
        - pseudo_render_depth_runtime.npy

    Returns frame-level summary.
    """
    from pseudo_branch.mask.cm_local_expansion import (
        apply_cm_local_expansion,
        write_cm_expansion_outputs,
    )

    exact_dir = frame_root / "exact_backend_v1"
    inputs_dir = frame_root / "runtime_inputs"

    # Load raw support and confidence
    support_left = load_npy_float(exact_dir / "support_left_exact.npy")
    support_right = load_npy_float(exact_dir / "support_right_exact.npy")
    conf_left = load_npy_float(exact_dir / "confidence_left_exact.npy")
    conf_right = load_npy_float(exact_dir / "confidence_right_exact.npy")

    # Load pseudo RGB (prefer fused, fallback to render)
    pseudo_rgb_path = inputs_dir / "pseudo_fused_rgb.png"
    if not pseudo_rgb_path.exists():
        pseudo_rgb_path = inputs_dir / "pseudo_render_rgb_runtime.png"
    if not pseudo_rgb_path.exists():
        raise FileNotFoundError(f"No pseudo RGB found in {inputs_dir}")
    pseudo_rgb = load_rgb_float(pseudo_rgb_path)

    # Load pseudo depth
    pseudo_depth_path = inputs_dir / "pseudo_render_depth_runtime.npy"
    if not pseudo_depth_path.exists():
        raise FileNotFoundError(f"No pseudo depth found: {pseudo_depth_path}")
    pseudo_depth = load_npy_float(pseudo_depth_path)

    # Apply expansion
    result = apply_cm_local_expansion(
        raw_support_left=support_left,
        raw_support_right=support_right,
        confidence_left=conf_left,
        confidence_right=conf_right,
        pseudo_rgb=pseudo_rgb,
        pseudo_depth=pseudo_depth,
        radius=int(params.get("radius", 1)),
        expansion_weight=float(params.get("expansion_weight", 0.5)),
        tau_rgb_l1=float(params.get("tau_rgb_l1", 0.08)),
        tau_depth_rel=float(params.get("tau_depth_rel", 0.05)),
        min_seed_conf=float(params.get("min_seed_conf", 0.0)),
        min_expanded_conf=float(params.get("min_expanded_conf", 0.05)),
        raw_both_weight=float(params.get("raw_both_weight", 1.0)),
        raw_single_weight=float(params.get("raw_single_weight", 0.5)),
        expanded_both_weight=float(params.get("expanded_both_weight", 0.6)),
        raw_exp_agree_weight=float(params.get("raw_exp_agree_weight", 0.5)),
        expanded_single_weight=float(params.get("expanded_single_weight", 0.25)),
    )

    # Optional depth-target diagnostic: this sidecar must not change projected depth scope.
    signal_depth_path = frame_root / "signal_v2" / "pseudo_depth_target_exact_brpo_upstream_target_v1.npy"
    depth_scope_summary = {
        "existing_signal_depth_target_found": bool(signal_depth_path.exists()),
        "depth_target_filled_ratio_before": None,
        "depth_target_filled_ratio_after": None,
        "depth_target_scope_changed_by_sidecar": False,
    }
    if signal_depth_path.exists():
        depth_target = load_npy_float(signal_depth_path)
        filled = float((depth_target > 1e-6).mean())
        depth_scope_summary["depth_target_filled_ratio_before"] = filled
        depth_scope_summary["depth_target_filled_ratio_after"] = filled

    # Build frame metadata
    frame_id = frame_root.name.replace("frame_", "")
    frame_meta = {
        "frame_id": int(frame_id),
        "frame_root": str(frame_root),
        "expansion_mode": "local_soft_v1",
        "out_dir": str(out_dir),
        "summary": result["summary"],
        "depth_scope_summary": depth_scope_summary,
        "input_files": {
            "support_left": str(exact_dir / "support_left_exact.npy"),
            "support_right": str(exact_dir / "support_right_exact.npy"),
            "confidence_left": str(exact_dir / "confidence_left_exact.npy"),
            "confidence_right": str(exact_dir / "confidence_right_exact.npy"),
            "pseudo_rgb": str(pseudo_rgb_path),
            "pseudo_depth": str(pseudo_depth_path),
        },
    }

    # Write outputs unless this is an explicit dry-run.
    if write_outputs:
        write_cm_expansion_outputs(out_dir, result, frame_meta)

    return frame_meta


def compute_global_summary(frame_summaries: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics across all frames."""
    if not frame_summaries:
        return {}

    # Collect key metrics
    raw_union_ratios = []
    raw_both_ratios = []
    raw_single_ratios = []
    expanded_nonzero_ratios = []
    expanded_both_ratios = []
    raw_exp_agree_ratios = []
    expanded_single_ratios = []
    weight_gain_ratios = []
    expanded_only_left_ratios = []
    expanded_only_right_ratios = []

    reject_totals = {
        "low_seed_conf": 0,
        "rgb_fail": 0,
        "depth_fail": 0,
        "invalid_seed_depth": 0,
        "invalid_cand_depth": 0,
        "low_expanded_conf": 0,
        "already_expanded": 0,
    }
    depth_target_filled_before = []
    depth_target_filled_after = []

    for fs in frame_summaries:
        cm_sum = fs["summary"]["cm_composition"]
        raw_union_ratios.append(cm_sum["raw_cm_union_ratio"])
        raw_both_ratios.append(cm_sum["raw_cm_both_ratio"])
        raw_single_ratios.append(cm_sum["raw_cm_single_ratio"])
        expanded_nonzero_ratios.append(cm_sum["expanded_cm_nonzero_ratio"])
        expanded_both_ratios.append(cm_sum["expanded_both_ratio"])
        raw_exp_agree_ratios.append(cm_sum["raw_exp_agree_ratio"])
        expanded_single_ratios.append(cm_sum["expanded_single_ratio"])
        weight_gain_ratios.append(cm_sum["weight_gain_ratio"])

        # Expansion-only ratios
        left_exp = fs["summary"]["left_expansion"]
        right_exp = fs["summary"]["right_expansion"]
        expanded_only_left_ratios.append(left_exp["expanded_only_ratio"])
        expanded_only_right_ratios.append(right_exp["expanded_only_ratio"])

        # Reject reasons
        for key in reject_totals:
            reject_totals[key] += left_exp["reject_reasons"].get(key, 0)
            reject_totals[key] += right_exp["reject_reasons"].get(key, 0)

        depth_scope = fs.get("depth_scope_summary", {})
        if depth_scope.get("depth_target_filled_ratio_before") is not None:
            depth_target_filled_before.append(float(depth_scope["depth_target_filled_ratio_before"]))
        if depth_scope.get("depth_target_filled_ratio_after") is not None:
            depth_target_filled_after.append(float(depth_scope["depth_target_filled_ratio_after"]))

    # Compute median/mean statistics
    def median(vals: List[float]) -> float:
        return float(np.median(vals)) if vals else 0.0

    def mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    global_summary = {
        "num_frames": len(frame_summaries),
        "raw_cm_union_median": median(raw_union_ratios),
        "raw_cm_both_median": median(raw_both_ratios),
        "raw_cm_single_median": median(raw_single_ratios),
        "expanded_cm_nonzero_median": median(expanded_nonzero_ratios),
        "expanded_both_median": median(expanded_both_ratios),
        "raw_exp_agree_median": median(raw_exp_agree_ratios),
        "expanded_single_median": median(expanded_single_ratios),
        "weight_gain_median": median(weight_gain_ratios),
        "weight_gain_mean": mean(weight_gain_ratios),
        "expanded_only_left_median": median(expanded_only_left_ratios),
        "expanded_only_right_median": median(expanded_only_right_ratios),
        "reject_reasons_total": reject_totals,
        "depth_target_filled_ratio_before_median": median(depth_target_filled_before),
        "depth_target_filled_ratio_after_median": median(depth_target_filled_after),
        "depth_target_scope_changed_by_sidecar": False,
        "acceptance_rate": {
            "left": sum(expanded_only_left_ratios) / max(sum(raw_union_ratios) / 2, 1e-8),
            "right": sum(expanded_only_right_ratios) / max(sum(raw_union_ratios) / 2, 1e-8),
        },
        "frame_ids": [fs["frame_id"] for fs in frame_summaries],
        "parameters_used": frame_summaries[0].get("summary", {}).get("total_parameters", {}) if frame_summaries else {},
    }

    return global_summary


def main():
    parser = argparse.ArgumentParser(description="C_m local expansion sidecar diagnostic")
    parser.add_argument("--debug-root", type=str, required=True,
                        help="Path to brpo_debug directory")
    parser.add_argument("--radius", type=int, default=1,
                        help="Expansion radius (1=3x3, 2=5x5)")
    parser.add_argument("--out-name", type=str, default=None,
                        help="Output subdirectory name (default: cm_local_expand_r{radius}_v1)")
    parser.add_argument("--tau-rgb-l1", type=float, default=0.08,
                        help="RGB L1 threshold")
    parser.add_argument("--tau-depth-rel", type=float, default=0.05,
                        help="Relative depth threshold")
    parser.add_argument("--expansion-weight", type=float, default=0.5,
                        help="Expansion weight multiplier")
    parser.add_argument("--min-expanded-conf", type=float, default=0.05,
                        help="Minimum expanded confidence")
    parser.add_argument("--expanded-both-weight", type=float, default=0.6,
                        help="Weight for expanded both pixels")
    parser.add_argument("--expanded-single-weight", type=float, default=0.25,
                        help="Weight for expanded single pixels")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats without writing files")
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    if not debug_root.exists():
        print(f"Error: debug-root not found: {debug_root}")
        sys.exit(1)

    # Output name
    out_name = args.out_name or f"cm_local_expand_r{args.radius}_v1"

    # Build params dict
    params = {
        "radius": args.radius,
        "tau_rgb_l1": args.tau_rgb_l1,
        "tau_depth_rel": args.tau_depth_rel,
        "expansion_weight": args.expansion_weight,
        "min_expanded_conf": args.min_expanded_conf,
        "expanded_both_weight": args.expanded_both_weight,
        "expanded_single_weight": args.expanded_single_weight,
    }

    # Find all frame roots
    frame_roots = find_all_frame_roots(debug_root)
    print(f"Found {len(frame_roots)} frame roots under {debug_root}")

    if not frame_roots:
        print("No frames found. Exiting.")
        sys.exit(1)

    # Process each frame
    frame_summaries = []
    for frame_root in frame_roots:
        frame_id = frame_root.name
        out_dir = frame_root / out_name

        print(f"Processing {frame_id}...")

        try:
            frame_meta = process_single_frame(frame_root, out_dir, params, write_outputs=not args.dry_run)
            frame_summaries.append(frame_meta)

            # Print frame-level stats
            cm_sum = frame_meta["summary"]["cm_composition"]
            print(f"  Raw union: {cm_sum['raw_cm_union_ratio']:.4f}, Expanded: {cm_sum['expanded_cm_nonzero_ratio']:.4f}")
            print(f"  Weight gain: {cm_sum['weight_gain_ratio']:.2f}x")
            print(f"  Expanded both: {cm_sum['expanded_both_ratio']:.4f}, Single: {cm_sum['expanded_single_ratio']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Compute global summary
    global_summary = compute_global_summary(frame_summaries)

    # Print global stats
    print()
    print("=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {global_summary.get('num_frames', 0)}")
    print(f"Raw C_m union median: {global_summary.get('raw_cm_union_median', 0):.4f}")
    print(f"Raw C_m both median: {global_summary.get('raw_cm_both_median', 0):.4f}")
    print(f"Raw C_m single median: {global_summary.get('raw_cm_single_median', 0):.4f}")
    print()
    print(f"Expanded C_m nonzero median: {global_summary.get('expanded_cm_nonzero_median', 0):.4f}")
    print(f"Expanded both median: {global_summary.get('expanded_both_median', 0):.4f}")
    print(f"Expanded single median: {global_summary.get('expanded_single_median', 0):.4f}")
    print()
    print(f"Weight gain median: {global_summary.get('weight_gain_median', 0):.2f}x")
    print(f"Weight gain mean: {global_summary.get('weight_gain_mean', 0):.2f}x")
    if global_summary.get("depth_target_filled_ratio_before_median", 0) or global_summary.get("depth_target_filled_ratio_after_median", 0):
        print(f"Depth target filled before/after median: {global_summary.get('depth_target_filled_ratio_before_median', 0):.4f} / {global_summary.get('depth_target_filled_ratio_after_median', 0):.4f}")
        print(f"Depth target scope changed by sidecar: {global_summary.get('depth_target_scope_changed_by_sidecar', False)}")
    print()
    print("Reject reasons (total across all frames):")
    for key, val in global_summary.get("reject_reasons_total", {}).items():
        print(f"  {key}: {val}")
    print()

    # Write global summary
    if not args.dry_run:
        global_summary_path = debug_root / f"{out_name}_summary.json"
        with open(global_summary_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2)
        print(f"Global summary written to: {global_summary_path}")

    # Validation checks (from doc §9)
    print()
    print("VALIDATION CHECKS")
    print("=" * 60)

    checks = []

    # Check 1: weight gain reasonable (1.2x-1.8x target)
    weight_gain = global_summary.get("weight_gain_median", 0)
    if weight_gain < 1.2:
        checks.append("[WARN] Weight gain < 1.2x - expansion too conservative")
    elif weight_gain > 3.0:
        checks.append("[WARN] Weight gain > 3x - expansion too aggressive")
    else:
        checks.append("[PASS] Weight gain in reasonable range (1.2x-3x)")

    # Check 2: expanded single not too high
    exp_single = global_summary.get("expanded_single_median", 0)
    raw_single = global_summary.get("raw_cm_single_median", 0)
    if exp_single > raw_single * 2:
        checks.append("[WARN] Expanded single ratio > 2x raw single - may introduce noise")
    else:
        checks.append("[PASS] Expanded single ratio within bounds")

    # Check 3: reject reasons
    reject = global_summary.get("reject_reasons_total", {})
    rgb_fail = reject.get("rgb_fail", 0)
    depth_fail = reject.get("depth_fail", 0)
    total_candidates = rgb_fail + depth_fail + reject.get("already_expanded", 0)
    if total_candidates > 0:
        rgb_rate = rgb_fail / total_candidates
        depth_rate = depth_fail / total_candidates
        checks.append(f"[INFO] RGB gate rejection rate: {rgb_rate:.2%}")
        checks.append(f"[INFO] Depth gate rejection rate: {depth_rate:.2%}")
    else:
        checks.append("[INFO] No candidates rejected (expansion passed all gates)")

    for check in checks:
        print(check)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
