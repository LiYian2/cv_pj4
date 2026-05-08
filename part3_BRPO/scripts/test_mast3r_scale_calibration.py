#!/usr/bin/env python3
"""Test MASt3R depth scale calibration with multiple approaches."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward, MASt3RPairBundle


def load_kf_render_depth(event_dir, side="left"):
    depth_path = event_dir / "exact_backend_v1" / f"ref_depth_{side}_render.npy"
    if depth_path.exists():
        return np.load(depth_path)
    return None


def load_mast3r_projected_depth(event_dir, side="left"):
    depth_path = event_dir / "exact_backend_v1" / f"projected_depth_{side}_exact.npy"
    if depth_path.exists():
        return np.load(depth_path)
    return None


def visualize_depth(depth, title, save_path):
    valid = depth[depth > 0]
    if len(valid) == 0:
        return
    p1, p99 = np.percentile(valid, [1, 99])
    depth_norm = np.clip(depth, p1, p99)
    depth_norm = (depth_norm - p1) / (p99 - p1)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_norm, cmap="turbo")
    plt.colorbar(label="Depth (normalized)")
    plt.title(title)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def calibrate_scale_kf_anchor(mast3r_depth, kf_depth):
    mast3r_valid = mast3r_depth[mast3r_depth > 0]
    kf_valid = kf_depth[kf_depth > 0]
    if len(mast3r_valid) == 0 or len(kf_valid) == 0:
        return mast3r_depth, 1.0
    scale = np.median(kf_valid) / np.median(mast3r_valid)
    calibrated = mast3r_depth * scale
    return calibrated, scale


def calibrate_scale_mono_target(mast3r_depth, target_median=5.0):
    valid = mast3r_depth[mast3r_depth > 0]
    if len(valid) == 0:
        return mast3r_depth, 1.0
    scale = target_median / np.median(valid)
    calibrated = mast3r_depth * scale
    return calibrated, scale


def main():
    parser = argparse.ArgumentParser(description="Test MASt3R scale calibration")
    parser.add_argument("--event-dir", type=str, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-median", type=float, default=5.0)
    args = parser.parse_args()
    
    event_dir = Path(args.event_dir)
    output_dir = Path(args.output_dir) if args.output_dir else event_dir / "scale_calibration_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Event dir: {event_dir}")
    print(f"Output dir: {output_dir}")
    
    kf_depth_left = load_kf_render_depth(event_dir, "left")
    kf_depth_right = load_kf_render_depth(event_dir, "right")
    projected_left = load_mast3r_projected_depth(event_dir, "left")
    projected_right = load_mast3r_projected_depth(event_dir, "right")
    
    if kf_depth_left is None:
        print("ERROR: Could not load KF render depth")
        return
    
    print("\n=== Existing Depth Statistics ===")
    print(f"KF render depth (left): median={np.median(kf_depth_left[kf_depth_left>0]):.3f}m")
    if kf_depth_right is not None:
        print(f"KF render depth (right): median={np.median(kf_depth_right[kf_depth_right>0]):.3f}m")
    
    if projected_left is not None:
        print(f"Projected depth (left): median={np.median(projected_left[projected_left>0]):.3f}m")
    if projected_right is not None:
        print(f"Projected depth (right): median={np.median(projected_right[projected_right>0]):.3f}m")
    
    print("\n=== Approach 1: KF Depth Anchor ===")
    if projected_left is not None and kf_depth_left is not None:
        calibrated_kf, scale_kf = calibrate_scale_kf_anchor(projected_left, kf_depth_left)
        print(f"Scale factor: {scale_kf:.4f}")
        print(f"Calibrated depth median: {np.median(calibrated_kf[calibrated_kf>0]):.3f}m")
        
        valid_mask = (calibrated_kf > 0) & (kf_depth_left > 0)
        diff = np.abs(calibrated_kf[valid_mask] - kf_depth_left[valid_mask])
        print(f"Mean absolute difference vs KF depth: {np.mean(diff):.3f}m")
        print(f"Median relative difference: {np.median(diff / kf_depth_left[valid_mask]):.3f}")
        
        np.save(output_dir / "depth_calibrated_kf_anchor.npy", calibrated_kf)
        visualize_depth(calibrated_kf, "Calibrated (KF Anchor)", output_dir / "depth_calibrated_kf_anchor.png")
    
    print("\n=== Approach 2: Mono Target Median ===")
    if projected_left is not None:
        calibrated_mono, scale_mono = calibrate_scale_mono_target(projected_left, args.target_median)
        print(f"Scale factor: {scale_mono:.4f}")
        print(f"Calibrated depth median: {np.median(calibrated_mono[calibrated_mono>0]):.3f}m")
        
        valid_mask = (calibrated_mono > 0) & (kf_depth_left > 0)
        diff = np.abs(calibrated_mono[valid_mask] - kf_depth_left[valid_mask])
        print(f"Mean absolute difference vs KF depth: {np.mean(diff):.3f}m")
        
        np.save(output_dir / "depth_calibrated_mono_target.npy", calibrated_mono)
        visualize_depth(calibrated_mono, f"Calibrated (Target {args.target_median}m)", 
                        output_dir / "depth_calibrated_mono_target.png")
    
    print("\n=== Approach 3: Pair Consistency Check ===")
    if projected_left is not None and projected_right is not None:
        valid_1 = projected_left[projected_left > 0]
        valid_2 = projected_right[projected_right > 0]
        if len(valid_1) > 0 and len(valid_2) > 0:
            scale_ratio = np.median(valid_2) / np.median(valid_1)
            print(f"Scale ratio (left vs right): {scale_ratio:.4f}")
            print("(Should be ~1.0 if MASt3R outputs are consistent)")
    
    summary = {
        "kf_depth_median": float(np.median(kf_depth_left[kf_depth_left>0])),
        "projected_depth_median": float(np.median(projected_left[projected_left>0])) if projected_left is not None else None,
    }
    
    if projected_left is not None:
        summary["scale_kf_anchor"] = float(scale_kf)
        summary["scale_mono_target"] = float(scale_mono)
    
    with open(output_dir / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary saved to {output_dir / 'calibration_summary.json'} ===")


if __name__ == "__main__":
    main()
