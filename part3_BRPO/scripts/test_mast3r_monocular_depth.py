# -*- coding: utf-8 -*-
"""Test MASt3R monocular depth estimation (img + img).

This script validates:
1. MASt3R(img, img) produces reasonable pts3d_1.z
2. Different scale calibration methods
3. Output visualization for multiple pseudo frames

No modification to existing online mapping pipeline.
"""
import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
import torch

# Add paths
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)

BRPO_ROOT = "/home/bzhang512/CV_Project/part3_BRPO"
if BRPO_ROOT not in sys.path:
    sys.path.insert(0, BRPO_ROOT)

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward, MASt3RPairBundle


def depth_to_color(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Convert depth map to color image for visualization."""
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if vmin is None:
        vmin = depth[depth > 0].min() if (depth > 0).any() else 0.0
    if vmax is None:
        vmax = depth[depth > 0].max() if (depth > 0).any() else 1.0
    
    # Normalize
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    
    # Apply colormap (simple blue-green-red)
    h, w = depth_norm.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Blue channel: low depth
    color[..., 2] = np.uint8(255 * (1.0 - depth_norm))
    # Green channel: mid depth
    mid = np.abs(depth_norm - 0.5)
    color[..., 1] = np.uint8(255 * (1.0 - 2 * mid))
    # Red channel: high depth
    color[..., 0] = np.uint8(255 * depth_norm)
    
    return color


def test_mast3r_monocular_depth(
    rgb_paths: list[str],
    output_dir: str,
    size: int = 512,
    device: str = "cuda",
):
    """Test MASt3R monocular depth estimation on multiple frames."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing MASt3R monocular depth on {len(rgb_paths)} frames")
    print(f"Output dir: {output_dir}")
    
    # Initialize MASt3R
    forwarder = MASt3RPairForward(device=device, use_pair_cache=False)
    
    results = []
    
    for i, rgb_path in enumerate(rgb_paths):
        rgb_path = Path(rgb_path)
        print(f"\n--- Frame {i}: {rgb_path.name} ---")
        
        # Load RGB for reference
        rgb_img = np.array(Image.open(rgb_path).resize((size, size)))
        
        # Call MASt3R with same image twice
        print(f"Calling MASt3R(img, img)...")
        bundle = forwarder.run_pair(str(rgb_path), str(rgb_path), size=size)
        
        # Extract pts3d_1.z (monocular depth)
        pts3d_1 = bundle.pts3d_1
        if pts3d_1 is None:
            print(f"WARNING: pts3d_1 is None!")
            continue
        
        depth_relative = pts3d_1[..., 2]  # Z coordinate
        confidence = bundle.conf1
        
        # Also check pts3d_2_in_1 (should be same as pts3d_1 for identical input)
        pts3d_2_in_1 = bundle.pts3d_2_in_1
        if pts3d_2_in_1 is not None:
            depth_2_in_1 = pts3d_2_in_1[..., 2]
            depth_diff = np.abs(depth_relative - depth_2_in_1)
            print(f"  pts3d_1.z vs pts3d_2_in_1.z: mean_diff={depth_diff.mean():.4f}, max_diff={depth_diff.max():.4f}")
        
        # Statistics
        valid_depth = depth_relative[depth_relative > 0]
        if valid_depth.size > 0:
            print(f"  depth_relative: min={valid_depth.min():.4f}, max={valid_depth.max():.4f}, mean={valid_depth.mean():.4f}, median={np.median(valid_depth):.4f}")
        
        valid_conf = confidence[confidence > 0]
        if valid_conf.size > 0:
            print(f"  confidence: min={valid_conf.min():.4f}, max={valid_conf.max():.4f}, mean={valid_conf.mean():.4f}")
        
        # Scale calibration methods
        scale_results = {}
        
        # Method 1: Identity (no calibration, use raw relative depth)
        scale_results["identity"] = {
            "depth": depth_relative.copy(),
            "scale": 1.0,
            "method": "identity (no calibration)",
        }
        
        # Method 2: Focal-based scale (use focal length as depth scale hint)
        if valid_depth.size > 0:
            focal = 408.0  # from config
            target_median = 5.0
            current_median = np.median(valid_depth)
            focal_scale = target_median / (current_median + 1e-8)
            depth_focal_calibrated = depth_relative * focal_scale
            scale_results["focal_hint"] = {
                "depth": depth_focal_calibrated,
                "scale": focal_scale,
                "method": f"focal_hint (target_median={target_median}m)",
            }
            print(f"  focal_hint scale: {focal_scale:.4f}")
        
        # Method 3: Percentile normalization
        if valid_depth.size > 0:
            p5 = np.percentile(valid_depth, 5)
            p95 = np.percentile(valid_depth, 95)
            target_min = 0.5
            target_max = 10.0
            percentile_scale = (target_max - target_min) / (p95 - p5 + 1e-8)
            depth_percentile_calibrated = (depth_relative - p5) * percentile_scale + target_min
            depth_percentile_calibrated = np.clip(depth_percentile_calibrated, target_min, target_max)
            scale_results["percentile_norm"] = {
                "depth": depth_percentile_calibrated,
                "scale": percentile_scale,
                "method": f"percentile_norm (0.5-10m)",
            }
            print(f"  percentile_norm scale: {percentile_scale:.4f}")
        
        # Method 4: Log-scale normalization
        if valid_depth.size > 0:
            depth_log = np.log(depth_relative + 1e-8)
            log_mean = depth_log[depth_relative > 0].mean()
            log_std = depth_log[depth_relative > 0].std()
            depth_log_norm = (depth_log - log_mean) / (log_std + 1e-8)
            depth_log_calibrated = np.exp(depth_log_norm * 1.0 + 1.5)
            depth_log_calibrated = np.clip(depth_log_calibrated, 0.5, 10.0)
            scale_results["log_norm"] = {
                "depth": depth_log_calibrated,
                "scale": "log-based",
                "method": f"log_norm (center ~4.5m)",
            }
        
        # Save results for this frame
        frame_output = output_dir / f"frame_{i:03d}_{rgb_path.stem}"
        frame_output.mkdir(parents=True, exist_ok=True)
        
        # Save RGB
        Image.fromarray(rgb_img).save(frame_output / "rgb.png")
        
        # Save each scale result
        for method_name, method_data in scale_results.items():
            depth_calibrated = method_data["depth"]
            np.save(frame_output / f"depth_{method_name}.npy", depth_calibrated)
            depth_color = depth_to_color(depth_calibrated, vmin=0.5, vmax=10.0)
            Image.fromarray(depth_color).save(frame_output / f"depth_{method_name}_vis.png")
        
        # Save confidence
        if confidence is not None:
            np.save(frame_output / "confidence.npy", confidence)
            conf_color = depth_to_color(confidence, vmin=0, vmax=confidence.max() if confidence.max() > 0 else 1.0)
            Image.fromarray(conf_color).save(frame_output / "confidence_vis.png")
        
        # Save raw depth
        np.save(frame_output / "depth_raw.npy", depth_relative)
        depth_raw_color = depth_to_color(depth_relative)
        Image.fromarray(depth_raw_color).save(frame_output / "depth_raw_vis.png")
        
        # Save pts3d_1 (full 3D)
        np.save(frame_output / "pts3d_1.npy", pts3d_1)
        
        # Save meta
        meta = {
            "rgb_path": str(rgb_path),
            "method": "MASt3R(img, img) monocular",
            "depth_stats": {
                "raw": {
                    "min": float(valid_depth.min()) if valid_depth.size > 0 else 0,
                    "max": float(valid_depth.max()) if valid_depth.size > 0 else 0,
                    "mean": float(valid_depth.mean()) if valid_depth.size > 0 else 0,
                    "median": float(np.median(valid_depth)) if valid_depth.size > 0 else 0,
                },
            },
            "scale_methods": {k: {"scale": v["scale"], "method": v["method"]} for k, v in scale_results.items()},
            "bundle_meta": bundle.meta,
        }
        with open(frame_output / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        results.append({
            "frame": rgb_path.stem,
            "depth_raw_median": float(np.median(valid_depth)) if valid_depth.size > 0 else 0,
            "scale_focal_hint": scale_results.get("focal_hint", {}).get("scale", 1.0),
        })
        
        print(f"  Saved to: {frame_output}")
    
    # Summary
    print(f"\n=== Summary ===")
    for r in results:
        print(f"  {r['frame']}: depth_median={r['depth_raw_median']:.4f}, scale_focal_hint={r['scale_focal_hint']:.4f}")
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_frames": len(results),
            "frames": results,
            "test_method": "MASt3R(img, img) monocular depth",
        }, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test MASt3R monocular depth")
    parser.add_argument("--dataset_path", type=str, 
                        default="/home/bzhang512/CV_Project/dataset/DL3DV-2/part2_s3po/full/rgb")
    parser.add_argument("--output_dir", type=str,
                        default="/home/bzhang512/CV_Project/part3_BRPO/output/mast3r_monocular_test")
    parser.add_argument("--frames", type=str, default="1,50,100,150,200",
                        help="Frame indices to test (comma-separated)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    frame_indices = [int(x) for x in args.frames.split(",")]
    
    all_frames = sorted(dataset_path.glob("frame_*.png"))
    if not all_frames:
        print(f"No frames found in {dataset_path}")
        return
    
    selected_frames = []
    for idx in frame_indices:
        if idx < len(all_frames):
            selected_frames.append(str(all_frames[idx]))
        else:
            print(f"Warning: frame index {idx} out of range (max {len(all_frames)-1})")
    
    if not selected_frames:
        print(f"No valid frames selected")
        return
    
    print(f"Selected frames: {[Path(f).name for f in selected_frames]}")
    
    test_mast3r_monocular_depth(
        rgb_paths=selected_frames,
        output_dir=args.output_dir,
        size=args.size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
