# -*- coding: utf-8 -*-
"""Compare MASt3R monocular depth (img+img) vs pair depth (img+ref).

Tests:
1. Monocular: MASt3R(pseudo_gt, pseudo_gt) -> pts3d_1.z
2. Pair: MASt3R(pseudo_gt, ref_gt) -> pts3d_1.z vs pts3d_2_in_1.z
3. Scale calibration comparison

"""
import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
import torch

S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)

BRPO_ROOT = "/home/bzhang512/CV_Project/part3_BRPO"
if BRPO_ROOT not in sys.path:
    sys.path.insert(0, BRPO_ROOT)

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward


def depth_to_color(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if vmin is None:
        vmin = depth[depth > 0].min() if (depth > 0).any() else 0.0
    if vmax is None:
        vmax = depth[depth > 0].max() if (depth > 0).any() else 1.0
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    h, w = depth_norm.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[..., 2] = np.uint8(255 * (1.0 - depth_norm))
    mid = np.abs(depth_norm - 0.5)
    color[..., 1] = np.uint8(255 * (1.0 - 2 * mid))
    color[..., 0] = np.uint8(255 * depth_norm)
    return color


def test_comparison(
    pseudo_paths: list[str],
    ref_paths: list[str],
    output_dir: str,
    size: int = 512,
    device: str = "cuda",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing MASt3R monocular vs pair depth")
    print(f"Pseudo frames: {len(pseudo_paths)}, Ref frames: {len(ref_paths)}")
    
    forwarder = MASt3RPairForward(device=device, use_pair_cache=False)
    
    results = []
    
    for i, (pseudo_path, ref_path) in enumerate(zip(pseudo_paths, ref_paths)):
        pseudo_path = Path(pseudo_path)
        ref_path = Path(ref_path)
        print(f"\n--- Frame {i}: pseudo={pseudo_path.name}, ref={ref_path.name} ---")
        
        # Method 1: Monocular (img, img)
        print(f"Method 1: MASt3R(pseudo, pseudo) monocular...")
        bundle_mono = forwarder.run_pair(str(pseudo_path), str(pseudo_path), size=size)
        depth_mono = bundle_mono.pts3d_1[..., 2]
        conf_mono = bundle_mono.conf1
        
        # Method 2: Pair (pseudo, ref)
        print(f"Method 2: MASt3R(pseudo, ref) pair...")
        bundle_pair = forwarder.run_pair(str(pseudo_path), str(ref_path), size=size)
        depth_pair_pts3d_1 = bundle_pair.pts3d_1[..., 2]
        depth_pair_pts3d_2_in_1 = bundle_pair.pts3d_2_in_1[..., 2]
        conf_pair_1 = bundle_pair.conf1
        conf_pair_2 = bundle_pair.conf2
        
        # Statistics
        print(f"\n  === Monocular ===")
        valid_mono = depth_mono[depth_mono > 0]
        if valid_mono.size > 0:
            print(f"    depth_mono: min={valid_mono.min():.4f}, max={valid_mono.max():.4f}, median={np.median(valid_mono):.4f}")
        
        print(f"\n  === Pair ===")
        valid_pts3d_1 = depth_pair_pts3d_1[depth_pair_pts3d_1 > 0]
        valid_pts3d_2 = depth_pair_pts3d_2_in_1[depth_pair_pts3d_2_in_1 > 0]
        if valid_pts3d_1.size > 0:
            print(f"    pts3d_1.z: min={valid_pts3d_1.min():.4f}, max={valid_pts3d_1.max():.4f}, median={np.median(valid_pts3d_1):.4f}")
        if valid_pts3d_2.size > 0:
            print(f"    pts3d_2_in_1.z: min={valid_pts3d_2.min():.4f}, max={valid_pts3d_2.max():.4f}, median={np.median(valid_pts3d_2):.4f}")
        
        # Scale calibration comparison
        # Method A: pair pts3d_2_in_1.z as scale anchor
        # Method B: monocular pts3d_1.z with scene-level scale
        
        # Scale A: pts3d_2_in_1.z / pts3d_1.z (pair)
        valid_both_pair = (depth_pair_pts3d_1 > 0) & (depth_pair_pts3d_2_in_1 > 0)
        if valid_both_pair.sum() > 100:
            scale_A = np.median(depth_pair_pts3d_2_in_1[valid_both_pair] / depth_pair_pts3d_1[valid_both_pair])
            depth_calibrated_A = depth_pair_pts3d_1 * scale_A
            print(f"\n  Scale A (pair pts3d_2_in_1 anchor): {scale_A:.4f}")
            valid_A = depth_calibrated_A[depth_calibrated_A > 0]
            if valid_A.size > 0:
                print(f"    depth_calibrated_A: median={np.median(valid_A):.4f}")
        else:
            scale_A = 1.0
            depth_calibrated_A = depth_pair_pts3d_1
        
        # Scale B: monocular pts3d_1.z with target median 5m
        valid_mono = depth_mono > 0
        if valid_mono.sum() > 100:
            target_median = 5.0
            current_median = np.median(depth_mono[valid_mono])
            scale_B = target_median / current_median
            depth_calibrated_B = depth_mono * scale_B
            print(f"\n  Scale B (monocular target median 5m): {scale_B:.4f}")
            valid_B = depth_calibrated_B[depth_calibrated_B > 0]
            if valid_B.size > 0:
                print(f"    depth_calibrated_B: median={np.median(valid_B):.4f}")
        else:
            scale_B = 1.0
            depth_calibrated_B = depth_mono
        
        # Compare calibrated depths
        # Check if Scale A and Scale B produce similar results
        if valid_both_pair.sum() > 100:
            # Compare in overlap region
            diff_A_B = np.abs(depth_calibrated_A - depth_calibrated_B)
            rel_diff = diff_A_B / (np.minimum(depth_calibrated_A, depth_calibrated_B) + 1e-8)
            print(f"\n  Comparison (calibrated A vs B):")
            print(f"    mean_abs_diff: {diff_A_B.mean():.4f}")
            print(f"    median_rel_diff: {np.median(rel_diff):.4f}")
        
        # Save outputs
        frame_output = output_dir / f"frame_{i:03d}"
        frame_output.mkdir(parents=True, exist_ok=True)
        
        # RGB
        rgb_img = np.array(Image.open(pseudo_path).resize((size, size)))
        Image.fromarray(rgb_img).save(frame_output / "pseudo_rgb.png")
        
        # Depths
        np.save(frame_output / "depth_mono_raw.npy", depth_mono)
        np.save(frame_output / "depth_pair_pts3d_1.npy", depth_pair_pts3d_1)
        np.save(frame_output / "depth_pair_pts3d_2_in_1.npy", depth_pair_pts3d_2_in_1)
        np.save(frame_output / "depth_calibrated_A.npy", depth_calibrated_A)
        np.save(frame_output / "depth_calibrated_B.npy", depth_calibrated_B)
        
        # Visualizations
        Image.fromarray(depth_to_color(depth_mono)).save(frame_output / "depth_mono_raw_vis.png")
        Image.fromarray(depth_to_color(depth_pair_pts3d_1)).save(frame_output / "depth_pair_pts3d_1_vis.png")
        Image.fromarray(depth_to_color(depth_pair_pts3d_2_in_1)).save(frame_output / "depth_pair_pts3d_2_in_1_vis.png")
        Image.fromarray(depth_to_color(depth_calibrated_A, vmin=0.5, vmax=10.0)).save(frame_output / "depth_calibrated_A_vis.png")
        Image.fromarray(depth_to_color(depth_calibrated_B, vmin=0.5, vmax=10.0)).save(frame_output / "depth_calibrated_B_vis.png")
        
        # Meta
        meta = {
            "pseudo_path": str(pseudo_path),
            "ref_path": str(ref_path),
            "depth_stats": {
                "mono_raw": {"median": float(np.median(depth_mono[valid_mono])) if valid_mono.sum() > 0 else 0},
                "pair_pts3d_1": {"median": float(np.median(valid_pts3d_1)) if valid_pts3d_1.size > 0 else 0},
                "pair_pts3d_2_in_1": {"median": float(np.median(valid_pts3d_2)) if valid_pts3d_2.size > 0 else 0},
                "calibrated_A": {"median": float(np.median(valid_A)) if 'valid_A' in dir() and valid_A.size > 0 else 0, "scale": float(scale_A)},
                "calibrated_B": {"median": float(np.median(valid_B)) if 'valid_B' in dir() and valid_B.size > 0 else 0, "scale": float(scale_B)},
            },
        }
        with open(frame_output / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        results.append({
            "frame": pseudo_path.stem,
            "scale_A": float(scale_A),
            "scale_B": float(scale_B),
            "median_calibrated_A": meta["depth_stats"]["calibrated_A"]["median"],
            "median_calibrated_B": meta["depth_stats"]["calibrated_B"]["median"],
        })
        
        print(f"  Saved to: {frame_output}")
    
    # Summary
    print(f"\n=== Summary ===")
    for r in results:
        print(f"  {r['frame']}: scale_A={r['scale_A']:.4f}, scale_B={r['scale_B']:.4f}, median_A={r['median_calibrated_A']:.4f}, median_B={r['median_calibrated_B']:.4f}")
    
    summary = {
        "total_frames": len(results),
        "frames": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test MASt3R monocular vs pair depth")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/bzhang512/CV_Project/dataset/DL3DV-2/part2_s3po/full/rgb")
    parser.add_argument("--output_dir", type=str,
                        default="/home/bzhang512/CV_Project/part3_BRPO/output/mast3r_mono_vs_pair_test")
    parser.add_argument("--pseudo_indices", type=str, default="50,100,150",
                        help="Pseudo frame indices")
    parser.add_argument("--ref_indices", type=str, default="0,33,67",
                        help="Ref frame indices (paired with pseudo)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    all_frames = sorted(dataset_path.glob("frame_*.png"))
    
    pseudo_indices = [int(x) for x in args.pseudo_indices.split(",")]
    ref_indices = [int(x) for x in args.ref_indices.split(",")]
    
    pseudo_paths = [str(all_frames[i]) for i in pseudo_indices if i < len(all_frames)]
    ref_paths = [str(all_frames[i]) for i in ref_indices if i < len(all_frames)]
    
    # Pair them (zip)
    if len(pseudo_paths) != len(ref_paths):
        print(f"Warning: pseudo ({len(pseudo_paths)}) and ref ({len(ref_paths)}) count mismatch")
        min_len = min(len(pseudo_paths), len(ref_paths))
        pseudo_paths = pseudo_paths[:min_len]
        ref_paths = ref_paths[:min_len]
    
    print(f"Pseudo: {[Path(f).name for f in pseudo_paths]}")
    print(f"Ref: {[Path(f).name for f in ref_paths]}")
    
    test_comparison(
        pseudo_paths=pseudo_paths,
        ref_paths=ref_paths,
        output_dir=args.output_dir,
        size=args.size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
