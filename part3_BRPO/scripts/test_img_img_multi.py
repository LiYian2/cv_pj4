#!/usr/bin/env python3
"""Test MASt3R(img, img) monocular depth across multiple frames."""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add project paths
sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward

def analyze_frame(forwarder, frame_dir, verbose=True):
    """Analyze a single frame with img+img MASt3R."""
    runtime_inputs = Path(frame_dir) / 'runtime_inputs'
    pseudo_rgb_path = runtime_inputs / 'pseudo_render_rgb_runtime.png'
    pseudo_depth_path = runtime_inputs / 'pseudo_render_depth_runtime.npy'
    
    if not pseudo_rgb_path.exists() or not pseudo_depth_path.exists():
        return None
    
    # Load pseudo render depth (metric ground truth)
    pseudo_render_depth = np.load(pseudo_depth_path)
    
    # Run MASt3R(img, img) - use same image twice
    bundle = forwarder.run_pair(str(pseudo_rgb_path), str(pseudo_rgb_path))
    
    # pts3d_1.z is the depth from the first image's perspective
    mast3r_depth = bundle.pts3d_1[..., 2]  # (H, W)
    
    # Compare
    valid = (mast3r_depth > 0.1) & (pseudo_render_depth > 0.1)
    if valid.sum() < 100:
        return None
    
    pred_valid = mast3r_depth[valid]
    target_valid = pseudo_render_depth[valid]
    
    # Scale
    scale = np.median(target_valid) / np.median(pred_valid)
    calibrated = mast3r_depth * scale
    
    calib_valid = calibrated[valid]
    abs_diff = np.abs(calib_valid - target_valid)
    rel_diff = abs_diff / (target_valid + 1e-6)
    
    # Coverage
    coverage = (mast3r_depth > 0.1).sum() / mast3r_depth.size * 100
    
    results = {
        'coverage': coverage,
        'scale': scale,
        'mae': np.mean(abs_diff),
        'mre': np.median(rel_diff),
        'n_valid': valid.sum(),
        'mast3r_median': np.median(pred_valid),
        'target_median': np.median(target_valid)
    }
    
    if verbose:
        print(f"  Coverage: {coverage:.1f}% (vs pair ~3-20%)")
        print(f"  Scale: {scale:.4f}")
        print(f"  MAE: {np.mean(abs_diff):.3f}m, MRE: {np.median(rel_diff):.1%}")
    
    return results

def main():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    device = 'cuda:1'
    print("Loading MASt3R model...")
    forwarder = MASt3RPairForward(device=device, use_pair_cache=False)
    
    base_dir = Path('/data3/bzhang512/part3_online_mapping_experiments/D4_gn_scale/brpo_debug')
    
    all_results = {}
    
    for event_dir in sorted(base_dir.glob('event_kf_*')):
        event_name = event_dir.name
        print(f"\n=== {event_name} ===")
        
        for frame_dir in sorted(event_dir.glob('frame_*')):
            frame_name = frame_dir.name
            print(f"\n{frame_name}:")
            
            result = analyze_frame(forwarder, frame_dir, verbose=True)
            if result:
                all_results[f"{event_name}/{frame_name}"] = result
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    scales = [r['scale'] for r in all_results.values()]
    mres = [r['mre'] for r in all_results.values()]
    coverages = [r['coverage'] for r in all_results.values()]
    
    print(f"\nIMG+IMG Monocular Depth (n={len(scales)}):")
    print(f"  Coverage: median={np.median(coverages):.1f}%, range=[{np.min(coverages):.1f}%, {np.max(coverages):.1f}%]")
    print(f"  Scale: median={np.median(scales):.4f}, std={np.std(scales):.4f}, range=[{np.min(scales):.4f}, {np.max(scales):.4f}]")
    print(f"  MRE: median={np.median(mres):.1%}, mean={np.mean(mres):.1%}")
    
    print("\nComparison with PAIR (pseudo + ref KF rgb):")
    print("  PAIR coverage: 3-20%")
    print(f"  IMG+IMG coverage: {np.median(coverages):.1f}%")
    print("  PAIR scale: ~1.0 (already metric)")
    print(f"  IMG+IMG scale: {np.median(scales):.4f}")

if __name__ == '__main__':
    main()
