#!/usr/bin/env python3
"""Test 2img + PAIR anchor calibration.
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward

def analyze_frame(forwarder, frame_dir, verbose=True):
    """Analyze a single frame with 2img + PAIR anchor."""
    runtime_inputs = Path(frame_dir) / 'runtime_inputs'
    pseudo_rgb_path = runtime_inputs / 'pseudo_render_rgb_runtime.png'
    left_ref_rgb_path = runtime_inputs / 'left_ref_rgb_runtime.png'
    pseudo_depth_path = runtime_inputs / 'pseudo_render_depth_runtime.npy'
    
    if not pseudo_rgb_path.exists() or not pseudo_depth_path.exists():
        return None
    
    has_left_ref = left_ref_rgb_path.exists()
    
    # Load pseudo render depth (metric ground truth)
    pseudo_render_depth = np.load(pseudo_depth_path)
    
    # Step 1: 2img - MASt3R(pseudo, pseudo)
    bundle_2img = forwarder.run_pair(str(pseudo_rgb_path), str(pseudo_rgb_path))
    depth_2img = bundle_2img.pts3d_1[..., 2]  # (H, W)
    coverage_2img = (depth_2img > 0.1).sum() / depth_2img.size * 100
    
    # Step 2: PAIR - MASt3R(pseudo, ref KF) if available
    depth_pair = None
    conf_pair = None
    coverage_pair = 0
    
    if has_left_ref:
        bundle_pair = forwarder.run_pair(str(pseudo_rgb_path), str(left_ref_rgb_path))
        depth_pair = bundle_pair.pts3d_1[..., 2]
        conf_pair = bundle_pair.conf1
        coverage_pair = (depth_pair > 0.1).sum() / depth_pair.size * 100
    
    # Step 3: Calibration methods
    
    # Method A: Direct pseudo_render anchor (baseline)
    valid_a = (depth_2img > 0.1) & (pseudo_render_depth > 0.1)
    scale_a, mre_a = None, None
    if valid_a.sum() > 100:
        scale_a = np.median(pseudo_render_depth[valid_a]) / np.median(depth_2img[valid_a])
        calibrated_a = depth_2img * scale_a
        mre_a = np.median(np.abs(calibrated_a[valid_a] - pseudo_render_depth[valid_a]) / pseudo_render_depth[valid_a])
    
    # Method B: PAIR anchor (use PAIR's high-confidence region)
    scale_b, mre_b, n_anchor = None, None, 0
    if has_left_ref and conf_pair is not None:
        high_conf_mask = conf_pair > 0.3
        valid_pair = high_conf_mask & (depth_pair > 0.1) & (depth_2img > 0.1)
        if valid_pair.sum() > 100:
            scale_b = np.median(depth_pair[valid_pair]) / np.median(depth_2img[valid_pair])
            calibrated_b = depth_2img * scale_b
            valid_b = (calibrated_b > 0.1) & (pseudo_render_depth > 0.1)
            mre_b = np.median(np.abs(calibrated_b[valid_b] - pseudo_render_depth[valid_b]) / pseudo_render_depth[valid_b])
            n_anchor = valid_pair.sum()
    
    # Method C: Lower threshold PAIR anchor (conf > 0.1)
    scale_c, mre_c, n_anchor_c = None, None, 0
    if has_left_ref and conf_pair is not None:
        valid_pair = (conf_pair > 0.1) & (depth_pair > 0.1) & (depth_2img > 0.1)
        if valid_pair.sum() > 100:
            scale_c = np.median(depth_pair[valid_pair]) / np.median(depth_2img[valid_pair])
            calibrated_c = depth_2img * scale_c
            valid_c = (calibrated_c > 0.1) & (pseudo_render_depth > 0.1)
            mre_c = np.median(np.abs(calibrated_c[valid_c] - pseudo_render_depth[valid_c]) / pseudo_render_depth[valid_c])
            n_anchor_c = valid_pair.sum()
    
    # Method D: Use projected_depth from exact_backend as anchor
    exact_backend_path = Path(frame_dir) / 'exact_backend_v1'
    projected_path = exact_backend_path / 'projected_depth_left_exact.npy'
    
    scale_d, mre_d, n_anchor_d = None, None, 0
    if projected_path.exists():
        projected_depth = np.load(projected_path)
        exact_conf_path = exact_backend_path / 'confidence_left_exact.npy'
        exact_conf = np.load(exact_conf_path) if exact_conf_path.exists() else None
        
        if exact_conf is not None:
            exact_valid = (exact_conf > 0.1) & (projected_depth > 0.1) & (depth_2img > 0.1)
            if exact_valid.sum() > 100:
                scale_d = np.median(projected_depth[exact_valid]) / np.median(depth_2img[exact_valid])
                calibrated_d = depth_2img * scale_d
                valid_d = (calibrated_d > 0.1) & (pseudo_render_depth > 0.1)
                mre_d = np.median(np.abs(calibrated_d[valid_d] - pseudo_render_depth[valid_d]) / pseudo_render_depth[valid_d])
                n_anchor_d = exact_valid.sum()
    
    results = {
        'coverage_2img': coverage_2img,
        'coverage_pair': coverage_pair,
        'scale_direct': scale_a,
        'mre_direct': mre_a,
        'scale_pair_anchor': scale_b,
        'mre_pair_anchor': mre_b,
        'n_anchor': n_anchor,
        'scale_pair_low_thresh': scale_c,
        'mre_pair_low_thresh': mre_c,
        'n_anchor_low_thresh': n_anchor_c,
        'scale_exact_anchor': scale_d,
        'mre_exact_anchor': mre_d,
        'n_exact_anchor': n_anchor_d,
    }
    
    if verbose:
        print(f"  2img coverage: {coverage_2img:.1f}%")
        if has_left_ref:
            print(f"  PAIR coverage (raw MASt3R): {coverage_pair:.1f}%")
        
        if scale_a:
            print(f"  Method A (pseudo_render): scale={scale_a:.4f}, MRE={mre_a:.1%}")
        if scale_b:
            print(f"  Method B (PAIR conf>0.3): scale={scale_b:.4f}, MRE={mre_b:.1%}, n={n_anchor}")
        if scale_c:
            print(f"  Method C (PAIR conf>0.1): scale={scale_c:.4f}, MRE={mre_c:.1%}, n={n_anchor_c}")
        if scale_d:
            print(f"  Method D (exact_backend): scale={scale_d:.4f}, MRE={mre_d:.1%}, n={n_anchor_d}")
    
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
    
    scales_a = [r['scale_direct'] for r in all_results.values() if r['scale_direct']]
    mres_a = [r['mre_direct'] for r in all_results.values() if r['mre_direct']]
    if scales_a:
        print(f"\nMethod A (pseudo_render anchor, n={len(scales_a)}):")
        print(f"  Scale: median={np.median(scales_a):.4f}, std={np.std(scales_a):.4f}")
        print(f"  MRE: median={np.median(mres_a):.1%}")
    
    scales_b = [r['scale_pair_anchor'] for r in all_results.values() if r['scale_pair_anchor']]
    mres_b = [r['mre_pair_anchor'] for r in all_results.values() if r['mre_pair_anchor']]
    if scales_b:
        print(f"\nMethod B (PAIR conf>0.3, n={len(scales_b)}):")
        print(f"  Scale: median={np.median(scales_b):.4f}, std={np.std(scales_b):.4f}")
        print(f"  MRE: median={np.median(mres_b):.1%}")
    
    scales_c = [r['scale_pair_low_thresh'] for r in all_results.values() if r['scale_pair_low_thresh']]
    mres_c = [r['mre_pair_low_thresh'] for r in all_results.values() if r['mre_pair_low_thresh']]
    if scales_c:
        print(f"\nMethod C (PAIR conf>0.1, n={len(scales_c)}):")
        print(f"  Scale: median={np.median(scales_c):.4f}, std={np.std(scales_c):.4f}")
        print(f"  MRE: median={np.median(mres_c):.1%}")
    
    scales_d = [r['scale_exact_anchor'] for r in all_results.values() if r['scale_exact_anchor']]
    mres_d = [r['mre_exact_anchor'] for r in all_results.values() if r['mre_exact_anchor']]
    if scales_d:
        print(f"\nMethod D (exact_backend anchor, n={len(scales_d)}):")
        print(f"  Scale: median={np.median(scales_d):.4f}, std={np.std(scales_d):.4f}")
        print(f"  MRE: median={np.median(mres_d):.1%}")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    if scales_a:
        print(f"Without anchor (Method A): MRE={np.median(mres_a):.1%}")
    if scales_d:
        print(f"With exact_backend anchor (Method D): MRE={np.median(mres_d):.1%}")
        if scales_a:
            print(f"Improvement: {np.median(mres_a) - np.median(mres_d):.1%}")
    
    # Analysis of PAIR vs exact_backend
    print("\nNote: PAIR (raw MASt3R) coverage=100% because MASt3R always outputs full depth.")
    print("exact_backend has low coverage (3-20%) because it's filtered by confidence.")

if __name__ == '__main__':
    main()
