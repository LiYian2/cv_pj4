#!/usr/bin/env python3
"""Test 2img + PAIR-proxy adaptive scale across multiple frames.
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward

def analyze_frame(forwarder, frame_dir, verbose=True):
    """Analyze a single frame with 2img + PAIR-proxy adaptive scale."""
    runtime_inputs = Path(frame_dir) / 'runtime_inputs'
    pseudo_rgb_path = runtime_inputs / 'pseudo_render_rgb_runtime.png'
    left_ref_rgb_path = runtime_inputs / 'left_ref_rgb_runtime.png'
    pseudo_depth_path = runtime_inputs / 'pseudo_render_depth_runtime.npy'
    
    if not pseudo_rgb_path.exists() or not pseudo_depth_path.exists():
        return None
    
    has_pair = left_ref_rgb_path.exists()
    
    # Load metric depth
    pseudo_render_depth = np.load(pseudo_depth_path)
    
    # 2img
    bundle_2img = forwarder.run_pair(str(pseudo_rgb_path), str(pseudo_rgb_path))
    depth_2img = bundle_2img.pts3d_1[..., 2]
    
    # PAIR
    depth_pair = None
    conf_pair = None
    if has_pair:
        bundle_pair = forwarder.run_pair(str(pseudo_rgb_path), str(left_ref_rgb_path))
        depth_pair = bundle_pair.pts3d_1[..., 2]
        conf_pair = bundle_pair.conf1
    
    valid = (depth_2img > 0.1) & (pseudo_render_depth > 0.1)
    
    results = {}
    
    # Method 1: Global scale
    global_scale = np.median(pseudo_render_depth[valid]) / np.median(depth_2img[valid])
    calibrated_global = depth_2img * global_scale
    mre_global = np.median(np.abs(calibrated_global[valid] - pseudo_render_depth[valid]) / pseudo_render_depth[valid])
    results['mre_global'] = mre_global
    results['scale_global'] = global_scale
    
    # Method 2: PAIR anchor (single scale)
    mre_pair_anchor = None
    scale_pair = None
    if has_pair and depth_pair is not None:
        scale_pair = np.median(depth_pair[valid]) / np.median(depth_2img[valid])
        calibrated_pair = depth_2img * scale_pair
        mre_pair_anchor = np.median(np.abs(calibrated_pair[valid] - pseudo_render_depth[valid]) / pseudo_render_depth[valid])
    results['mre_pair_anchor'] = mre_pair_anchor
    results['scale_pair'] = scale_pair
    
    # Method 3: PAIR-proxy adaptive scale
    mre_adaptive = None
    if has_pair and depth_pair is not None:
        ranges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
        scales_by_range = {}
        
        for r in ranges:
            mask = valid & (depth_pair >= r[0]) & (depth_pair < r[1])
            if mask.sum() > 50:
                pred = depth_2img[mask]
                target = pseudo_render_depth[mask]
                s = np.median(target) / np.median(pred)
                scales_by_range[r] = s
        
        # Apply adaptive scale
        calibrated_adaptive = np.zeros_like(depth_2img)
        for r in ranges:
            if r in scales_by_range:
                s = scales_by_range[r]
                mask = (depth_pair >= r[0]) & (depth_pair < r[1])
                calibrated_adaptive[mask] = depth_2img[mask] * s
        
        # Handle pixels not in any range
        uncovered = calibrated_adaptive == 0
        if uncovered.sum() > 0:
            # Use global scale for uncovered pixels
            calibrated_adaptive[uncovered] = depth_2img[uncovered] * global_scale
        
        valid_calib = (calibrated_adaptive > 0.1) & (pseudo_render_depth > 0.1)
        mre_adaptive = np.median(np.abs(calibrated_adaptive[valid_calib] - pseudo_render_depth[valid_calib]) / pseudo_render_depth[valid_calib])
    results['mre_adaptive'] = mre_adaptive
    
    # Method 4: Confidence-weighted PAIR-proxy adaptive
    mre_conf_adaptive = None
    if has_pair and depth_pair is not None and conf_pair is not None:
        ranges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
        scales_by_range = {}
        
        for r in ranges:
            mask = valid & (depth_pair >= r[0]) & (depth_pair < r[1]) & (conf_pair > 0.1)
            if mask.sum() > 50:
                pred = depth_2img[mask]
                target = pseudo_render_depth[mask]
                s = np.median(target) / np.median(pred)
                scales_by_range[r] = s
        
        calibrated_adaptive = np.zeros_like(depth_2img)
        for r in ranges:
            if r in scales_by_range:
                s = scales_by_range[r]
                mask = (depth_pair >= r[0]) & (depth_pair < r[1])
                calibrated_adaptive[mask] = depth_2img[mask] * s
        
        uncovered = calibrated_adaptive == 0
        if uncovered.sum() > 0:
            calibrated_adaptive[uncovered] = depth_2img[uncovered] * global_scale
        
        valid_calib = (calibrated_adaptive > 0.1) & (pseudo_render_depth > 0.1)
        mre_conf_adaptive = np.median(np.abs(calibrated_adaptive[valid_calib] - pseudo_render_depth[valid_calib]) / pseudo_render_depth[valid_calib])
    results['mre_conf_adaptive'] = mre_conf_adaptive
    
    if verbose:
        print(f"  Global: MRE={mre_global:.1%}")
        if mre_pair_anchor:
            print(f"  PAIR anchor: MRE={mre_pair_anchor:.1%}")
        if mre_adaptive:
            print(f"  PAIR-proxy adaptive: MRE={mre_adaptive:.1%}")
        if mre_conf_adaptive:
            print(f"  Conf-weighted adaptive: MRE={mre_conf_adaptive:.1%}")
    
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
    print("SUMMARY - Multi-frame Stability Analysis")
    print("="*60)
    
    mres_global = [r['mre_global'] for r in all_results.values() if r['mre_global']]
    mres_pair = [r['mre_pair_anchor'] for r in all_results.values() if r['mre_pair_anchor']]
    mres_adaptive = [r['mre_adaptive'] for r in all_results.values() if r['mre_adaptive']]
    mres_conf_adaptive = [r['mre_conf_adaptive'] for r in all_results.values() if r['mre_conf_adaptive']]
    
    print(f"\nGlobal scale (n={len(mres_global)}):")
    print(f"  MRE: median={np.median(mres_global):.1%}, mean={np.mean(mres_global):.1%}, std={np.std(mres_global):.1%}")
    
    if mres_pair:
        print(f"\nPAIR anchor (n={len(mres_pair)}):")
        print(f"  MRE: median={np.median(mres_pair):.1%}, mean={np.mean(mres_pair):.1%}, std={np.std(mres_pair):.1%}")
    
    if mres_adaptive:
        print(f"\nPAIR-proxy adaptive (n={len(mres_adaptive)}):")
        print(f"  MRE: median={np.median(mres_adaptive):.1%}, mean={np.mean(mres_adaptive):.1%}, std={np.std(mres_adaptive):.1%}")
        print(f"  Range: [{np.min(mres_adaptive):.1%}, {np.max(mres_adaptive):.1%}]")
    
    if mres_conf_adaptive:
        print(f"\nConf-weighted adaptive (n={len(mres_conf_adaptive)}):")
        print(f"  MRE: median={np.median(mres_conf_adaptive):.1%}, mean={np.mean(mres_conf_adaptive):.1%}, std={np.std(mres_conf_adaptive):.1%}")
    
    # Final assessment
    print("\n" + "="*60)
    print("STABILITY ASSESSMENT")
    print("="*60)
    
    if mres_adaptive:
        median_mre = np.median(mres_adaptive)
        std_mre = np.std(mres_adaptive)
        
        if median_mre < 25 and std_mre < 10:
            print(f"✓ Method is STABLE: median={median_mre:.1%}, std={std_mre:.1%}")
            print("Recommendation: Proceed to integration planning.")
        elif median_mre < 30 and std_mre < 15:
            print(f"? Method is MARGINAL: median={median_mre:.1%}, std={std_mre:.1%}")
            print("Recommendation: Consider further tuning before integration.")
        else:
            print(f"✗ Method is UNSTABLE: median={median_mre:.1%}, std={std_mre:.1%}")
            print("Recommendation: Investigate failure cases or consider alternative approaches.")
    
    # Compare improvements
    if mres_global and mres_adaptive:
        improvement_global_to_adaptive = np.mean(mres_global) - np.mean(mres_adaptive)
        print(f"\nImprovement from global to adaptive: {improvement_global_to_adaptive:.1%}")
    
    if mres_pair and mres_adaptive:
        improvement_pair_to_adaptive = np.mean(mres_pair) - np.mean(mres_adaptive)
        print(f"Improvement from PAIR anchor to adaptive: {improvement_pair_to_adaptive:.1%}")
    
    # Best case analysis
    if mres_adaptive:
        best_idx = np.argmin(mres_adaptive)
        best_key = list(all_results.keys())[best_idx]
        best_mre = mres_adaptive[best_idx]
        worst_idx = np.argmax(mres_adaptive)
        worst_key = list(all_results.keys())[worst_idx]
        worst_mre = mres_adaptive[worst_idx]
        print(f"\nBest case: {best_key} with MRE={best_mre:.1%}")
        print(f"Worst case: {worst_key} with MRE={worst_mre:.1%}")
        
        # Analyze worst case
        print(f"\nAnalyzing worst case ({worst_key}):")
        worst_result = all_results[worst_key]
        print(f"  Global MRE: {worst_result['mre_global']:.1%}")
        if worst_result['mre_pair_anchor']:
            print(f"  PAIR anchor MRE: {worst_result['mre_pair_anchor']:.1%}")
        print(f"  Adaptive MRE: {worst_result['mre_adaptive']:.1%}")
        
        # Check if adaptive still helps in worst case
        if worst_result['mre_adaptive'] < worst_result['mre_global']:
            print(f"  Adaptive still improves over global by {worst_result['mre_global'] - worst_result['mre_adaptive']:.1%}")
        else:
            print(f"  WARNING: Adaptive worse than global in this case!")

if __name__ == '__main__':
    main()
