#!/usr/bin/env python3
"""Multi-frame scale calibration comparison for MASt3R depth."""
import os
import sys
import json
import numpy as np
from pathlib import Path

def load_depth(path):
    """Load depth .npy file."""
    if os.path.exists(path):
        return np.load(path)
    return None

def compute_scale_and_error(pred_depth, target_depth, valid_mask=None):
    """Compute scale factor and error metrics."""
    if valid_mask is None:
        valid_mask = (pred_depth > 0.1) & (target_depth > 0.1)
    else:
        valid_mask = valid_mask & (pred_depth > 0.1) & (target_depth > 0.1)
    
    if valid_mask.sum() < 100:
        return None, None, None, 0
    
    pred_valid = pred_depth[valid_mask]
    target_valid = target_depth[valid_mask]
    
    # Scale = median(target) / median(pred)
    scale = np.median(target_valid) / np.median(pred_valid)
    
    # Calibrate
    calibrated = pred_depth * scale
    
    # Error
    calib_valid = calibrated[valid_mask]
    abs_diff = np.abs(calib_valid - target_valid)
    rel_diff = abs_diff / (target_valid + 1e-6)
    
    return scale, np.mean(abs_diff), np.median(rel_diff), valid_mask.sum()

def analyze_frame(frame_dir, verbose=True):
    """Analyze a single frame for scale calibration comparison."""
    results = {}
    
    # Paths
    runtime_inputs = Path(frame_dir) / 'runtime_inputs'
    exact_backend = Path(frame_dir) / 'exact_backend_v1'
    
    # Load metric anchor (pseudo render depth)
    pseudo_render_path = runtime_inputs / 'pseudo_render_depth_runtime.npy'
    pseudo_render = load_depth(pseudo_render_path)
    
    # Load MASt3R projected depth
    projected_path = exact_backend / 'projected_depth_left_exact.npy'
    projected = load_depth(projected_path)
    
    # Load confidence
    conf_path = exact_backend / 'confidence_left_exact.npy'
    confidence = load_depth(conf_path)
    
    if pseudo_render is None or projected is None:
        return None
    
    results['pseudo_render_median'] = np.median(pseudo_render[pseudo_render > 0.1]) if (pseudo_render > 0.1).any() else 0
    results['projected_median'] = np.median(projected[projected > 0.1]) if (projected > 0.1).any() else 0
    results['projected_coverage'] = (projected > 0.1).sum() / projected.size * 100
    
    # Method 1: Pseudo render anchor
    scale1, mae1, mre1, n1 = compute_scale_and_error(projected, pseudo_render)
    results['pseudo_anchor'] = {'scale': scale1, 'mae': mae1, 'mre': mre1, 'n_valid': n1}
    
    # Method 2: High confidence anchor (if confidence available)
    if confidence is not None:
        high_conf_mask = confidence > 0.3
        scale2, mae2, mre2, n2 = compute_scale_and_error(projected, pseudo_render, high_conf_mask)
        results['high_conf_anchor'] = {'scale': scale2, 'mae': mae2, 'mre': mre2, 'n_valid': n2}
    
    # Method 3: Mono depth style (scale to median=5m)
    if results['projected_median'] > 0:
        mono_scale = 5.0 / results['projected_median']
        calibrated_mono = projected * mono_scale
        valid_mask = (projected > 0.1) & (pseudo_render > 0.1)
        if valid_mask.sum() > 100:
            calib_valid = calibrated_mono[valid_mask]
            target_valid = pseudo_render[valid_mask]
            abs_diff = np.abs(calib_valid - target_valid)
            rel_diff = abs_diff / (target_valid + 1e-6)
            results['mono_5m'] = {'scale': mono_scale, 'mae': np.mean(abs_diff), 'mre': np.median(rel_diff), 'n_valid': valid_mask.sum()}
    
    if verbose:
        print(f"  pseudo_render_median: {results['pseudo_render_median']:.3f}m")
        print(f"  projected_median: {results['projected_median']:.3f}m, coverage: {results['projected_coverage']:.2f}%")
        if 'pseudo_anchor' in results and results['pseudo_anchor']['scale']:
            print(f"  pseudo_anchor: scale={results['pseudo_anchor']['scale']:.4f}, MAE={results['pseudo_anchor']['mae']:.3f}m, MRE={results['pseudo_anchor']['mre']:.1%}")
        if 'high_conf_anchor' in results and results['high_conf_anchor']['scale']:
            print(f"  high_conf_anchor: scale={results['high_conf_anchor']['scale']:.4f}, MAE={results['high_conf_anchor']['mae']:.3f}m, MRE={results['high_conf_anchor']['mre']:.1%}")
        if 'mono_5m' in results:
            print(f"  mono_5m: scale={results['mono_5m']['scale']:.4f}, MAE={results['mono_5m']['mae']:.3f}m, MRE={results['mono_5m']['mre']:.1%}")
    
    return results

def main():
    base_dir = Path('/data3/bzhang512/part3_online_mapping_experiments/D4_gn_scale/brpo_debug')
    
    all_results = {}
    
    for event_dir in sorted(base_dir.glob('event_kf_*')):
        event_name = event_dir.name
        print(f"\n=== {event_name} ===")
        
        for frame_dir in sorted(event_dir.glob('frame_*')):
            frame_name = frame_dir.name
            print(f"\n{frame_name}:")
            
            result = analyze_frame(frame_dir, verbose=True)
            if result:
                all_results[f"{event_name}/{frame_name}"] = result
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    scales = []
    mres = []
    coverages = []
    
    for key, res in all_results.items():
        if 'pseudo_anchor' in res and res['pseudo_anchor']['scale']:
            scales.append(res['pseudo_anchor']['scale'])
            mres.append(res['pseudo_anchor']['mre'])
            coverages.append(res['projected_coverage'])
    
    if scales:
        print(f"\nPseudo Render Anchor (n={len(scales)}):")
        print(f"  Scale: median={np.median(scales):.4f}, std={np.std(scales):.4f}, range=[{np.min(scales):.4f}, {np.max(scales):.4f}]")
        print(f"  MRE: median={np.median(mres):.1%}, mean={np.mean(mres):.1%}")
        print(f"  Coverage: median={np.median(coverages):.2f}%, range=[{np.min(coverages):.2f}%, {np.max(coverages):.2f}%]")
    
    # High confidence summary
    hc_scales = []
    hc_mres = []
    hc_n_valids = []
    
    for key, res in all_results.items():
        if 'high_conf_anchor' in res and res['high_conf_anchor']['scale']:
            hc_scales.append(res['high_conf_anchor']['scale'])
            hc_mres.append(res['high_conf_anchor']['mre'])
            hc_n_valids.append(res['high_conf_anchor']['n_valid'])
    
    if hc_scales:
        print(f"\nHigh Confidence Anchor (n={len(hc_scales)}):")
        print(f"  Scale: median={np.median(hc_scales):.4f}, std={np.std(hc_scales):.4f}")
        print(f"  MRE: median={np.median(hc_mres):.1%}, mean={np.mean(hc_mres):.1%}")
        print(f"  N valid pixels: median={np.median(hc_n_valids):.0f}, range=[{np.min(hc_n_valids):.0f}, {np.max(hc_n_valids):.0f}]")

if __name__ == '__main__':
    main()
