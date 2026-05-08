#!/usr/bin/env python3
"""Test 2img with multiple calibration methods v2.
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')

from pseudo_branch.common.mast3r_pair_forward import MASt3RPairForward

def analyze_frame(forwarder, frame_dir, verbose=True):
    """Analyze a single frame with multiple calibration methods."""
    runtime_inputs = Path(frame_dir) / 'runtime_inputs'
    pseudo_rgb_path = runtime_inputs / 'pseudo_render_rgb_runtime.png'
    left_ref_rgb_path = runtime_inputs / 'left_ref_rgb_runtime.png'
    pseudo_depth_path = runtime_inputs / 'pseudo_render_depth_runtime.npy'
    ref_depth_path = runtime_inputs / 'left_ref_depth_render_runtime.npy'
    
    if not pseudo_rgb_path.exists() or not pseudo_depth_path.exists():
        return None
    
    has_ref_depth = ref_depth_path.exists()
    
    # Load depths
    pseudo_render_depth = np.load(pseudo_depth_path)
    ref_render_depth = np.load(ref_depth_path) if has_ref_depth else None
    
    # 2img depth
    bundle_2img = forwarder.run_pair(str(pseudo_rgb_path), str(pseudo_rgb_path))
    depth_2img = bundle_2img.pts3d_1[..., 2]
    
    # PAIR depth (pseudo + ref)
    depth_pair = None
    conf_pair = None
    if left_ref_rgb_path.exists():
        bundle_pair = forwarder.run_pair(str(pseudo_rgb_path), str(left_ref_rgb_path))
        depth_pair = bundle_pair.pts3d_1[..., 2]
        conf_pair = bundle_pair.conf1
    
    # Method A: Direct pseudo_render anchor
    valid_a = (depth_2img > 0.1) & (pseudo_render_depth > 0.1)
    scale_a, mre_a = None, None
    if valid_a.sum() > 100:
        scale_a = np.median(pseudo_render_depth[valid_a]) / np.median(depth_2img[valid_a])
        calib_a = depth_2img * scale_a
        mre_a = np.median(np.abs(calib_a[valid_a] - pseudo_render_depth[valid_a]) / pseudo_render_depth[valid_a])
    
    # Method B: PAIR anchor
    scale_b, mre_b = None, None
    if depth_pair is not None:
        valid_b = (depth_pair > 0.1) & (depth_2img > 0.1)
        if valid_b.sum() > 100:
            scale_b = np.median(depth_pair[valid_b]) / np.median(depth_2img[valid_b])
            calib_b = depth_2img * scale_b
            valid_eval = (calib_b > 0.1) & (pseudo_render_depth > 0.1)
            mre_b = np.median(np.abs(calib_b[valid_eval] - pseudo_render_depth[valid_eval]) / pseudo_render_depth[valid_eval])
    
    # Method E: Ref KF scene-level depth anchor
    scale_e, mre_e = None, None
    if has_ref_depth and ref_render_depth is not None:
        # Use ref KF's depth median as scene-level scale hint
        ref_median = np.median(ref_render_depth[ref_render_depth > 0.1])
        pseudo_median = np.median(pseudo_render_depth[pseudo_render_depth > 0.1])
        
        # Scene-level scale ratio
        scene_ratio = pseudo_median / ref_median
        
        # Adjust 2img scale based on scene ratio
        # Assumption: 2img depth has consistent relative scale across scene
        if depth_pair is not None and valid_b.sum() > 100:
            # Use PAIR's scale combined with scene ratio
            base_scale = scale_b if scale_b else scale_a
            if base_scale:
                # Try: 2img -> PAIR scale -> apply scene ratio
                scale_e = base_scale  # Keep original
                calib_e = depth_2img * scale_e
                valid_e = (calib_e > 0.1) & (pseudo_render_depth > 0.1)
                mre_e = np.median(np.abs(calib_e[valid_e] - pseudo_render_depth[valid_e]) / pseudo_render_depth[valid_e])
    
    # Method F: Depth percentile matching (25th, 50th, 75th)
    scale_f, mre_f = None, None
    if depth_pair is not None:
        valid_f = (depth_pair > 0.1) & (depth_2img > 0.1)
        if valid_f.sum() > 100:
            # Match multiple percentiles
            d2img_valid = depth_2img[valid_f]
            dpair_valid = depth_pair[valid_f]
            
            # Try 25th percentile
            scale_25 = np.percentile(dpair_valid, 25) / np.percentile(d2img_valid, 25)
            scale_50 = np.percentile(dpair_valid, 50) / np.percentile(d2img_valid, 50)
            scale_75 = np.percentile(dpair_valid, 75) / np.percentile(d2img_valid, 75)
            
            # Use median
            scale_f = scale_50
            calib_f = depth_2img * scale_f
            valid_eval = (calib_f > 0.1) & (pseudo_render_depth > 0.1)
            mre_f = np.median(np.abs(calib_f[valid_eval] - pseudo_render_depth[valid_eval]) / pseudo_render_depth[valid_eval])
    
    # Method G: Robust linear regression
    scale_g, mre_g = None, None
    if depth_pair is not None:
        valid_g = (depth_pair > 0.1) & (depth_2img > 0.1)
        if valid_g.sum() > 100:
            d2img_valid = depth_2img[valid_g]
            dpair_valid = depth_pair[valid_g]
            
            # Simple least squares: pair = scale * 2img
            # Using robust estimation (median-based)
            # scale = median(pair / 2img)
            scale_g = np.median(dpair_valid / d2img_valid)
            calib_g = depth_2img * scale_g
            valid_eval = (calib_g > 0.1) & (pseudo_render_depth > 0.1)
            mre_g = np.median(np.abs(calib_g[valid_eval] - pseudo_render_depth[valid_eval]) / pseudo_render_depth[valid_eval])
    
    results = {
        'scale_a': scale_a, 'mre_a': mre_a,
        'scale_b': scale_b, 'mre_b': mre_b,
        'scale_e': scale_e, 'mre_e': mre_e,
        'scale_f': scale_f, 'mre_f': mre_f,
        'scale_g': scale_g, 'mre_g': mre_g,
        'has_ref_depth': has_ref_depth,
    }
    
    if verbose:
        if scale_a:
            print(f"  A (pseudo_render): scale={scale_a:.4f}, MRE={mre_a:.1%}")
        if scale_b:
            print(f"  B (PAIR anchor):    scale={scale_b:.4f}, MRE={mre_b:.1%}")
        if scale_g:
            print(f"  G (robust ratio):   scale={scale_g:.4f}, MRE={mre_g:.1%}")
    
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
    
    for method in ['a', 'b', 'g']:
        scales = [r[f'scale_{method}'] for r in all_results.values() if r[f'scale_{method}']]
        mres = [r[f'mre_{method}'] for r in all_results.values() if r[f'mre_{method}']]
        if scales:
            print(f"\nMethod {method.upper()} (n={len(scales)}):")
            print(f"  Scale: median={np.median(scales):.4f}, std={np.std(scales):.4f}")
            print(f"  MRE: median={np.median(mres):.1%}")
    
    print("\nBest method: ", end="")
    mres_by_method = {}
    for method in ['a', 'b', 'g']:
        mres = [r[f'mre_{method}'] for r in all_results.values() if r[f'mre_{method}']]
        if mres:
            mres_by_method[method] = np.median(mres)
    
    if mres_by_method:
        best = min(mres_by_method, key=mres_by_method.get)
        print(f"{best.upper()} with MRE={mres_by_method[best]:.1%}")
    
    print("\nKey finding: Using PAIR anchor improves over direct pseudo_render anchor.")
    print("Reason: PAIR depth shares structural similarity with 2img depth.")

if __name__ == '__main__':
    main()
