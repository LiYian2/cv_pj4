#!/usr/bin/env python3
"""Sidecar materialize script: Generate 2IMG+PAIR-proxy depth on existing brpo_debug.

This script does NOT modify signal_v2. It only generates new depth materials
for verification before integration into runtime_exact_backend.

Usage:
    python scripts/materialize_twoimg_pair_proxy_depth.py         --brpo_debug_root /data3/bzhang512/part3_online_mapping_experiments/E5a_jointprimary_maskedcolor_rgbonly_cm_difix/brpo_debug         --output_suffix twoimg_pair_proxy_v1

Output structure (per frame):
    frame_XXXX/twoimg_pair_proxy_v1/
        depth_2img_raw.npy
        depth_pair_anchor.npy
        depth_calibrated.npy
        depth_effective.npy
        cm_mask.npy
        scale_by_range.json
        metadata.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')

from pseudo_branch.common.mast3r_pair_forward import get_shared_mast3r_pair_forward
from pseudo_branch.common.twoimg_pair_proxy_depth import (
    build_twoimg_pair_proxy_depth,
    DEFAULT_DEPTH_RANGES,
    TwoImgPairProxyDepthResult,
)


def find_frame_dirs(brpo_debug_root: Path) -> List[Path]:
    """Find all frame directories under brpo_debug.
    
    Structure: event_kf_XXXX/frame_YYYY/
    """
    frame_dirs = []
    for event_dir in brpo_debug_root.iterdir():
        if event_dir.is_dir() and event_dir.name.startswith('event_kf_'):
            for frame_dir in event_dir.iterdir():
                if frame_dir.is_dir() and frame_dir.name.startswith('frame_'):
                    frame_dirs.append(frame_dir)
    return sorted(frame_dirs)


def load_exact_backend_anchor(frame_dir: Path, branch: str = 'left') -> np.ndarray:
    """Load PAIR projected depth from exact_backend_v1 as scale anchor.
    
    Args:
        frame_dir: Path to frame_XXXX directory
        branch: 'left' or 'right' (use left by default)
        
    Returns:
        projected_depth: (H, W) metric depth from PAIR projected
    """
    exact_dir = frame_dir / 'exact_backend_v1'
    if not exact_dir.exists():
        raise FileNotFoundError(f"exact_backend_v1 not found in {frame_dir}")
    
    depth_path = exact_dir / f'projected_depth_{branch}_exact.npy'
    if not depth_path.exists():
        raise FileNotFoundError(f"{depth_path} not found")
    
    return np.load(depth_path).astype(np.float32)


def load_signal_v2_cm_mask(frame_dir: Path) -> np.ndarray:
    """Load confidence mask (C_m) from signal_v2.
    
    Args:
        frame_dir: Path to frame_XXXX directory
        
    Returns:
        cm_mask: (H, W) binary mask from pseudo_confidence_exact_brpo_upstream_target_v1
    """
    signal_dir = frame_dir / 'signal_v2'
    if not signal_dir.exists():
        raise FileNotFoundError(f"signal_v2 not found in {frame_dir}")
    
    cm_path = signal_dir / 'pseudo_confidence_exact_brpo_upstream_target_v1.npy'
    if not cm_path.exists():
        raise FileNotFoundError(f"{cm_path} not found")
    
    return np.load(cm_path).astype(np.float32)


def find_pseudo_rgb_path(frame_dir: Path) -> Optional[str]:
    """Find pseudo RGB path for 2IMG MASt3R.
    
    Preference: pseudo_fused_rgb (if Difix enabled) > pseudo_render_rgb_runtime
    """
    runtime_dir = frame_dir / 'runtime_inputs'
    
    # Check for fused RGB (Difix output)
    fused_path = runtime_dir / 'pseudo_fused_rgb.png'
    if fused_path.exists():
        return str(fused_path)
    
    # Fallback to render RGB
    render_path = runtime_dir / 'pseudo_render_rgb_runtime.png'
    if render_path.exists():
        return str(render_path)
    
    # Fallback to GT RGB
    gt_path = runtime_dir / 'pseudo_gt_rgb_runtime.png'
    if gt_path.exists():
        return str(gt_path)
    
    return None


def materialize_single_frame(
    forwarder,
    frame_dir: Path,
    output_suffix: str,
    depth_ranges: List = DEFAULT_DEPTH_RANGES,
) -> Dict:
    """Materialize 2IMG+PAIR-proxy depth for a single frame.
    
    Returns summary metadata for the frame.
    """
    pseudo_rgb_path = find_pseudo_rgb_path(frame_dir)
    if pseudo_rgb_path is None:
        return {"frame": frame_dir.name, "status": "no_pseudo_rgb", "error": None}
    
    try:
        # Load inputs
        depth_pair_anchor = load_exact_backend_anchor(frame_dir, branch='left')
        cm_mask = load_signal_v2_cm_mask(frame_dir)
        
        # Run 2IMG+PAIR-proxy
        output_dir = frame_dir / output_suffix
        result = build_twoimg_pair_proxy_depth(
            forwarder=forwarder,
            pseudo_rgb_path=pseudo_rgb_path,
            depth_pair_anchor=depth_pair_anchor,
            cm_mask=cm_mask,
            depth_ranges=depth_ranges,
            output_dir=output_dir,
        )
        
        return {
            "frame": frame_dir.name,
            "status": "success",
            "pseudo_rgb": pseudo_rgb_path,
            "metadata": result.metadata,
            "scales": {f"{r[0]}-{r[1]}": s for r, s in result.scale_by_range.items()},
        }
    except Exception as e:
        return {"frame": frame_dir.name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Materialize 2IMG+PAIR-proxy depth on brpo_debug")
    parser.add_argument('--brpo_debug_root', type=str, required=True,
                        help="Path to brpo_debug directory")
    parser.add_argument('--output_suffix', type=str, default='twoimg_pair_proxy_v1',
                        help="Output directory suffix under each frame")
    parser.add_argument('--model_name', type=str, 
                        default='/home/bzhang512/CV_Project/third_party/S3PO-GS/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                        help="MASt3R model path")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for MASt3R")
    args = parser.parse_args()
    
    brpo_root = Path(args.brpo_debug_root)
    if not brpo_root.exists():
        print(f"Error: {brpo_root} does not exist")
        sys.exit(1)
    
    # Find frames
    frame_dirs = find_frame_dirs(brpo_root)
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Initialize MASt3R forwarder
    forwarder = get_shared_mast3r_pair_forward(model_name=args.model_name, device=args.device)
    
    # Process each frame
    results = []
    for frame_dir in frame_dirs:
        print(f"Processing {frame_dir.name}...")
        result = materialize_single_frame(
            forwarder=forwarder,
            frame_dir=frame_dir,
            output_suffix=args.output_suffix,
        )
        results.append(result)
        
        if result['status'] == 'success':
            meta = result['metadata']
            print(f"  cm_ratio={meta['cm_nonzero_ratio']:.3f}, ")
            print(f"  pair_coverage_in_cm={meta['projected_depth_union_ratio']:.3f}, ")
            print(f"  twoimg_coverage_in_cm={meta['twoimg_depth_effective_ratio_after_cm_cap']:.3f}")
        elif result['status'] == 'error':
            print(f"  Error: {result['error']}")
        else:
            print(f"  Skipped: {result['status']}")
    
    # Save summary
    summary_path = brpo_root / f"{args.output_suffix}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "brpo_debug_root": str(brpo_root),
            "output_suffix": args.output_suffix,
            "num_frames": len(frame_dirs),
            "num_success": sum(1 for r in results if r['status'] == 'success'),
            "frames": results,
        }, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    
    # Aggregate statistics
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        cm_ratios = [r['metadata']['cm_nonzero_ratio'] for r in successful]
        pair_coverages = [r['metadata']['projected_depth_union_ratio'] for r in successful]
        twoimg_coverages = [r['metadata']['twoimg_depth_effective_ratio_after_cm_cap'] for r in successful]
        
        print(f"\n=== Aggregate Statistics ===")
        print(f"C_m coverage: mean={np.mean(cm_ratios):.3f}, std={np.std(cm_ratios):.3f}")
        print(f"PAIR coverage in C_m: mean={np.mean(pair_coverages):.3f}, std={np.std(pair_coverages):.3f}")
        print(f"2IMG coverage in C_m: mean={np.mean(twoimg_coverages):.3f}, std={np.std(twoimg_coverages):.3f}")
        print(f"\nGap filled: {np.mean(twoimg_coverages) - np.mean(pair_coverages):.3f} (avg per frame)")


if __name__ == '__main__':
    main()
