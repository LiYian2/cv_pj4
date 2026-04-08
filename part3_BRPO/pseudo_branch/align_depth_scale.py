#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align EDP depth scale to render_depth.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image


def align_edp_depth(target_depth, render_depth, confidence, min_overlap=100, conf_threshold=0.5):
    """
    Robust EDP depth scale alignment.
    """
    # Try with high confidence first
    overlap = (
        (target_depth > 0.1) & 
        (target_depth < 100) & 
        (render_depth > 0.1) & 
        (confidence > conf_threshold)
    )
    
    # If not enough, try with lower confidence
    if overlap.sum() < min_overlap:
        overlap = (
            (target_depth > 0.1) & 
            (target_depth < 100) & 
            (render_depth > 0.1)
        )
        used_conf_filter = False
    else:
        used_conf_filter = True
    
    if overlap.sum() < min_overlap:
        return np.zeros_like(target_depth), 1.0, False, 0
    
    # Median scale estimation
    scales = render_depth[overlap] / (target_depth[overlap] + 1e-8)
    scale = np.median(scales)
    
    # Align
    aligned = target_depth * scale
    aligned = np.clip(aligned, 0.1, 100)
    
    # Depth consistency filtering
    rel_diff = np.abs(render_depth - aligned) / (render_depth + 1e-3)
    consistent = rel_diff < 0.5
    aligned = np.where(consistent & (target_depth > 0), aligned, 0)
    
    return aligned.astype(np.float32), float(scale), True, overlap.sum()


def process_sample(sample_dir, min_overlap=100, dry_run=False):
    """Align depth for one sample."""
    sample_dir = Path(sample_dir)
    
    target_depth_path = sample_dir / "target_depth.npy"
    render_depth_path = sample_dir / "render_depth.npy"
    confidence_path = sample_dir / "confidence_mask.npy"
    
    if not all(p.exists() for p in [target_depth_path, render_depth_path, confidence_path]):
        return {"error": "missing files"}
    
    target_depth = np.load(target_depth_path)
    render_depth = np.load(render_depth_path)
    confidence = np.load(confidence_path)
    
    # Check if already aligned
    score_path = sample_dir / "diag" / "score.json"
    if score_path.exists():
        score = json.load(open(score_path))
        if "depth_scale" in score:
            return {"status": "already aligned", "scale": score["depth_scale"]}
    
    # Align
    aligned, scale, success, overlap_px = align_edp_depth(
        target_depth, render_depth, confidence, min_overlap
    )
    
    if not success:
        return {"error": "insufficient overlap", "overlap": overlap_px}
    
    # Update files
    if not dry_run:
        np.save(target_depth_path, aligned)
        
        # Update confidence mask
        new_conf = (aligned > 0).astype(np.float32)
        np.save(confidence_path, new_conf)
        Image.fromarray((new_conf * 255).astype(np.uint8)).save(
            sample_dir / "confidence_mask.png"
        )
        
        # Update score.json
        if score_path.exists():
            score = json.load(open(score_path))
            score["depth_scale"] = scale
            score["aligned_valid"] = int((aligned > 0).sum())
            with open(score_path, "w") as f:
                json.dump(score, f, indent=2)
    
    return {
        "scale": scale,
        "aligned_valid": int((aligned > 0).sum()),
        "original_valid": int((target_depth > 0).sum()),
        "overlap_used": overlap_px,
    }


def main():
    args = parse_args()
    cache_root = Path(args.pseudo_cache_root)
    
    manifest_path = cache_root / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    scene = manifest.get("scene_name", "unknown")
    depth_method = manifest.get("depth_method", "unknown")
    
    if depth_method != "edp":
        print(f"Skipping {scene}: depth_method={depth_method}")
        return
    
    print(f"Aligning {scene} ({len(manifest['sample_ids'])} samples)")
    
    results = []
    for sample_id in manifest["sample_ids"]:
        sample_dir = cache_root / "samples" / str(sample_id)
        result = process_sample(sample_dir, args.min_overlap, args.dry_run)
        results.append(result)
        
        status = "OK" if "scale" in result else "SKIP"
        scale = result.get("scale", 0)
        overlap = result.get("overlap_used", 0)
        print(f"  sample {sample_id}: {status}, scale={scale:.4f}, overlap={overlap}")
    
    aligned = sum(1 for r in results if "scale" in r)
    print(f"\nAligned: {aligned}/{len(results)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pseudo-cache-root", required=True)
    p.add_argument("--min-overlap", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
