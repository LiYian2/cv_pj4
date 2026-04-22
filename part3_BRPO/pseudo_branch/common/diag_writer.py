# -*- coding: utf-8 -*-
"""Write diagnostic outputs for pseudo cache."""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

def write_sample_outputs(sample_dir, target_depth, confidence, render_depth, stats):
    """Write all sample outputs."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(sample_dir / "target_depth.npy", target_depth)
    np.save(sample_dir / "confidence_mask.npy", confidence)
    conf_img = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(conf_img).save(sample_dir / "confidence_mask.png")
    
    diag_dir = sample_dir / "diag"
    diag_dir.mkdir(exist_ok=True)
    
    validity = (target_depth > 0).astype(np.float32)
    validity_img = (validity * 255).astype(np.uint8)
    Image.fromarray(validity_img).save(diag_dir / "validity_mask.png")
    
    with open(diag_dir / "score.json", "w") as f:
        json.dump(stats, f, indent=2)


def write_depth_consistency_map(diag_dir, render_depth, target_depth, sigma=0.1):
    """
    Compute and save depth consistency visualization.
    
    consistency = exp(-|D_render - D_target| / (D_render * sigma))
    """
    diag_dir = Path(diag_dir)
    diag_dir.mkdir(exist_ok=True)
    
    # Compute consistency
    valid = (render_depth > 0) & (target_depth > 0)
    consistency = np.zeros_like(render_depth)
    
    if valid.any():
        rel_diff = np.abs(render_depth - target_depth) / (render_depth + 1e-3)
        consistency[valid] = np.exp(-rel_diff[valid] / sigma)
    
    # Save
    consistency_img = (np.clip(consistency, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(consistency_img).save(diag_dir / "depth_consistency_map.png")
    
    return consistency


def write_epipolar_distance_map(diag_dir, H, W, pts_pseudo, epipolar_distances, confidence):
    """
    Visualize epipolar distance as heatmap.
    """
    diag_dir = Path(diag_dir)
    diag_dir.mkdir(exist_ok=True)
    
    # Create distance map
    dist_map = np.zeros((H, W), dtype=np.float32)
    
    for pt, dist, conf in zip(pts_pseudo, epipolar_distances, confidence):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < W and 0 <= y < H:
            dist_map[y, x] = max(dist_map[y, x], dist)
    
    # Normalize and visualize
    if dist_map.max() > 0:
        dist_norm = dist_map / (dist_map.max() + 1e-8)
    else:
        dist_norm = dist_map
    
    dist_img = (dist_norm * 255).astype(np.uint8)
    Image.fromarray(dist_img).save(diag_dir / "epipolar_distance.png")
    
    return dist_map


def write_flow_visualization(diag_dir, img1_path, img2_path, pts1, pts2, confidence, sample_id):
    """
    Create flow visualization image showing matches.
    """
    diag_dir = Path(diag_dir)
    diag_dir.mkdir(exist_ok=True)
    
    try:
        img1 = np.array(Image.open(img1_path).convert("RGB"))
        img2 = np.array(Image.open(img2_path).convert("RGB"))
        
        H, W = img1.shape[:2]
        
        # Create side-by-side visualization
        vis = np.zeros((H, W * 2, 3), dtype=np.uint8)
        vis[:, :W] = img1
        vis[:, W:] = img2
        
        # Draw matches (subsample for clarity)
        n_matches = len(pts1)
        step = max(1, n_matches // 200)
        
        for i in range(0, n_matches, step):
            x1, y1 = int(pts1[i][0]), int(pts1[i][1])
            x2, y2 = int(pts2[i][0]) + W, int(pts2[i][1])
            
            if 0 <= x1 < W and 0 <= y1 < H and W <= x2 < 2*W and 0 <= y2 < H:
                # Color by confidence
                color = (0, int(min(confidence[i] / 100, 1) * 255), int(max(255 - confidence[i]/100 * 255, 0)))
                # Draw line
                for t in np.linspace(0, 1, 20):
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    if 0 <= x < 2*W and 0 <= y < H:
                        vis[y, x] = color
        
        Image.fromarray(vis).save(diag_dir / "flow_visualization.png")
        
    except Exception as e:
        print(f"  Warning: could not create flow visualization: {e}")


def write_full_diag(
    sample_dir,
    target_depth,
    confidence,
    render_depth,
    stats,
    diag_extra=None
):
    """
    Write all diagnostic outputs including extra visualizations.
    
    Args:
        sample_dir: sample directory path
        target_depth: (H, W) target depth
        confidence: (H, W) confidence mask
        render_depth: (H, W) render depth
        stats: dict with statistics
        diag_extra: optional dict with:
            - epipolar_distances: per-match distances
            - pts_pseudo: matched points in pseudo view
            - pts_ref: matched points in ref view
            - match_confidence: match confidence scores
            - img1_path: pseudo RGB path
            - img2_path: ref RGB path
    """
    sample_dir = Path(sample_dir)
    diag_dir = sample_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    # Write basic outputs
    np.save(sample_dir / "target_depth.npy", target_depth)
    np.save(sample_dir / "confidence_mask.npy", confidence)
    conf_img = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(conf_img).save(sample_dir / "confidence_mask.png")
    
    # Validity mask
    validity = (target_depth > 0).astype(np.float32)
    validity_img = (validity * 255).astype(np.uint8)
    Image.fromarray(validity_img).save(diag_dir / "validity_mask.png")
    
    # Depth consistency map
    write_depth_consistency_map(diag_dir, render_depth, target_depth, sigma=0.1)
    
    # Extra visualizations
    if diag_extra:
        H, W = render_depth.shape
        
        # Epipolar distance map
        if "epipolar_distances" in diag_extra and "pts_pseudo" in diag_extra:
            write_epipolar_distance_map(
                diag_dir, H, W,
                diag_extra["pts_pseudo"],
                diag_extra["epipolar_distances"],
                diag_extra.get("match_confidence", np.ones(len(diag_extra["pts_pseudo"])))
            )
        
        # Flow visualization
        if all(k in diag_extra for k in ["pts_pseudo", "pts_ref", "match_confidence", "img1_path", "img2_path"]):
            write_flow_visualization(
                diag_dir,
                diag_extra["img1_path"],
                diag_extra["img2_path"],
                diag_extra["pts_pseudo"],
                diag_extra["pts_ref"],
                diag_extra["match_confidence"],
                stats.get("frame_id", 0)
            )
    
    # Score
    with open(diag_dir / "score.json", "w") as f:
        json.dump(stats, f, indent=2)
