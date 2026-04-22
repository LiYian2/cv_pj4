#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build pseudo cache with EDP and full diagnostics."""
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Tuple
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_depth(path, depth_scale=5000.0):
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    elif path.suffix in (".png", ".jpg"):
        img = np.array(Image.open(path))
        if img.dtype == np.uint16:
            return img.astype(np.float32) / depth_scale
        return img.astype(np.float32) / 255.0
    return None


def get_intrinsic_matrix(intr, height=512, width=512):
    fx = intr.get("fx", 500)
    fy = intr.get("fy", fx)
    cx = intr.get("cx", width / 2)
    cy = intr.get("cy", height / 2)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def reproject_depth(depth_src, K_src, K_dst, pose_src_c2w, pose_dst_c2w, height=512, width=512):
    if depth_src is None:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)
    w2c_dst = np.linalg.inv(pose_dst_c2w)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    depth_flat = depth_src.reshape(-1)
    valid_src = depth_flat > 0.001
    rays = (np.linalg.inv(K_src) @ uv.T).T
    pts_cam = rays * depth_flat[:, None]
    pts_world = (pose_src_c2w[:3, :3] @ pts_cam.T + pose_src_c2w[:3, 3:4]).T
    pts_dst = (w2c_dst[:3, :3] @ pts_world.T + w2c_dst[:3, 3:4]).T
    pts_2d = (K_dst @ pts_dst.T).T
    z = pts_2d[:, 2]
    valid_z = z > 0.01
    u_dst = pts_2d[:, 0] / (z + 1e-8)
    v_dst = pts_2d[:, 1] / (z + 1e-8)
    valid_u = (u_dst >= 0) & (u_dst < width)
    valid_v = (v_dst >= 0) & (v_dst < height)
    valid = valid_src & valid_z & valid_u & valid_v
    depth_reproj = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    if valid.any():
        u_idx = np.clip(np.round(u_dst[valid]).astype(int), 0, width-1)
        v_idx = np.clip(np.round(v_dst[valid]).astype(int), 0, height-1)
        z_vals = z[valid]
        for i in range(len(u_idx)):
            ui, vi = u_idx[i], v_idx[i]
            if valid_mask[vi, ui]:
                if z_vals[i] < depth_reproj[vi, ui]:
                    depth_reproj[vi, ui] = z_vals[i]
            else:
                depth_reproj[vi, ui] = z_vals[i]
                valid_mask[vi, ui] = True
    return depth_reproj, valid_mask


def find_ref_depth_path(sparse_depth_dir, frame_id):
    patterns = ["{fid:06d}.png", "{fid:05d}.png", "frame_{fid:05d}.png".format(fid=frame_id+1)]
    for pat in patterns:
        p = sparse_depth_dir / pat.format(fid=frame_id)
        if p.exists():
            return p
    return None


def find_ref_rgb_path(ref_rgb_dir, frame_id, scene_name):
    if scene_name == "DL3DV-2":
        patterns = ["frame_{fid:05d}.png".format(fid=frame_id+1)]
    elif scene_name == "Re10k-1":
        patterns = ["{fid:05d}.png".format(fid=frame_id)]
    else:
        patterns = ["{fid:06d}.png".format(fid=frame_id)]
    for pat in patterns:
        p = ref_rgb_dir / pat
        if p.exists():
            return p
    return None


def write_diag_gt(sample_dir, target_depth, confidence, render_depth, stats, sparse_rgb_dir, refs, scene_name):
    """Write diag files for GT method (simpler, no flow/epipolar)."""
    from pseudo_branch.common.diag_writer import write_depth_consistency_map
    
    sample_dir = Path(sample_dir)
    diag_dir = sample_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic outputs
    np.save(sample_dir / "target_depth.npy", target_depth)
    np.save(sample_dir / "confidence_mask.npy", confidence)
    conf_img = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(conf_img).save(sample_dir / "confidence_mask.png")
    
    # Validity mask
    validity = (target_depth > 0).astype(np.float32)
    Image.fromarray((validity * 255).astype(np.uint8)).save(diag_dir / "validity_mask.png")
    
    # Depth consistency map
    write_depth_consistency_map(diag_dir, render_depth, target_depth, sigma=0.1)
    
    # For GT method, create placeholder images for epipolar/flow
    H, W = render_depth.shape
    placeholder = np.zeros((H, W), dtype=np.uint8)
    Image.fromarray(placeholder).save(diag_dir / "epipolar_distance.png")
    Image.fromarray(placeholder).save(diag_dir / "flow_visualization.png")
    
    # Score
    with open(diag_dir / "score.json", "w") as f:
        json.dump(stats, f, indent=2)


def build_sample_gt_depth(sample_dir, sparse_depth_dir, sparse_rgb_dir, scene_name, depth_threshold, dry_run):
    camera_path = sample_dir / "camera.json"
    refs_path = sample_dir / "refs.json"
    render_depth_path = sample_dir / "render_depth.npy"
    
    if not camera_path.exists() or not render_depth_path.exists():
        return {"error": "missing required files"}
    
    with open(camera_path) as f:
        camera = json.load(f)
    with open(refs_path) as f:
        refs = json.load(f)
    
    frame_id = camera["frame_id"]
    left_fid = refs["left_ref_frame_id"]
    right_fid = refs["right_ref_frame_id"]
    
    left_ref_depth = find_ref_depth_path(sparse_depth_dir, left_fid)
    right_ref_depth = find_ref_depth_path(sparse_depth_dir, right_fid)
    
    render_depth = load_depth(render_depth_path)
    H, W = render_depth.shape
    
    K_pseudo = get_intrinsic_matrix(camera.get("intrinsics_px", {}), H, W)
    pose_pseudo = np.array(camera["pose_c2w"], dtype=np.float32)
    
    reproj_depths = []
    valid_masks = []
    
    left_depth = load_depth(left_ref_depth) if left_ref_depth else None
    right_depth = load_depth(right_ref_depth) if right_ref_depth else None
    
    if left_depth is not None:
        pose_left = np.array(refs["left_ref_pose"], dtype=np.float32)
        dr, vm = reproject_depth(left_depth, K_pseudo, K_pseudo, pose_left, pose_pseudo, H, W)
        reproj_depths.append(dr)
        valid_masks.append(vm)
    
    if right_depth is not None:
        pose_right = np.array(refs["right_ref_pose"], dtype=np.float32)
        dr, vm = reproject_depth(right_depth, K_pseudo, K_pseudo, pose_right, pose_pseudo, H, W)
        reproj_depths.append(dr)
        valid_masks.append(vm)
    
    target_depth = np.zeros_like(render_depth)
    consistency_count = np.zeros_like(render_depth, dtype=np.int32)
    
    for reproj_d, valid in zip(reproj_depths, valid_masks):
        diff = np.abs(render_depth - reproj_d)
        rel_diff = diff / (render_depth + 1e-3)
        consistent = valid & (rel_diff < depth_threshold) & (render_depth > 0)
        target_depth = np.where(consistent & (target_depth == 0), render_depth, target_depth)
        consistency_count += consistent.astype(np.int32)
    
    valid_final = consistency_count >= 1
    target_depth = np.where(valid_final, target_depth, 0)
    confidence = (target_depth > 0).astype(np.float32)
    
    stats = {
        "frame_id": frame_id,
        "method": "gt_depth_reprojection",
        "total_pixels": H * W,
        "render_valid": int((render_depth > 0).sum()),
        "target_valid": int((target_depth > 0).sum()),
        "consistency_ratio": float((target_depth > 0).sum() / max((render_depth > 0).sum(), 1)),
        "has_sparse_depth": left_ref_depth is not None or right_ref_depth is not None,
    }
    
    if not dry_run:
        write_diag_gt(sample_dir, target_depth, confidence, render_depth, stats, sparse_rgb_dir, refs, scene_name)
    
    return stats


def build_sample_edp(sample_dir, sparse_rgb_dir, scene_name, flow_matcher, dry_run, size=512):
    from pseudo_branch.common.epipolar_depth import compute_edp_depth
    from pseudo_branch.common.diag_writer import write_full_diag
    
    camera_path = sample_dir / "camera.json"
    refs_path = sample_dir / "refs.json"
    render_depth_path = sample_dir / "render_depth.npy"
    render_rgb_path = sample_dir / "render_rgb.png"
    
    if not camera_path.exists() or not render_rgb_path.exists():
        return {"error": "missing required files"}
    
    with open(camera_path) as f:
        camera = json.load(f)
    with open(refs_path) as f:
        refs = json.load(f)
    
    frame_id = camera["frame_id"]
    left_fid = refs["left_ref_frame_id"]
    right_fid = refs["right_ref_frame_id"]
    
    left_rgb = find_ref_rgb_path(sparse_rgb_dir, left_fid, scene_name)
    right_rgb = find_ref_rgb_path(sparse_rgb_dir, right_fid, scene_name)
    
    left_camera = {
        "pose_c2w": refs["left_ref_pose"],
        "intrinsics_px": camera["intrinsics_px"],
        "image_size": camera["image_size"]
    }
    right_camera = {
        "pose_c2w": refs["right_ref_pose"],
        "intrinsics_px": camera["intrinsics_px"],
        "image_size": camera["image_size"]
    }
    
    depth_left, conf_left, stats_left, diag_left = compute_edp_depth(
        str(render_rgb_path), camera,
        str(left_rgb) if left_rgb else None, left_camera,
        flow_matcher, size=size
    )
    
    depth_right, conf_right, stats_right, diag_right = compute_edp_depth(
        str(render_rgb_path), camera,
        str(right_rgb) if right_rgb else None, right_camera,
        flow_matcher, size=size
    )
    
    target_depth = np.where(conf_left >= conf_right, depth_left, depth_right)
    confidence = np.maximum(conf_left, conf_right)
    
    render_depth = load_depth(render_depth_path) if render_depth_path.exists() else np.zeros((size, size), dtype=np.float32)
    
    stats = {
        "frame_id": frame_id,
        "method": "edp",
        "total_pixels": size * size,
        "target_valid": int((target_depth > 0).sum()),
        "left": stats_left,
        "right": stats_right,
    }
    
    diag_extra = diag_left if diag_left else diag_right
    
    if not dry_run:
        write_full_diag(sample_dir, target_depth, confidence, render_depth, stats, diag_extra)
        
        # --- Fusion step ---
        np.save(sample_dir / "depth_left.npy", depth_left)
        np.save(sample_dir / "conf_left.npy", conf_left)
        np.save(sample_dir / "depth_right.npy", depth_right)
        np.save(sample_dir / "conf_right.npy", conf_right)
        
        # RGB fusion
        from pseudo_branch.observation.pseudo_fusion import run_fusion_for_sample, get_fusion_diag_images
        target_rgb_left_path = sample_dir / "target_rgb_left.png"
        target_rgb_right_path = sample_dir / "target_rgb_right.png"
        render_rgb_path = sample_dir / "render_rgb.png"
        
        fusion_stats, I_r, I_L, I_R, W_L, W_R, A, C_fused = run_fusion_for_sample(
            str(render_rgb_path), str(target_rgb_left_path), str(target_rgb_right_path),
            depth_left, conf_left, depth_right, conf_right, render_depth,
            str(sample_dir),
            use_depth_agreement=False
        )
        
        # Fusion diagnostic images
        diag_dir = sample_dir / "diag"
        diag_dir.mkdir(exist_ok=True)
        fusion_diag = get_fusion_diag_images(I_L, I_R, W_L, W_R, A, C_fused, conf_left, conf_right)
        for name, img in fusion_diag.items():
            from PIL import Image
            Image.fromarray(img).save(diag_dir / (name + ".png"))
        
        stats["fusion"] = fusion_stats
    
    return stats


def main():
    args = parse_args()
    cache_root = Path(args.pseudo_cache_root)
    sparse_depth_dir = Path(args.sparse_depth_dir) if args.sparse_depth_dir else None
    sparse_rgb_dir = Path(args.sparse_rgb_dir) if args.sparse_rgb_dir else None
    
    manifest_path = cache_root / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    scene = manifest.get("scene_name", "unknown")
    n = manifest.get("num_samples", 0)
    size = manifest.get("image_size", {}).get("width", 512)
    
    print("Building for " + scene + ", " + str(n) + " samples")
    
    use_edp = scene in ["Re10k-1", "DL3DV-2"]
    
    flow_matcher = None
    if use_edp:
        from pseudo_branch.common.flow_matcher import FlowMatcher
        flow_matcher = FlowMatcher()
        print("Using EDP")
    else:
        print("Using GT depth reproject")
    
    all_stats = []
    for sample_id in manifest["sample_ids"]:
        sample_dir = cache_root / "samples" / str(sample_id)
        
        if use_edp:
            stats = build_sample_edp(sample_dir, sparse_rgb_dir, scene, flow_matcher, args.dry_run, size)
        else:
            stats = build_sample_gt_depth(sample_dir, sparse_depth_dir, sparse_rgb_dir, scene, args.depth_threshold, args.dry_run)
        
        all_stats.append(stats)
        valid = stats.get("target_valid", 0)
        method = stats.get("method", "unknown")
        err = stats.get("error", "")
        print("  sample " + str(sample_id) + ": valid=" + str(valid) + " method=" + method + ("" if not err else " ERROR=" + err))
    
    if not args.dry_run:
        for s in manifest["samples"]:
            sid = s["sample_id"]
            s["target_depth_path"] = "samples/" + str(sid) + "/target_depth.npy"
            s["confidence_mask_path"] = "samples/" + str(sid) + "/confidence_mask.npy"
            s["depth_left_path"] = "samples/" + str(sid) + "/depth_left.npy"
            s["depth_right_path"] = "samples/" + str(sid) + "/depth_right.npy"
            s["conf_left_path"] = "samples/" + str(sid) + "/conf_left.npy"
            s["conf_right_path"] = "samples/" + str(sid) + "/conf_right.npy"
            s["target_rgb_fused_path"] = "samples/" + str(sid) + "/target_rgb_fused.png"
            s["confidence_mask_fused_path"] = "samples/" + str(sid) + "/confidence_mask_fused.npy"
            s["fusion_meta_path"] = "samples/" + str(sid) + "/fusion_meta.json"
        manifest["additional_built"] = True
        manifest["depth_method"] = "edp" if use_edp else "gt_reprojection"
        manifest["has_fusion"] = True
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    total_valid = sum(s.get("target_valid", 0) for s in all_stats)
    errors = [s for s in all_stats if "error" in s]
    print("Done. Total valid: " + str(total_valid) + ", Errors: " + str(len(errors)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pseudo-cache-root", required=True)
    p.add_argument("--sparse-depth-dir", default="")
    p.add_argument("--sparse-rgb-dir", default="")
    p.add_argument("--depth-threshold", type=float, default=0.3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
