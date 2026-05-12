import json
import os
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.system_utils import mkdir_p


def _to_serializable(x: Any):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    return x




def _frame_items(frames):
    """Return ordered (idx, frame) pairs for list/dict camera containers."""
    if isinstance(frames, dict):
        return sorted([(int(k), v) for k, v in frames.items()], key=lambda x: x[0])
    return list(enumerate(frames))

def _camera_state(frame, idx: int, kf_set, dataset=None) -> Dict[str, Any]:
    w2c = torch.eye(4, device=frame.R.device, dtype=frame.R.dtype)
    w2c[:3, :3] = frame.R
    w2c[:3, 3] = frame.T
    c2w = torch.linalg.inv(w2c)

    record = {
        "frame_id": int(idx),
        "uid": int(getattr(frame, "uid", idx)),
        "is_keyframe": bool(idx in kf_set),
        "R": _to_serializable(frame.R),
        "T": _to_serializable(frame.T),
        "pose_c2w": _to_serializable(c2w),
        "fx": float(frame.fx),
        "fy": float(frame.fy),
        "cx": float(frame.cx),
        "cy": float(frame.cy),
        "FoVx": float(frame.FoVx),
        "FoVy": float(frame.FoVy),
        "image_height": int(frame.image_height),
        "image_width": int(frame.image_width),
        "projection_matrix": _to_serializable(frame.projection_matrix),
        "cam_rot_delta": _to_serializable(getattr(frame, "cam_rot_delta", None)),
        "cam_trans_delta": _to_serializable(getattr(frame, "cam_trans_delta", None)),
        "exposure_a": _to_serializable(getattr(frame, "exposure_a", None)),
        "exposure_b": _to_serializable(getattr(frame, "exposure_b", None)),
        "R_gt": _to_serializable(getattr(frame, "R_gt", None)),
        "T_gt": _to_serializable(getattr(frame, "T_gt", None)),
    }
    if dataset is not None and hasattr(dataset, "color_paths") and idx < len(dataset.color_paths):
        image_path = dataset.color_paths[idx]
        record["image_path"] = image_path
        record["image_name"] = os.path.basename(image_path)
    return record


def export_shared_camera_states(frames, kf_indices, dataset, cache_root, save_dir, pipeline_params, background):
    mkdir_p(cache_root)
    kf_list = [int(x) for x in kf_indices]
    kf_set = set(kf_list)
    frame_items = _frame_items(frames)
    all_ids = [int(idx) for idx, _ in frame_items]
    non_kf = [int(i) for i in all_ids if i not in kf_set]
    camera_states = [_camera_state(frame, idx, kf_set, dataset) for idx, frame in frame_items]

    with open(os.path.join(cache_root, "camera_states.json"), "w", encoding="utf-8") as f:
        json.dump(camera_states, f, indent=2)

    manifest = {
        "schema_version": "internal-eval-cache-v1",
        "camera_layout": "shared_across_before_after",
        "color_refinement_updates_pose": False,
        "camera_state_source": "frontend.cameras before color refinement; reused for after_opt because color_refinement only optimizes gaussians",
        "num_frames": int(len(frames)),
        "kf_indices": kf_list,
        "non_kf_indices": non_kf,
        "background": _to_serializable(background),
        "pipeline_params": _to_serializable(dict(pipeline_params)),
        "dataset_type": getattr(dataset, "__class__", type(dataset)).__name__,
        "dataset_path": getattr(dataset, "path", None),
        "source_save_dir": save_dir,
        "stages": {},
    }
    with open(os.path.join(cache_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _update_manifest_stage(cache_root, stage_tag: str, stage_info: Dict[str, Any]):
    manifest_path = os.path.join(cache_root, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("stages", {})[stage_tag] = stage_info
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def export_stage_snapshot(frames, gaussians, dataset, cache_root, pipe, background, kf_indices, stage_tag, metrics=None):
    stage_root = os.path.join(cache_root, stage_tag)
    rgb_dir = os.path.join(stage_root, "render_rgb")
    depth_npy_dir = os.path.join(stage_root, "render_depth_npy")
    depth_vis_dir = os.path.join(stage_root, "render_depth_vis")
    ply_dir = os.path.join(stage_root, "point_cloud")
    for d in [stage_root, rgb_dir, depth_npy_dir, depth_vis_dir, ply_dir]:
        mkdir_p(d)

    gaussians.save_ply(os.path.join(ply_dir, "point_cloud.ply"))

    kf_set = set(int(x) for x in kf_indices)
    rendered = []
    for idx, frame in _frame_items(frames):
        if int(idx) in kf_set:
            continue
        render_pkg = render(frame, gaussians, pipe, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        depth = render_pkg["depth"].squeeze()

        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        Image.fromarray(pred).save(os.path.join(rgb_dir, f"{idx}_pred.png"))

        depth_np = depth.detach().cpu().numpy()
        np.save(os.path.join(depth_npy_dir, f"{idx}_pred.npy"), depth_np)
        if float(depth_np.max()) > float(depth_np.min()):
            depth_vis = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
        Image.fromarray(depth_vis).save(os.path.join(depth_vis_dir, f"{idx}_pred.png"))
        rendered.append(int(idx))

    stage_info = {
        "stage_tag": stage_tag,
        "camera_states_file": "../camera_states.json",
        "point_cloud": "point_cloud/point_cloud.ply",
        "render_rgb_dir": "render_rgb",
        "render_depth_npy_dir": "render_depth_npy",
        "render_depth_vis_dir": "render_depth_vis",
        "rendered_non_kf_frames": rendered,
        "num_rendered_non_kf_frames": len(rendered),
        "metrics": _to_serializable(metrics),
    }
    with open(os.path.join(stage_root, "stage_meta.json"), "w", encoding="utf-8") as f:
        json.dump(stage_info, f, indent=2)
    _update_manifest_stage(cache_root, stage_tag, stage_info)


def export_internal_eval_cache(frames, gaussians, dataset, save_dir, pipe, background, kf_indices, stage_tag, metrics=None, export_camera_states=False):
    cache_root = os.path.join(save_dir, "internal_eval_cache")
    if export_camera_states or not os.path.exists(os.path.join(cache_root, "camera_states.json")):
        export_shared_camera_states(frames, kf_indices, dataset, cache_root, save_dir, pipe, background)
    export_stage_snapshot(frames, gaussians, dataset, cache_root, pipe, background, kf_indices, stage_tag, metrics=metrics)
