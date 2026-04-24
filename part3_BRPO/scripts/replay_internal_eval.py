#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)
if f"{S3PO_ROOT}/gaussian_splatting" not in sys.path:
    sys.path.insert(0, f"{S3PO_ROOT}/gaussian_splatting")
from munch import munchify
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.external_eval_utils import load_gaussians_from_ply


def parse_args():
    p = argparse.ArgumentParser(description="Replay internal eval using saved internal camera states")
    p.add_argument("--internal-cache-root", required=True)
    p.add_argument("--stage-tag", choices=["before_opt", "after_opt"], required=True)
    p.add_argument("--ply-path", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--config", default=None, help="Path to S3PO config.yml; defaults to parent of internal cache root")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--save-render-rgb", action="store_true")
    p.add_argument("--save-render-depth", action="store_true")
    p.add_argument("--save-render-depth-npy", action="store_true")
    p.add_argument("--sh-degree", type=int, default=None, help="Override SH degree for loading stage PLY; default auto-infer from PLY header")
    return p.parse_args()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_uint8_rgb(image_tensor):
    return (
        image_tensor.detach().cpu().numpy().transpose((1, 2, 0)).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _save_depth_outputs(depth_tensor, frame_idx, output_dirs):
    depth_np = depth_tensor.detach().cpu().numpy()
    if "render_depth_npy" in output_dirs:
        np.save(os.path.join(output_dirs["render_depth_npy"], f"{frame_idx:04d}.npy"), depth_np)

    if "render_depth" in output_dirs:
        depth_min = float(depth_np.min())
        depth_max = float(depth_np.max())
        if depth_max > depth_min:
            depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_np)
        Image.fromarray((depth_norm * 255).astype(np.uint8)).save(
            os.path.join(output_dirs["render_depth"], f"{frame_idx:04d}.png")
        )


def create_viewpoint_from_state(state, device="cuda"):
    pose_c2w = np.array(state["pose_c2w"], dtype=np.float32)
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    T = pose_w2c[:3, 3]

    fx = float(state["fx"])
    fy = float(state["fy"])
    cx = float(state["cx"])
    cy = float(state["cy"])
    W = int(state["image_width"])
    H = int(state["image_height"])
    FoVx = float(state["FoVx"])
    FoVy = float(state["FoVy"])

    R_tensor = torch.from_numpy(R).float().to(device)
    T_tensor = torch.from_numpy(T).float().to(device)

    world_view_transform = getWorld2View2(R_tensor, T_tensor).transpose(0, 1)
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, cx=cx, cy=cy, fx=fx, fy=fy, W=W, H=H
    ).transpose(0, 1).to(device)
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0)
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    vp = SimpleNamespace()
    vp.uid = int(state.get("uid", state["frame_id"]))
    vp.R = R_tensor
    vp.T = T_tensor
    vp.FoVx = FoVx
    vp.FoVy = FoVy
    vp.image_height = H
    vp.image_width = W
    vp.original_image = None
    vp.exposure_a = torch.tensor(0.0, device=device)
    vp.exposure_b = torch.tensor(0.0, device=device)
    vp.world_view_transform = world_view_transform
    vp.projection_matrix = projection_matrix
    vp.full_proj_transform = full_proj_transform
    vp.camera_center = camera_center
    vp.cam_rot_delta = torch.zeros(3, device=device)
    vp.cam_trans_delta = torch.zeros(3, device=device)
    vp.fx = fx
    vp.fy = fy
    vp.cx = cx
    vp.cy = cy
    return vp


def prepare_output_dirs(save_dir, save_render_rgb, save_render_depth, save_render_depth_npy):
    mkdir_p(save_dir)
    output_dirs = {}
    if save_render_rgb:
        output_dirs["render_rgb"] = os.path.join(save_dir, "render_rgb")
        mkdir_p(output_dirs["render_rgb"])
    if save_render_depth:
        output_dirs["render_depth"] = os.path.join(save_dir, "render_depth")
        mkdir_p(output_dirs["render_depth"])
    if save_render_depth_npy:
        output_dirs["render_depth_npy"] = os.path.join(save_dir, "render_depth_npy")
        mkdir_p(output_dirs["render_depth_npy"])
    return output_dirs


def run_replay_eval(states_by_id, frame_ids, dataset, gaussians, pipe, background, save_dir, save_render_rgb, save_render_depth, save_render_depth_npy):
    output_dirs = prepare_output_dirs(save_dir, save_render_rgb, save_render_depth, save_render_depth_npy)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")

    frame_metrics = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for idx in frame_ids:
        state = states_by_id[int(idx)]
        camera = create_viewpoint_from_state(state)
        gt_image, _, _, _ = dataset[int(idx)]
        render_pkg = render(camera, gaussians, pipe, background)
        pred_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        depth_map = render_pkg["depth"].squeeze()

        if pred_image.shape != gt_image.shape:
            raise ValueError(
                f"Replay image shape mismatch for frame {int(idx):04d}: pred={tuple(pred_image.shape)} gt={tuple(gt_image.shape)} image_path={state.get('image_path')} config_source={config.get('model_params', {}).get('source_path', '')}. Rebuild the internal cache with a dataset/config that matches the replay resolution."
            )

        valid_mask = gt_image > 0
        if torch.count_nonzero(valid_mask) > 0:
            psnr_score = psnr(pred_image[valid_mask].unsqueeze(0), gt_image[valid_mask].unsqueeze(0))
        else:
            psnr_score = psnr(pred_image.unsqueeze(0), gt_image.unsqueeze(0))
        ssim_score = ssim(pred_image.unsqueeze(0), gt_image.unsqueeze(0))
        lpips_score = lpips_metric(pred_image.unsqueeze(0), gt_image.unsqueeze(0))

        psnr_value = float(psnr_score.item())
        ssim_value = float(ssim_score.item())
        lpips_value = float(lpips_score.item())

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
        frame_metrics.append({
            "frame_idx": int(idx),
            "frame_id": f"{int(idx):04d}",
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
        })

        if "render_rgb" in output_dirs:
            Image.fromarray(_to_uint8_rgb(pred_image)).save(
                os.path.join(output_dirs["render_rgb"], f"{int(idx):04d}_pred.png")
            )
        if "render_depth" in output_dirs or "render_depth_npy" in output_dirs:
            _save_depth_outputs(depth_map, int(idx), output_dirs)

    return {
        "num_frames": len(frame_metrics),
        "avg_psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "avg_ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
        "avg_lpips": float(np.mean(lpips_values)) if lpips_values else 0.0,
        "metrics": frame_metrics,
    }


def main():
    args = parse_args()
    cache_root = Path(args.internal_cache_root)
    manifest = _load_json(cache_root / "manifest.json")
    camera_states = _load_json(cache_root / "camera_states.json")
    stage_meta = _load_json(cache_root / args.stage_tag / "stage_meta.json")
    states_by_id = {int(s["frame_id"]): s for s in camera_states}
    frame_ids = [int(x) for x in stage_meta["rendered_non_kf_frames"]]

    config_path = Path(args.config) if args.config else cache_root.parent / "config.yml"
    config = load_config(str(config_path))
    model_params = config["model_params"]
    dataset = load_dataset(model_params, model_params.get("source_path", ""), config=config)
    pipe = munchify(config["pipeline_params"])
    background = torch.tensor(manifest["background"], dtype=torch.float32, device="cuda")
    gaussians = load_gaussians_from_ply(config, args.ply_path, sh_degree=args.sh_degree)

    label = args.label or Path(args.ply_path).stem
    save_dir = Path(args.save_dir) if args.save_dir else cache_root / "replay_eval" / args.stage_tag / label
    mkdir_p(str(save_dir))

    replay_summary = run_replay_eval(
        states_by_id=states_by_id,
        frame_ids=frame_ids,
        dataset=dataset,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        save_dir=str(save_dir),
        save_render_rgb=args.save_render_rgb,
        save_render_depth=args.save_render_depth,
        save_render_depth_npy=args.save_render_depth_npy,
    )

    internal_result_path = cache_root.parent / "psnr" / args.stage_tag / "final_result.json"
    internal_summary = _load_json(internal_result_path) if internal_result_path.exists() else None
    compare = None
    if internal_summary is not None:
        compare = {
            "internal_eval": internal_summary,
            "replay_eval": {
                "avg_psnr": replay_summary["avg_psnr"],
                "avg_ssim": replay_summary["avg_ssim"],
                "avg_lpips": replay_summary["avg_lpips"],
                "num_frames": replay_summary["num_frames"],
            },
            "delta": {
                "psnr": replay_summary["avg_psnr"] - float(internal_summary["mean_psnr"]),
                "ssim": replay_summary["avg_ssim"] - float(internal_summary["mean_ssim"]),
                "lpips": replay_summary["avg_lpips"] - float(internal_summary["mean_lpips"]),
                "num_frames": replay_summary["num_frames"] - int(stage_meta["num_rendered_non_kf_frames"]),
            },
        }

    meta = {
        "internal_cache_root": str(cache_root),
        "stage_tag": args.stage_tag,
        "ply_path": args.ply_path,
        "label": label,
        "config_path": str(config_path),
        "frame_ids": frame_ids,
        "num_frames": len(frame_ids),
        "compare_to_internal": compare,
    }

    with open(save_dir / "replay_eval.json", "w", encoding="utf-8") as f:
        json.dump(replay_summary, f, indent=4)
    with open(save_dir / "replay_eval_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    compact_dir = save_dir / "psnr" / "replay_internal"
    mkdir_p(str(compact_dir))
    with open(compact_dir / "final_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "avg_psnr": replay_summary["avg_psnr"],
            "avg_ssim": replay_summary["avg_ssim"],
            "avg_lpips": replay_summary["avg_lpips"],
            "num_frames": replay_summary["num_frames"],
        }, f, indent=4)

    print(json.dumps({
        "save_dir": str(save_dir),
        "avg_psnr": replay_summary["avg_psnr"],
        "avg_ssim": replay_summary["avg_ssim"],
        "avg_lpips": replay_summary["avg_lpips"],
        "num_frames": replay_summary["num_frames"],
        "compare_to_internal": compare,
    }, indent=2))


if __name__ == "__main__":
    main()
