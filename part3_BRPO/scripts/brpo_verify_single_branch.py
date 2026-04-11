#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from munch import munchify

from utils.config_utils import load_config
from utils.external_eval_utils import load_gaussians_from_ply
from pseudo_branch.flow_matcher import FlowMatcher
from pseudo_branch.brpo_reprojection_verify import (
    find_neighbor_kfs,
    render_depth_from_state,
    verify_single_branch,
    save_float_map_png,
    save_mask_png,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase B prototype: BRPO single-branch reprojection verification")
    p.add_argument("--internal-cache-root", required=True)
    p.add_argument("--stage-tag", choices=["before_opt", "after_opt"], default="after_opt")
    p.add_argument("--frame-ids", type=int, nargs="+", required=True)
    p.add_argument("--ref-side", choices=["left", "right"], default="left")
    p.add_argument("--output-root", default=None)
    p.add_argument("--tau-reproj-px", type=float, default=4.0)
    p.add_argument("--tau-rel-depth", type=float, default=0.15)
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    cache_root = Path(args.internal_cache_root)
    run_root = cache_root.parent
    config = load_config(str(run_root / "config.yml"))
    pipe = munchify(config["pipeline_params"])

    manifest = load_json(cache_root / "manifest.json")
    camera_states = load_json(cache_root / "camera_states.json")
    states_by_id = {int(s["frame_id"]): s for s in camera_states}
    kf_indices = [int(x) for x in manifest["kf_indices"]]
    background = torch.tensor(manifest["background"], dtype=torch.float32, device="cuda")

    stage_ply = cache_root / args.stage_tag / "point_cloud" / "point_cloud.ply"
    gaussians = load_gaussians_from_ply(config, str(stage_ply))
    matcher = FlowMatcher()

    default_out = cache_root / "brpo_phaseB" / args.stage_tag / f"{args.ref_side}_branch"
    output_root = Path(args.output_root) if args.output_root else default_out
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for frame_id in args.frame_ids:
        pseudo_state = states_by_id[int(frame_id)]
        if pseudo_state["is_keyframe"]:
            raise ValueError(f"frame_id={frame_id} is a keyframe; choose a non-KF pseudo frame")

        left_kf, right_kf = find_neighbor_kfs(int(frame_id), kf_indices)
        ref_id = left_kf if args.ref_side == "left" else right_kf
        if ref_id is None:
            raise ValueError(f"frame_id={frame_id} has no {args.ref_side} ref KF")
        ref_state = states_by_id[int(ref_id)]

        pseudo_rgb_path = cache_root / args.stage_tag / "render_rgb" / f"{int(frame_id)}_pred.png"
        pseudo_depth_path = cache_root / args.stage_tag / "render_depth_npy" / f"{int(frame_id)}_pred.npy"
        ref_rgb_path = Path(ref_state["image_path"])
        pseudo_depth = np.load(pseudo_depth_path).astype(np.float32)
        ref_depth = render_depth_from_state(gaussians, ref_state, pipe, background)

        pts_pseudo, pts_ref, _ = matcher.match_pair(str(pseudo_rgb_path), str(ref_rgb_path), size=int(pseudo_state["image_width"]))
        result = verify_single_branch(
            pseudo_state=pseudo_state,
            ref_state=ref_state,
            pseudo_depth=pseudo_depth,
            ref_depth=ref_depth,
            pts_pseudo=pts_pseudo,
            pts_ref=pts_ref,
            tau_reproj_px=args.tau_reproj_px,
            tau_rel_depth=args.tau_rel_depth,
        )

        frame_out = output_root / f"frame_{int(frame_id):04d}"
        diag_dir = frame_out / "diag"
        diag_dir.mkdir(parents=True, exist_ok=True)

        side = args.ref_side
        np.save(frame_out / f"support_{side}.npy", result["support_mask"])
        np.save(frame_out / f"reproj_error_{side}.npy", result["reproj_error_map"])
        np.save(frame_out / f"rel_depth_error_{side}.npy", result["rel_depth_error_map"])
        np.save(frame_out / "ref_depth_render.npy", ref_depth)

        save_mask_png(result["support_mask"], str(frame_out / f"support_{side}.png"))
        save_float_map_png(result["reproj_error_map"], str(diag_dir / f"reproj_error_{side}.png"), scale=args.tau_reproj_px)
        save_float_map_png(result["rel_depth_error_map"], str(diag_dir / f"rel_depth_error_{side}.png"), scale=args.tau_rel_depth)
        save_float_map_png(result["match_density"], str(diag_dir / "match_density.png"))
        save_float_map_png(ref_depth, str(diag_dir / "ref_depth_render.png"))

        meta = {
            "frame_id": int(frame_id),
            "stage_tag": args.stage_tag,
            "ref_side": args.ref_side,
            "ref_frame_id": int(ref_id),
            "pseudo_rgb_path": str(pseudo_rgb_path),
            "pseudo_depth_path": str(pseudo_depth_path),
            "ref_rgb_path": str(ref_rgb_path),
            "stage_ply": str(stage_ply),
            **result["stats"],
        }
        with open(frame_out / "verification_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        summary.append(meta)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "output_root": str(output_root),
        "num_frames": len(summary),
        "frames": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
