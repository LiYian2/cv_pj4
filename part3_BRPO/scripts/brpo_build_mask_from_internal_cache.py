#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from munch import munchify
from PIL import Image

from utils.config_utils import load_config
from utils.external_eval_utils import load_gaussians_from_ply
from pseudo_branch.flow_matcher import FlowMatcher
from pseudo_branch.brpo_reprojection_verify import (
    find_neighbor_kfs,
    render_depth_from_state,
    verify_single_branch,
    save_float_map_png,
)
from pseudo_branch.brpo_confidence_mask import (
    build_brpo_confidence_mask,
    summarize_brpo_mask,
    write_frame_outputs,
)
from pseudo_branch.brpo_train_mask import build_train_confidence_masks


VERIFICATION_VERSION = "brpo-verify-v3-stageM3"


def parse_args():
    p = argparse.ArgumentParser(description="Phase C/M3: build BRPO mask + sparse verified depth from internal cache")
    p.add_argument("--internal-cache-root", required=True)
    p.add_argument("--stage-tag", choices=["before_opt", "after_opt"], default="after_opt")
    p.add_argument("--frame-ids", type=int, nargs="+", required=True)
    p.add_argument("--output-root", default=None)
    p.add_argument("--pseudo-left-root", default=None,
                   help="Optional root containing branch-specific left repaired pseudo RGBs")
    p.add_argument("--pseudo-right-root", default=None,
                   help="Optional root containing branch-specific right repaired pseudo RGBs")
    p.add_argument("--pseudo-fused-root", default=None,
                   help="Optional root containing fused pseudo RGBs (target_rgb_fused.png layout or image_name files)")
    p.add_argument("--verification-mode", choices=["branch_first", "fused_first"], default="branch_first")
    p.add_argument("--tau-reproj-px", type=float, default=4.0)
    p.add_argument("--tau-rel-depth", type=float, default=0.15)
    p.add_argument("--mask-value-both", type=float, default=1.0)
    p.add_argument("--mask-value-single", type=float, default=0.5)
    p.add_argument("--mask-value-none", type=float, default=0.0)
    p.add_argument("--train-mask-mode", choices=["none", "propagate"], default="propagate")
    p.add_argument("--prop-radius-px", type=int, default=2)
    p.add_argument("--prop-tau-rel-depth", type=float, default=0.01)
    p.add_argument("--prop-tau-rgb-l1", type=float, default=0.05)
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_branch_pseudo_rgb(side: str, frame_id: int, image_name: str, default_path: Path, pseudo_left_root: str = None, pseudo_right_root: str = None) -> Path:
    if side == "left" and pseudo_left_root:
        cand = Path(pseudo_left_root) / image_name
        if cand.exists():
            return cand
    if side == "right" and pseudo_right_root:
        cand = Path(pseudo_right_root) / image_name
        if cand.exists():
            return cand
    return default_path


def resolve_fused_pseudo_rgb(frame_id: int, image_name: str, default_path: Path, pseudo_fused_root: str = None) -> Path:
    if not pseudo_fused_root:
        return default_path
    root = Path(pseudo_fused_root)
    candidates = [
        root / image_name,
        root / str(int(frame_id)) / "target_rgb_fused.png",
        root / f"frame_{int(frame_id):04d}" / "target_rgb_fused.png",
        root / f"{int(frame_id):05d}.png",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return default_path


def run_branch(side, frame_id, pseudo_state, ref_state, pseudo_rgb_path, pseudo_depth_path, stage_ply, gaussians, pipe, background, matcher, tau_reproj_px, tau_rel_depth, pseudo_left_root=None, pseudo_right_root=None, pseudo_role="branch"):
    ref_rgb_path = Path(ref_state["image_path"])
    branch_pseudo_rgb_path = resolve_branch_pseudo_rgb(
        side,
        int(frame_id),
        pseudo_state.get("image_name", f"{int(frame_id):05d}.png"),
        Path(pseudo_rgb_path),
        pseudo_left_root,
        pseudo_right_root,
    )
    pseudo_depth = np.load(pseudo_depth_path).astype(np.float32)
    ref_depth = render_depth_from_state(gaussians, ref_state, pipe, background)
    pts_pseudo, pts_ref, _ = matcher.match_pair(str(branch_pseudo_rgb_path), str(ref_rgb_path), size=int(pseudo_state["image_width"]))
    result = verify_single_branch(
        pseudo_state=pseudo_state,
        ref_state=ref_state,
        pseudo_depth=pseudo_depth,
        ref_depth=ref_depth,
        pts_pseudo=pts_pseudo,
        pts_ref=pts_ref,
        tau_reproj_px=tau_reproj_px,
        tau_rel_depth=tau_rel_depth,
    )
    meta = {
        "frame_id": int(frame_id),
        "ref_side": side,
        "ref_frame_id": int(ref_state["frame_id"]),
        "image_name": pseudo_state.get("image_name", f"{int(frame_id):05d}.png"),
        "pseudo_rgb_path": str(branch_pseudo_rgb_path),
        "pseudo_rgb_default_path": str(pseudo_rgb_path),
        "used_branch_override": str(branch_pseudo_rgb_path) != str(pseudo_rgb_path),
        "pseudo_role": pseudo_role,
        "pseudo_depth_path": str(pseudo_depth_path),
        "ref_rgb_path": str(ref_rgb_path),
        "stage_ply": str(stage_ply),
        "verification_version": VERIFICATION_VERSION,
        **result["stats"],
    }
    return result, meta, ref_depth


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

    default_out = cache_root / "brpo_phaseC" / args.stage_tag
    output_root = Path(args.output_root) if args.output_root else default_out
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for frame_id in args.frame_ids:
        pseudo_state = states_by_id[int(frame_id)]
        if pseudo_state["is_keyframe"]:
            raise ValueError(f"frame_id={frame_id} is a keyframe; choose a non-KF pseudo frame")

        left_kf, right_kf = find_neighbor_kfs(int(frame_id), kf_indices)
        if left_kf is None or right_kf is None:
            raise ValueError(f"frame_id={frame_id} missing left/right KF neighbor")
        left_state = states_by_id[int(left_kf)]
        right_state = states_by_id[int(right_kf)]

        pseudo_rgb_path = cache_root / args.stage_tag / "render_rgb" / f"{int(frame_id)}_pred.png"
        pseudo_depth_path = cache_root / args.stage_tag / "render_depth_npy" / f"{int(frame_id)}_pred.npy"
        image_name = pseudo_state.get("image_name", f"{int(frame_id):05d}.png")
        fused_pseudo_rgb_path = resolve_fused_pseudo_rgb(int(frame_id), image_name, Path(pseudo_rgb_path), args.pseudo_fused_root)

        if args.verification_mode == "fused_first":
            left_result, left_meta, left_ref_depth = run_branch(
                "left", frame_id, pseudo_state, left_state, fused_pseudo_rgb_path, pseudo_depth_path,
                stage_ply, gaussians, pipe, background, matcher, args.tau_reproj_px, args.tau_rel_depth,
                None, None, pseudo_role="fused"
            )
            right_result, right_meta, right_ref_depth = run_branch(
                "right", frame_id, pseudo_state, right_state, fused_pseudo_rgb_path, pseudo_depth_path,
                stage_ply, gaussians, pipe, background, matcher, args.tau_reproj_px, args.tau_rel_depth,
                None, None, pseudo_role="fused"
            )
        else:
            left_result, left_meta, left_ref_depth = run_branch(
                "left", frame_id, pseudo_state, left_state, pseudo_rgb_path, pseudo_depth_path,
                stage_ply, gaussians, pipe, background, matcher, args.tau_reproj_px, args.tau_rel_depth,
                args.pseudo_left_root, args.pseudo_right_root, pseudo_role="branch"
            )
            right_result, right_meta, right_ref_depth = run_branch(
                "right", frame_id, pseudo_state, right_state, pseudo_rgb_path, pseudo_depth_path,
                stage_ply, gaussians, pipe, background, matcher, args.tau_reproj_px, args.tau_rel_depth,
                args.pseudo_left_root, args.pseudo_right_root, pseudo_role="branch"
            )

        fused = build_brpo_confidence_mask(
            left_result["support_mask"],
            right_result["support_mask"],
            value_both=args.mask_value_both,
            value_single=args.mask_value_single,
            value_none=args.mask_value_none,
        )

        frame_meta = summarize_brpo_mask(
            frame_id=int(frame_id),
            left_stats=left_result["stats"],
            right_stats=right_result["stats"],
            fused=fused,
        )
        frame_meta.update({
            "verification_version": VERIFICATION_VERSION,
            "verification_mode": args.verification_mode,
            "stage_tag": args.stage_tag,
            "pseudo_rgb_path": str(pseudo_rgb_path),
            "fused_pseudo_rgb_path": str(fused_pseudo_rgb_path),
            "pseudo_depth_path": str(pseudo_depth_path),
            "left_branch_pseudo_rgb_path": left_meta.get("pseudo_rgb_path"),
            "right_branch_pseudo_rgb_path": right_meta.get("pseudo_rgb_path"),
            "left_branch_pseudo_rgb_default_path": left_meta.get("pseudo_rgb_default_path"),
            "right_branch_pseudo_rgb_default_path": right_meta.get("pseudo_rgb_default_path"),
            "left_branch_used_override": left_meta.get("used_branch_override"),
            "right_branch_used_override": right_meta.get("used_branch_override"),
            "left_ref_frame_id": int(left_kf),
            "right_ref_frame_id": int(right_kf),
            "left_ref_rgb_path": str(left_state["image_path"]),
            "right_ref_rgb_path": str(right_state["image_path"]),
            "stage_ply": str(stage_ply),
            "verification_policy": {
                "tau_reproj_px": float(args.tau_reproj_px),
                "tau_rel_depth": float(args.tau_rel_depth),
                "mask_value_both": float(args.mask_value_both),
                "mask_value_single": float(args.mask_value_single),
                "mask_value_none": float(args.mask_value_none),
                "train_mask_mode": args.train_mask_mode,
                "prop_radius_px": int(args.prop_radius_px),
                "prop_tau_rel_depth": float(args.prop_tau_rel_depth),
                "prop_tau_rgb_l1": float(args.prop_tau_rgb_l1),
            },
            "matcher": {
                "type": "FlowMatcher",
                "branch_pseudo_rgb_inputs": {
                    "left_root": args.pseudo_left_root,
                    "right_root": args.pseudo_right_root,
                    "fused_root": args.pseudo_fused_root,
                },
                "train_mask_mode": args.train_mask_mode,
            },
            "ref_depth_source": {
                "type": "render_from_stage_ply",
                "stage_ply": str(stage_ply),
            },
            "left_branch_meta": left_meta,
            "right_branch_meta": right_meta,
        })

        frame_out = output_root / f"frame_{int(frame_id):04d}"
        pseudo_rgb_for_train = fused_pseudo_rgb_path if args.verification_mode == "fused_first" else pseudo_rgb_path
        pseudo_rgb_img = np.asarray(Image.open(pseudo_rgb_for_train).convert("RGB"), dtype=np.float32) / 255.0
        pseudo_depth_arr = np.load(pseudo_depth_path).astype(np.float32)
        train_masks = None
        if args.train_mask_mode == "propagate":
            train_masks = build_train_confidence_masks(
                seed_left=fused["support_left"],
                seed_right=fused["support_right"],
                pseudo_rgb=pseudo_rgb_img,
                render_depth=pseudo_depth_arr,
                value_both=args.mask_value_both,
                value_single=args.mask_value_single,
                value_none=args.mask_value_none,
                max_radius_px=args.prop_radius_px,
                tau_rel_depth=args.prop_tau_rel_depth,
                tau_rgb_l1=args.prop_tau_rgb_l1,
            )
            frame_meta.update(train_masks["summary"])
            frame_meta["train_mask_mode"] = args.train_mask_mode
        else:
            frame_meta["train_mask_mode"] = args.train_mask_mode

        frame_meta.update({
            "projected_depth_left_path": f"frame_{int(frame_id):04d}/projected_depth_left.npy",
            "projected_depth_right_path": f"frame_{int(frame_id):04d}/projected_depth_right.npy",
            "projected_depth_valid_left_path": f"frame_{int(frame_id):04d}/projected_depth_valid_left.npy",
            "projected_depth_valid_right_path": f"frame_{int(frame_id):04d}/projected_depth_valid_right.npy",
        })

        write_frame_outputs(frame_out, fused, frame_meta, train_masks=train_masks)

        diag_dir = frame_out / "diag"
        np.save(frame_out / "ref_depth_left_render.npy", left_ref_depth)
        np.save(frame_out / "ref_depth_right_render.npy", right_ref_depth)
        np.save(frame_out / "reproj_error_left.npy", left_result["reproj_error_map"])
        np.save(frame_out / "reproj_error_right.npy", right_result["reproj_error_map"])
        np.save(frame_out / "rel_depth_error_left.npy", left_result["rel_depth_error_map"])
        np.save(frame_out / "rel_depth_error_right.npy", right_result["rel_depth_error_map"])
        np.save(frame_out / "projected_depth_left.npy", left_result["projected_depth_map"])
        np.save(frame_out / "projected_depth_right.npy", right_result["projected_depth_map"])
        np.save(frame_out / "projected_depth_valid_left.npy", left_result["projected_depth_valid_mask"])
        np.save(frame_out / "projected_depth_valid_right.npy", right_result["projected_depth_valid_mask"])

        save_float_map_png(left_ref_depth, str(diag_dir / "ref_depth_left_render.png"))
        save_float_map_png(right_ref_depth, str(diag_dir / "ref_depth_right_render.png"))
        save_float_map_png(left_result["reproj_error_map"], str(diag_dir / "reproj_error_left.png"), scale=args.tau_reproj_px)
        save_float_map_png(right_result["reproj_error_map"], str(diag_dir / "reproj_error_right.png"), scale=args.tau_reproj_px)
        save_float_map_png(left_result["rel_depth_error_map"], str(diag_dir / "rel_depth_error_left.png"), scale=args.tau_rel_depth)
        save_float_map_png(right_result["rel_depth_error_map"], str(diag_dir / "rel_depth_error_right.png"), scale=args.tau_rel_depth)
        save_float_map_png(left_result["match_density"], str(diag_dir / "match_density_left.png"))
        save_float_map_png(right_result["match_density"], str(diag_dir / "match_density_right.png"))
        save_float_map_png(left_result["projected_depth_map"], str(diag_dir / "projected_depth_left.png"))
        save_float_map_png(right_result["projected_depth_map"], str(diag_dir / "projected_depth_right.png"))
        save_float_map_png(left_result["projected_depth_valid_mask"], str(diag_dir / "projected_depth_valid_left.png"), scale=1.0)
        save_float_map_png(right_result["projected_depth_valid_mask"], str(diag_dir / "projected_depth_valid_right.png"), scale=1.0)

        summary.append(frame_meta)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_root / "summary_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "verification_version": VERIFICATION_VERSION,
            "internal_cache_root": str(cache_root),
            "stage_tag": args.stage_tag,
            "stage_ply": str(stage_ply),
            "frame_ids": [int(x) for x in args.frame_ids],
            "verification_mode": args.verification_mode,
            "pseudo_left_root": args.pseudo_left_root,
            "pseudo_right_root": args.pseudo_right_root,
            "pseudo_fused_root": args.pseudo_fused_root,
            "tau_reproj_px": float(args.tau_reproj_px),
            "tau_rel_depth": float(args.tau_rel_depth),
            "mask_value_both": float(args.mask_value_both),
            "mask_value_single": float(args.mask_value_single),
            "mask_value_none": float(args.mask_value_none),
            "train_mask_mode": args.train_mask_mode,
            "prop_radius_px": int(args.prop_radius_px),
            "prop_tau_rel_depth": float(args.prop_tau_rel_depth),
            "prop_tau_rgb_l1": float(args.prop_tau_rgb_l1),
            "exports": {
                "projected_depth": True,
                "projected_depth_valid_mask": True,
            },
        }, f, indent=2)

    print(json.dumps({
        "output_root": str(output_root),
        "num_frames": len(summary),
        "frames": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
