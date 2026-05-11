#!/usr/bin/env python3
"""Compute full-frame ATE from S3PO-GS internal_eval_cache/camera_states.json.

This is intended for BRPO/S3PO-GS runs where the online mapper exported
internal_eval_cache/camera_states.json. That file already contains, per frame:
  - predicted pose: pose_c2w (or R/T as w2c fallback)
  - GT pose: R_gt/T_gt (w2c, converted here to c2w)
  - keyframe flag and image name

Example:
  /home/bzhang512/miniconda3/envs/s3po-gs/bin/python \
    /home/bzhang512/CV_Project/part3_BRPO/scripts/compute_full_ate_from_camera_states.py \
    --pred_pose_dir /path/to/experiment/run \
    --out_dir /path/to/experiment/run/plot/full_ate

Use --kf-only to reproduce the old keyframe-only ATE path for sanity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from evo.tools import plot as evo_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute aligned full-frame ATE from S3PO-GS camera_states.json")
    p.add_argument(
        "--pred_pose_dir",
        type=str,
        required=True,
        help="Experiment/run dir, internal_eval_cache dir, or a directory containing camera_states.json.",
    )
    p.add_argument(
        "--camera_states",
        type=str,
        default=None,
        help="Explicit camera_states.json path. Overrides --pred_pose_dir auto-resolution.",
    )
    p.add_argument("--out_dir", type=str, default=None, help="Output directory. Default: <run>/plot/full_ate")
    p.add_argument("--label", type=str, default="full", help="Output label suffix.")
    p.add_argument("--kf-only", action="store_true", help="Evaluate only frames with is_keyframe=true; useful to reproduce old ATE.")
    p.add_argument("--align", choices=["sim3", "se3", "none"], default="sim3", help="Trajectory alignment. sim3 = SE3 + scale.")
    p.add_argument("--plot-plane", choices=["auto", "xy", "xz", "yz"], default="auto")
    p.add_argument("--strict", action="store_true", help="Fail if any selected frame lacks pred or GT pose.")
    return p.parse_args()


def resolve_camera_states(pred_pose_dir: str, camera_states: str | None) -> Path:
    if camera_states:
        path = Path(camera_states).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--camera_states not found: {path}")
        return path

    root = Path(pred_pose_dir).expanduser()
    candidates = []
    if root.is_file():
        candidates.append(root)
    else:
        candidates.extend([
            root / "internal_eval_cache" / "camera_states.json",
            root / "camera_states.json",
        ])
        if root.name == "internal_eval_cache":
            candidates.insert(0, root / "camera_states.json")

    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Could not find camera_states.json. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


def default_out_dir(camera_states_path: Path) -> Path:
    # .../<run>/internal_eval_cache/camera_states.json -> .../<run>/plot/full_ate
    if camera_states_path.parent.name == "internal_eval_cache":
        run_dir = camera_states_path.parent.parent
        return run_dir / "plot" / "full_ate"
    return camera_states_path.parent / "full_ate"


def make_w2c(R, T) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(R, dtype=np.float64)
    M[:3, 3] = np.asarray(T, dtype=np.float64)
    return M


def pose_from_state(state: dict, kind: str) -> np.ndarray | None:
    """Return c2w pose for pred or gt."""
    if kind == "pred":
        if state.get("pose_c2w") is not None:
            M = np.asarray(state["pose_c2w"], dtype=np.float64)
            if M.shape == (4, 4):
                return M
        if state.get("R") is not None and state.get("T") is not None:
            return np.linalg.inv(make_w2c(state["R"], state["T"]))
    elif kind == "gt":
        if state.get("gt_pose_c2w") is not None:
            M = np.asarray(state["gt_pose_c2w"], dtype=np.float64)
            if M.shape == (4, 4):
                return M
        if state.get("R_gt") is not None and state.get("T_gt") is not None:
            return np.linalg.inv(make_w2c(state["R_gt"], state["T_gt"]))
    else:
        raise ValueError(kind)
    return None


def load_poses(camera_states_path: Path, kf_only: bool, strict: bool):
    states = json.load(open(camera_states_path, "r", encoding="utf-8"))
    if not isinstance(states, list):
        raise ValueError(f"Expected a list in {camera_states_path}")

    states = sorted(states, key=lambda s: int(s.get("frame_id", s.get("uid", 0))))
    frame_ids, image_names, gt_poses, pred_poses, skipped = [], [], [], [], []
    for s in states:
        if kf_only and not bool(s.get("is_keyframe", False)):
            continue
        fid = int(s.get("frame_id", s.get("uid", len(frame_ids))))
        gt = pose_from_state(s, "gt")
        pred = pose_from_state(s, "pred")
        if gt is None or pred is None:
            skipped.append(fid)
            if strict:
                raise ValueError(f"Frame {fid} lacks {GT if gt is None else pred} pose")
            continue
        frame_ids.append(fid)
        image_names.append(s.get("image_name") or os.path.basename(str(s.get("image_path", ""))) or str(fid))
        gt_poses.append(gt)
        pred_poses.append(pred)

    if len(gt_poses) < 2:
        raise ValueError(f"Need at least 2 matched poses, got {len(gt_poses)}")
    return frame_ids, image_names, gt_poses, pred_poses, skipped, len(states)


def choose_plot_mode(gt_poses: List[np.ndarray], requested: str):
    if requested != "auto":
        return getattr(evo_plot.PlotMode, requested)
    pts = np.asarray([p[:3, 3] for p in gt_poses], dtype=np.float64)
    variances = np.var(pts, axis=0)
    axes = tuple(sorted(np.argsort(variances)[-2:]))
    if axes == (0, 1):
        return evo_plot.PlotMode.xy
    if axes == (0, 2):
        return evo_plot.PlotMode.xz
    if axes == (1, 2):
        return evo_plot.PlotMode.yz
    return evo_plot.PlotMode.xy


def compute_evo(gt_poses, pred_poses, align: str):
    traj_ref = PosePath3D(poses_se3=gt_poses)
    traj_est = PosePath3D(poses_se3=pred_poses)
    if align == "none":
        traj_aligned = traj_est
    else:
        traj_aligned = trajectory.align_trajectory(
            traj_est,
            traj_ref,
            correct_scale=(align == "sim3"),
            correct_only_scale=False,
        )
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_aligned))
    stats = ape.get_all_statistics()
    errors = np.asarray(ape.error, dtype=np.float64)
    return traj_ref, traj_est, traj_aligned, stats, errors


def estimate_scale(gt_poses, pred_poses, aligned_poses) -> float:
    pred = np.asarray([p[:3, 3] for p in pred_poses], dtype=np.float64)
    ali = np.asarray([p[:3, 3] for p in aligned_poses], dtype=np.float64)
    pred_d = np.linalg.norm(np.diff(pred, axis=0), axis=1)
    ali_d = np.linalg.norm(np.diff(ali, axis=0), axis=1)
    mask = pred_d > 1e-12
    if not np.any(mask):
        return 1.0
    return float(np.median(ali_d[mask] / pred_d[mask]))


def write_outputs(out_dir: Path, label: str, frame_ids, image_names, gt_poses, pred_poses, traj_aligned, stats, errors, meta, plot_mode):
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_poses = traj_aligned.poses_se3
    stats_out = dict(stats)
    stats_out.update(meta)
    stats_path = out_dir / f"stats_{label}.json"
    json.dump(stats_out, open(stats_path, "w", encoding="utf-8"), indent=2)

    trj_path = out_dir / f"trj_{label}.json"
    json.dump(
        {
            "trj_id": frame_ids,
            "image_name": image_names,
            "trj_gt": [p.tolist() for p in gt_poses],
            "trj_est": [p.tolist() for p in pred_poses],
            "trj_est_aligned": [p.tolist() for p in aligned_poses],
        },
        open(trj_path, "w", encoding="utf-8"),
        indent=2,
    )

    csv_path = out_dir / f"errors_{label}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "image_name", "ate_error"])
        for fid, name, err in zip(frame_ids, image_names, errors):
            w.writerow([fid, name, float(err)])

    fig = plt.figure(figsize=(8, 7))
    ax = evo_plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"{label} ATE RMSE: {stats['rmse']:.6f} ({meta['align']}, n={meta['num_matched']})")
    evo_plot.traj(ax, plot_mode, PosePath3D(poses_se3=gt_poses), "--", "gray", "gt")
    evo_plot.traj_colormap(
        ax,
        traj_aligned,
        errors,
        plot_mode,
        min_map=float(np.min(errors)),
        max_map=float(np.max(errors)),
    )
    ax.legend()
    fig.tight_layout()
    traj_png = out_dir / f"evo_2dplot_{label}.png"
    fig.savefig(traj_png, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(frame_ids, errors, linewidth=1.2)
    ax.axhline(stats["mean"], color="orange", linestyle="--", linewidth=1, label=f"mean={stats['mean']:.4f}")
    ax.axhline(stats["median"], color="green", linestyle=":", linewidth=1, label=f"median={stats['median']:.4f}")
    ax.set_xlabel("frame_id")
    ax.set_ylabel("ATE translation error")
    ax.set_title(f"Per-frame ATE error ({label})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    err_png = out_dir / f"ate_error_{label}.png"
    fig.savefig(err_png, dpi=200)
    plt.close(fig)

    return stats_path, trj_path, csv_path, traj_png, err_png


def main() -> None:
    args = parse_args()
    camera_states_path = resolve_camera_states(args.pred_pose_dir, args.camera_states)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out_dir(camera_states_path)

    frame_ids, image_names, gt_poses, pred_poses, skipped, total_states = load_poses(
        camera_states_path, kf_only=args.kf_only, strict=args.strict
    )
    traj_ref, traj_est, traj_aligned, stats, errors = compute_evo(gt_poses, pred_poses, args.align)
    plot_mode = choose_plot_mode(gt_poses, args.plot_plane)
    aligned_poses = traj_aligned.poses_se3

    meta = {
        "camera_states": str(camera_states_path),
        "pred_pose_dir": str(Path(args.pred_pose_dir).expanduser()),
        "mode": "keyframes" if args.kf_only else "full_frames",
        "align": args.align,
        "plot_plane": str(plot_mode).split(".")[-1],
        "num_states_total": int(total_states),
        "num_matched": int(len(frame_ids)),
        "num_skipped_missing_pose": int(len(skipped)),
        "skipped_frame_ids": skipped,
        "first_frame_id": int(frame_ids[0]),
        "last_frame_id": int(frame_ids[-1]),
        "estimated_scale_from_aligned_path": estimate_scale(gt_poses, pred_poses, aligned_poses),
    }
    outputs = write_outputs(out_dir, args.label, frame_ids, image_names, gt_poses, pred_poses, traj_aligned, stats, errors, meta, plot_mode)

    print(json.dumps({"stats": stats, "meta": meta, "outputs": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
