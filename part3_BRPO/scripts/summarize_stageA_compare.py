#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args():
    p = argparse.ArgumentParser(description="Summarize Part3 StageA compare runs into JSON/CSV/Markdown tables")
    p.add_argument("--run-dirs", nargs="+", required=True, help="One or more StageA run directories")
    p.add_argument("--output-root", required=True)
    p.add_argument("--reference-replay-json", default=None, help="Optional replay_eval.json used as before-refine baseline")
    p.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --run-dirs order")
    p.add_argument("--strict-replay", action="store_true", help="Fail if replay_eval.json cannot be found for a run")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_series(series: list[Any], prefix: str) -> dict[str, float | None]:
    vals = [safe_float(x) for x in series]
    vals = [x for x in vals if x is not None]
    if not vals:
        return {
            f"{prefix}_last": None,
            f"{prefix}_mean": None,
            f"{prefix}_first10_mean": None,
            f"{prefix}_first20_mean": None,
        }
    return {
        f"{prefix}_last": float(vals[-1]),
        f"{prefix}_mean": float(sum(vals) / len(vals)),
        f"{prefix}_first10_mean": float(sum(vals[:10]) / len(vals[:10])),
        f"{prefix}_first20_mean": float(sum(vals[:20]) / len(vals[:20])),
    }


def load_stageA_blob(run_dir: Path) -> dict[str, Any]:
    stageA_path = run_dir / "stageA_history.json"
    if stageA_path.exists():
        return load_json(stageA_path)
    refinement_path = run_dir / "refinement_history.json"
    if refinement_path.exists():
        blob = load_json(refinement_path)
        if isinstance(blob, dict) and "stageA" in blob:
            return blob["stageA"]
        return blob
    raise FileNotFoundError(f"No stageA_history.json or refinement_history.json under {run_dir}")


def discover_replay_json(run_dir: Path) -> Path | None:
    direct = run_dir / "replay_eval" / "replay_eval.json"
    if direct.exists():
        return direct
    direct2 = run_dir / "replay_eval.json"
    if direct2.exists():
        return direct2
    matches = sorted(run_dir.rglob("replay_eval.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred = [m for m in matches if m.parent.name == "replay_eval"]
        if len(preferred) == 1:
            return preferred[0]
        return matches[0]
    return None


def load_replay_pair(run_dir: Path, strict: bool) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
    replay_json = discover_replay_json(run_dir)
    if replay_json is None:
        if strict:
            raise FileNotFoundError(f"No replay_eval.json found under {run_dir}")
        return None, None, None
    meta_path = replay_json.parent / "replay_eval_meta.json"
    replay = load_json(replay_json)
    meta = load_json(meta_path) if meta_path.exists() else None
    return replay, meta, replay_json


def build_row(run_dir: Path, label: str, reference_replay: dict[str, Any] | None, strict_replay: bool) -> dict[str, Any]:
    blob = load_stageA_blob(run_dir)
    eff = blob.get("effective_source_summary", {}) or {}
    hist = blob.get("history", {}) or {}
    args = blob.get("args", {}) or {}
    final_pose = blob.get("final_true_pose_delta_aggregate") or blob.get("true_pose_delta_aggregate") or {}
    replay, replay_meta, replay_json = load_replay_pair(run_dir, strict=strict_replay)

    row: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "stage_mode": blob.get("stage_mode"),
        "num_pseudo_viewpoints_loaded": blob.get("num_pseudo_viewpoints_loaded"),
        "signal_pipeline": eff.get("signal_pipeline", args.get("signal_pipeline")),
        "signal_v2_root": eff.get("signal_v2_root", args.get("signal_v2_root")),
        "stageA_rgb_mask_mode_requested": eff.get("stageA_rgb_mask_mode_requested", args.get("stageA_rgb_mask_mode")),
        "stageA_depth_mask_mode_requested": eff.get("stageA_depth_mask_mode_requested", args.get("stageA_depth_mask_mode")),
        "stageA_target_depth_mode_requested": eff.get("stageA_target_depth_mode_requested", args.get("stageA_target_depth_mode")),
        "stageA_depth_loss_mode": eff.get("stageA_depth_loss_mode", args.get("stageA_depth_loss_mode")),
        "stageA_iters": args.get("stageA_iters"),
        "num_pseudo_views": args.get("num_pseudo_views"),
        "stageA_lambda_abs_pose": eff.get("stageA_lambda_abs_pose", args.get("stageA_lambda_abs_pose")),
        "stageA_lambda_abs_t": eff.get("stageA_lambda_abs_t", args.get("stageA_lambda_abs_t")),
        "stageA_lambda_abs_r": eff.get("stageA_lambda_abs_r", args.get("stageA_lambda_abs_r")),
        "stageA_abs_pose_robust": eff.get("stageA_abs_pose_robust", args.get("stageA_abs_pose_robust")),
        "stageA_abs_pose_scale_source": eff.get("stageA_abs_pose_scale_source", args.get("stageA_abs_pose_scale_source")),
        "mean_stageA_scene_scale": eff.get("mean_stageA_scene_scale"),
        "mean_confidence_nonzero_ratio": eff.get("mean_confidence_nonzero_ratio"),
        "mean_target_depth_verified_ratio": eff.get("mean_target_depth_verified_ratio"),
        "mean_target_depth_render_fallback_ratio": eff.get("mean_target_depth_render_fallback_ratio"),
        "mean_target_depth_seed_ratio": eff.get("mean_target_depth_seed_ratio"),
        "mean_target_depth_dense_ratio": eff.get("mean_target_depth_dense_ratio"),
        "final_mean_trans_norm": final_pose.get("mean_trans_norm"),
        "final_max_trans_norm": final_pose.get("max_trans_norm"),
        "final_mean_rot_fro_norm": final_pose.get("mean_rot_fro_norm"),
        "final_max_rot_fro_norm": final_pose.get("max_rot_fro_norm"),
        "replay_json_path": str(replay_json) if replay_json else None,
    }

    for key in [
        "loss_total",
        "loss_rgb",
        "loss_depth",
        "loss_depth_seed",
        "loss_depth_dense",
        "loss_depth_fallback",
        "loss_pose_reg",
        "loss_abs_pose_reg",
        "loss_abs_pose_trans",
        "loss_abs_pose_rot",
        "abs_pose_rho_norm",
        "abs_pose_theta_norm",
        "scene_scale_used",
        "grad_norm_xyz",
    ]:
        row.update(summarize_series(hist.get(key, []), key))

    if replay:
        row.update({
            "replay_avg_psnr": replay.get("avg_psnr"),
            "replay_avg_ssim": replay.get("avg_ssim"),
            "replay_avg_lpips": replay.get("avg_lpips"),
            "replay_num_frames": replay.get("num_frames"),
        })
    else:
        row.update({
            "replay_avg_psnr": None,
            "replay_avg_ssim": None,
            "replay_avg_lpips": None,
            "replay_num_frames": None,
        })

    compare_to_internal = (replay_meta or {}).get("compare_to_internal") if replay_meta else None
    if compare_to_internal:
        delta = compare_to_internal.get("delta", {}) or {}
        row.update({
            "delta_vs_internal_psnr": delta.get("psnr"),
            "delta_vs_internal_ssim": delta.get("ssim"),
            "delta_vs_internal_lpips": delta.get("lpips"),
        })
    else:
        row.update({
            "delta_vs_internal_psnr": None,
            "delta_vs_internal_ssim": None,
            "delta_vs_internal_lpips": None,
        })

    if reference_replay and replay:
        row.update({
            "delta_vs_reference_psnr": safe_float(replay.get("avg_psnr")) - safe_float(reference_replay.get("avg_psnr")),
            "delta_vs_reference_ssim": safe_float(replay.get("avg_ssim")) - safe_float(reference_replay.get("avg_ssim")),
            "delta_vs_reference_lpips": safe_float(replay.get("avg_lpips")) - safe_float(reference_replay.get("avg_lpips")),
        })
    else:
        row.update({
            "delta_vs_reference_psnr": None,
            "delta_vs_reference_ssim": None,
            "delta_vs_reference_lpips": None,
        })

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]):
    columns = [
        "label",
        "signal_pipeline",
        "stageA_rgb_mask_mode_requested",
        "stageA_depth_mask_mode_requested",
        "stageA_target_depth_mode_requested",
        "stageA_depth_loss_mode",
        "stageA_iters",
        "num_pseudo_views",
        "stageA_lambda_abs_t",
        "stageA_lambda_abs_r",
        "mean_stageA_scene_scale",
        "mean_confidence_nonzero_ratio",
        "mean_target_depth_verified_ratio",
        "mean_target_depth_render_fallback_ratio",
        "loss_total_last",
        "loss_rgb_last",
        "loss_depth_last",
        "loss_depth_seed_last",
        "loss_depth_dense_last",
        "loss_abs_pose_trans_last",
        "loss_abs_pose_rot_last",
        "abs_pose_rho_norm_last",
        "abs_pose_theta_norm_last",
        "final_mean_trans_norm",
        "final_mean_rot_fro_norm",
        "replay_avg_psnr",
        "replay_avg_ssim",
        "replay_avg_lpips",
        "delta_vs_reference_psnr",
        "delta_vs_reference_ssim",
        "delta_vs_reference_lpips",
        "delta_vs_internal_psnr",
        "delta_vs_internal_ssim",
        "delta_vs_internal_lpips",
        "run_dir",
        "replay_json_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def fmt(value: Any, digits: int = 6) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], reference_path: str | None):
    lines = ["# StageA Compare Summary", ""]
    if reference_path:
        lines += [f"reference_replay_json: `{reference_path}`", ""]
    headers = [
        "label", "pipeline", "rgb_mask", "depth_mask", "depth_target", "abs_t", "abs_r",
        "mask_cov", "verified", "depth_seed_last", "depth_dense_last", "abs_t_last", "abs_r_last",
        "rho_last", "theta_last", "replay_psnr", "dPSNR(ref)", "dPSNR(internal)"
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append(
            "| " + " | ".join([
                str(row.get("label", "-")),
                str(row.get("signal_pipeline", "-")),
                str(row.get("stageA_rgb_mask_mode_requested", "-")),
                str(row.get("stageA_depth_mask_mode_requested", "-")),
                str(row.get("stageA_target_depth_mode_requested", "-")),
                fmt(row.get("stageA_lambda_abs_t")),
                fmt(row.get("stageA_lambda_abs_r")),
                fmt(row.get("mean_confidence_nonzero_ratio")),
                fmt(row.get("mean_target_depth_verified_ratio")),
                fmt(row.get("loss_depth_seed_last")),
                fmt(row.get("loss_depth_dense_last")),
                fmt(row.get("loss_abs_pose_trans_last")),
                fmt(row.get("loss_abs_pose_rot_last")),
                fmt(row.get("abs_pose_rho_norm_last")),
                fmt(row.get("abs_pose_theta_norm_last")),
                fmt(row.get("replay_avg_psnr"), digits=4),
                fmt(row.get("delta_vs_reference_psnr"), digits=4),
                fmt(row.get("delta_vs_internal_psnr"), digits=4),
            ]) + " |"
        )
    lines += ["", "## Run paths", ""]
    for row in rows:
        lines.append(f"- {row['label']}: `{row['run_dir']}`")
        if row.get("replay_json_path"):
            lines.append(f"  - replay: `{row['replay_json_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_dirs = [Path(x).resolve() for x in args.run_dirs]
    if args.labels and len(args.labels) != len(run_dirs):
        raise ValueError("--labels count must match --run-dirs count")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reference_replay = load_json(Path(args.reference_replay_json)) if args.reference_replay_json else None

    rows = []
    for idx, run_dir in enumerate(run_dirs):
        label = args.labels[idx] if args.labels else run_dir.name
        rows.append(build_row(run_dir, label, reference_replay=reference_replay, strict_replay=args.strict_replay))

    summary = {
        "num_runs": len(rows),
        "reference_replay_json": str(Path(args.reference_replay_json).resolve()) if args.reference_replay_json else None,
        "rows": rows,
    }
    summary_path = output_root / "summary.json"
    csv_path = output_root / "compare_table.csv"
    md_path = output_root / "compare_table.md"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, summary.get("reference_replay_json"))

    print(json.dumps({
        "output_root": str(output_root),
        "summary_json": str(summary_path),
        "compare_csv": str(csv_path),
        "compare_md": str(md_path),
        "num_runs": len(rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
