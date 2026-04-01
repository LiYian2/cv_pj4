#!/usr/bin/env python3
"""Read part2_s3po run outputs and export CSV summaries.

This script scans all historical timestamp runs under:
  /home/bzhang512/CV_Project/output/part2_s3po

It ignores *_latest symlinks and collects per-run metrics from:
- config.yml
- plot/stats_final.json
- psnr/before_opt/final_result.json
- psnr/after_opt/final_result.json (optional)

Outputs are written to:
- /home/bzhang512/CV_Project/results/part2_s3po/final
- /home/bzhang512/CV_Project/results/part2_s3po/qc
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


DEFAULT_OUTPUT_ROOT = Path("/home/bzhang512/CV_Project/output/part2_s3po")
DEFAULT_RESULTS_ROOT = Path("/home/bzhang512/CV_Project/results/part2_s3po")

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


def safe_read_json(path: Path) -> tuple[Optional[Dict[str, Any]], str]:
    if not path.is_file():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "parse_error"
        return data, "ok"
    except Exception:
        return None, "parse_error"


def safe_read_yaml(path: Path) -> tuple[Optional[Dict[str, Any]], str]:
    if not path.is_file():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None, "parse_error"
        return data, "ok"
    except Exception:
        return None, "parse_error"


def get_nested(data: Optional[Dict[str, Any]], keys: List[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    return None


def parse_timestamp(name: str) -> Optional[str]:
    if not TIMESTAMP_RE.match(name):
        return None
    try:
        return datetime.strptime(name, "%Y-%m-%d-%H-%M-%S").isoformat()
    except Exception:
        return None


def discover_run_dirs(output_root: Path) -> List[Path]:
    run_dirs: List[Path] = []

    for dataset_dir in sorted(output_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.is_symlink():
            continue
        for experiment_dir in sorted(dataset_dir.iterdir()):
            if not experiment_dir.is_dir() or experiment_dir.is_symlink():
                continue
            for group_dir in sorted(experiment_dir.iterdir()):
                if not group_dir.is_dir() or group_dir.is_symlink():
                    continue
                for ts_dir in sorted(group_dir.iterdir()):
                    if not ts_dir.is_dir() or ts_dir.is_symlink():
                        continue
                    run_dirs.append(ts_dir)

    return run_dirs


def build_run_record(run_dir: Path) -> Dict[str, Any]:
    # run_dir layout:
    # <output_root>/<dataset>/<experiment>/<group>/<timestamp>
    dataset = run_dir.parents[2].name
    experiment_name = run_dir.parents[1].name
    group_name = run_dir.parents[0].name
    run_timestamp = run_dir.name
    run_timestamp_iso = parse_timestamp(run_timestamp)

    config_path = run_dir / "config.yml"
    stats_path = run_dir / "plot" / "stats_final.json"
    psnr_before_path = run_dir / "psnr" / "before_opt" / "final_result.json"
    psnr_after_path = run_dir / "psnr" / "after_opt" / "final_result.json"
    viz_dir = run_dir / "viz"

    cfg, config_status = safe_read_yaml(config_path)
    stats, stats_status = safe_read_json(stats_path)
    before, psnr_before_status = safe_read_json(psnr_before_path)
    after, psnr_after_status = safe_read_json(psnr_after_path)
    viz_status = "ok" if viz_dir.is_dir() else "missing"

    required_checks = {
        "config.yml": config_status,
        "plot/stats_final.json": stats_status,
        "psnr/before_opt/final_result.json": psnr_before_status,
    }
    missing_required = [k for k, v in required_checks.items() if v != "ok"]
    is_incomplete = len(missing_required) > 0
    skip_reason = ";".join(missing_required)

    before_psnr = as_float(get_nested(before, ["mean_psnr"]))
    before_ssim = as_float(get_nested(before, ["mean_ssim"]))
    before_lpips = as_float(get_nested(before, ["mean_lpips"]))

    after_psnr = as_float(get_nested(after, ["mean_psnr"]))
    after_ssim = as_float(get_nested(after, ["mean_ssim"]))
    after_lpips = as_float(get_nested(after, ["mean_lpips"]))

    psnr_gain = None
    if before_psnr is not None and after_psnr is not None:
        psnr_gain = after_psnr - before_psnr

    lpips_change = None
    if before_lpips is not None and after_lpips is not None:
        lpips_change = after_lpips - before_lpips

    rec: Dict[str, Any] = {
        "dataset": dataset,
        "experiment_name": experiment_name,
        "group_name": group_name,
        "run_timestamp": run_timestamp,
        "run_timestamp_iso": run_timestamp_iso,
        "run_dir": str(run_dir),
        "run_mtime": run_dir.stat().st_mtime,
        "viz_dir": str(viz_dir),
        "config_path": str(config_path),
        "stats_final_path": str(stats_path),
        "psnr_before_path": str(psnr_before_path),
        "psnr_after_path": str(psnr_after_path),
        "config_status": config_status,
        "stats_status": stats_status,
        "psnr_before_status": psnr_before_status,
        "psnr_after_status": psnr_after_status,
        "viz_status": viz_status,
        "is_incomplete": is_incomplete,
        "skip_reason": skip_reason,
        # requested params from config
        "color_refinement": as_bool(get_nested(cfg, ["Results", "color_refinement"])),
        "alpha": as_float(get_nested(cfg, ["Training", "alpha"])),
        "kf_interval": as_float(get_nested(cfg, ["Training", "kf_interval"])),
        "mapping_itr_num": as_float(get_nested(cfg, ["Training", "mapping_itr_num"])),
        "tracking_itr_num": as_float(get_nested(cfg, ["Training", "tracking_itr_num"])),
        "window_size": as_float(get_nested(cfg, ["Training", "window_size"])),
        "patch_size": as_float(get_nested(cfg, ["depth", "patch_size"])),
        "pose_window": as_float(get_nested(cfg, ["Training", "pose_window"])),
        # stats_final.json
        "stats_rmse": as_float(get_nested(stats, ["rmse"])),
        "stats_mean": as_float(get_nested(stats, ["mean"])),
        "stats_median": as_float(get_nested(stats, ["median"])),
        "stats_std": as_float(get_nested(stats, ["std"])),
        "stats_min": as_float(get_nested(stats, ["min"])),
        "stats_max": as_float(get_nested(stats, ["max"])),
        "stats_sse": as_float(get_nested(stats, ["sse"])),
        # psnr jsons
        "before_mean_psnr": before_psnr,
        "before_mean_ssim": before_ssim,
        "before_mean_lpips": before_lpips,
        "after_mean_psnr": after_psnr,
        "after_mean_ssim": after_ssim,
        "after_mean_lpips": after_lpips,
        # requested extra columns
        "psnr_gain_after_minus_before": psnr_gain,
        "lpips_change_after_minus_before": lpips_change,
    }

    return rec


def build_dataset_summary(main_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "num_runs_total",
                "num_runs_complete",
                "num_runs_incomplete",
                "before_mean_psnr_mean",
                "after_mean_psnr_mean",
                "psnr_gain_after_minus_before_mean",
                "before_mean_lpips_mean",
                "after_mean_lpips_mean",
                "lpips_change_after_minus_before_mean",
                "stats_rmse_mean",
            ]
        )

    grp = main_df.groupby("dataset", dropna=False)
    out = grp.agg(
        num_runs_total=("run_dir", "count"),
        num_runs_complete=("is_incomplete", lambda s: int((~s).sum())),
        num_runs_incomplete=("is_incomplete", lambda s: int(s.sum())),
        before_mean_psnr_mean=("before_mean_psnr", "mean"),
        after_mean_psnr_mean=("after_mean_psnr", "mean"),
        psnr_gain_after_minus_before_mean=("psnr_gain_after_minus_before", "mean"),
        before_mean_lpips_mean=("before_mean_lpips", "mean"),
        after_mean_lpips_mean=("after_mean_lpips", "mean"),
        lpips_change_after_minus_before_mean=("lpips_change_after_minus_before", "mean"),
        stats_rmse_mean=("stats_rmse", "mean"),
    ).reset_index()

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Read part2_s3po run outputs and export CSV summaries.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    output_root: Path = args.output_root
    results_root: Path = args.results_root

    if not output_root.exists():
        raise FileNotFoundError(f"output-root not found: {output_root}")

    final_dir = results_root / "final"
    qc_dir = results_root / "qc"
    final_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(output_root)
    records = [build_run_record(d) for d in run_dirs]

    main_df = pd.DataFrame(records)
    if not main_df.empty:
        sort_cols = [
            "dataset",
            "experiment_name",
            "group_name",
            "run_timestamp",
            "run_mtime",
        ]
        main_df = main_df.sort_values(sort_cols, ascending=[True, True, True, True, True]).reset_index(drop=True)

        # Column layout for all-timestamps main table:
        # 1) key results first (15 cols), with the first 4 explicitly prioritized
        # 2) dataset/experiment + requested config params in the middle
        # 3) path/status/qc fields at the end
        key_result_cols = [
            "stats_rmse",
            "before_mean_psnr",
            "after_mean_psnr",
            "psnr_gain_after_minus_before",
            "stats_mean",
            "stats_median",
            "stats_std",
            "stats_min",
            "stats_max",
            "stats_sse",
            "before_mean_ssim",
            "after_mean_ssim",
            "before_mean_lpips",
            "after_mean_lpips",
            "lpips_change_after_minus_before",
        ]

        front_cols = [
            "dataset",
            "experiment_name",
        ]

        middle_cols = [
            "color_refinement",
            "alpha",
            "kf_interval",
            "mapping_itr_num",
            "tracking_itr_num",
            "window_size",
            "patch_size",
            "pose_window",
        ]

        tail_cols = [
            "group_name",
            "run_timestamp",
            "run_timestamp_iso",
            "run_dir",
            "run_mtime",
            "viz_dir",
            "config_path",
            "stats_final_path",
            "psnr_before_path",
            "psnr_after_path",
            "config_status",
            "stats_status",
            "psnr_before_status",
            "psnr_after_status",
            "viz_status",
            "is_incomplete",
            "skip_reason",
        ]

        preferred_order = front_cols + key_result_cols + middle_cols + tail_cols
        ordered_existing = [c for c in preferred_order if c in main_df.columns]
        remaining_cols = [c for c in main_df.columns if c not in ordered_existing]
        main_df = main_df[ordered_existing + remaining_cols]

    inventory_cols = [
        "dataset",
        "experiment_name",
        "group_name",
        "run_timestamp",
        "run_dir",
        "run_mtime",
        "config_status",
        "stats_status",
        "psnr_before_status",
        "psnr_after_status",
        "viz_status",
        "is_incomplete",
        "skip_reason",
    ]

    inventory_df = main_df[inventory_cols].copy() if not main_df.empty else pd.DataFrame(columns=inventory_cols)
    incomplete_df = inventory_df[inventory_df["is_incomplete"]].copy() if not inventory_df.empty else inventory_df.copy()
    dataset_summary_df = build_dataset_summary(main_df)

    main_csv = final_dir / "part2_s3po_run_summary_all_timestamps.csv"
    dataset_summary_csv = final_dir / "part2_s3po_dataset_summary.csv"
    inventory_csv = qc_dir / "part2_s3po_run_inventory.csv"
    incomplete_csv = qc_dir / "part2_s3po_incomplete_runs.csv"

    main_df.to_csv(main_csv, index=False)
    dataset_summary_df.to_csv(dataset_summary_csv, index=False)
    inventory_df.to_csv(inventory_csv, index=False)
    incomplete_df.to_csv(incomplete_csv, index=False)

    print("output_root =", output_root)
    print("results_root =", results_root)
    print("runs discovered =", len(run_dirs))
    print("runs incomplete =", int(main_df["is_incomplete"].sum()) if not main_df.empty else 0)
    print("main_csv =", main_csv)
    print("dataset_summary_csv =", dataset_summary_csv)
    print("inventory_csv =", inventory_csv)
    print("incomplete_csv =", incomplete_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
