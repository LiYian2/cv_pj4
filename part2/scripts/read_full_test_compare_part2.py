#!/usr/bin/env python3
"""Build cross-model full-test comparison CSV for Part2 (RegGS vs S3PO).

Reads six groups of full-test outputs:
- RegGS (3 datasets): eval_test_all_non_train_subset_v1.json + ate_test_all_non_train_subset_v1.json
- S3PO (3 datasets): external_eval/*/infer_*/eval_external.json + plot/stats_external_infer.json

Writes a compact CSV (no parameter/path columns) to:
  /home/bzhang512/CV_Project/results/part2/part2_full_test_model_compare.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

CV_ROOT = Path("/home/bzhang512/CV_Project")
REGGS_OUTPUT_ROOT = Path("/home/bzhang512/my_storage_500G/CV_Project/output/part2")
S3PO_EXTERNAL_ROOT = Path("/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/external_eval")
RESULTS_ROOT = CV_ROOT / "results" / "part2"


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def canonical_reggs_dataset(name: str) -> str:
    mapping = {
        "re10k_1": "re10k_1",
        "dl3dv_2": "dl3dv_2",
        "405841": "405841",
    }
    return mapping.get(name, name)


def canonical_s3po_dataset(name: str) -> str:
    mapping = {
        "re10k1": "re10k_1",
        "dl3dv2": "dl3dv_2",
        "405841": "405841",
    }
    return mapping.get(name, name)


def collect_reggs_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    eval_files = sorted(REGGS_OUTPUT_ROOT.glob("*/*/eval_test_all_non_train_subset_v1.json"))
    for eval_path in eval_files:
        run_dir = eval_path.parent
        dataset_raw = run_dir.parent.name
        run_name = run_dir.name
        dataset = canonical_reggs_dataset(dataset_raw)

        ate_path = run_dir / "ate_test_all_non_train_subset_v1.json"
        eval_data = safe_load_json(eval_path)
        ate_data = safe_load_json(ate_path)

        if eval_data is None or ate_data is None:
            continue

        aligned = ate_data.get("aligned_test_ate", {}) if isinstance(ate_data.get("aligned_test_ate"), dict) else {}

        rows.append(
            {
                "model": "reggs",
                "dataset": dataset,
                "run_name": run_name,
                "eval_scope": "full_test_all_non_train_subset_v1",
                "num_eval_frames": as_int(eval_data.get("n_test_frames_selected")),
                "num_ate_pairs": as_int(aligned.get("compared_pose_pairs")),
                "avg_psnr": as_float(eval_data.get("avg_psnr")),
                "avg_ssim": as_float(eval_data.get("avg_ssim")),
                "avg_lpips": as_float(eval_data.get("avg_lpips")),
                "ate_rmse": as_float(aligned.get("rmse")),
                "ate_mean": as_float(aligned.get("mean")),
                "ate_median": as_float(aligned.get("median")),
                "ate_std": as_float(aligned.get("std")),
                "ate_min": as_float(aligned.get("min")),
                "ate_max": as_float(aligned.get("max")),
            }
        )

    return rows


def collect_s3po_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    eval_files = sorted(S3PO_EXTERNAL_ROOT.glob("*/infer_*/eval_external.json"))
    for eval_path in eval_files:
        infer_dir = eval_path.parent
        dataset_raw = infer_dir.parent.name
        dataset = canonical_s3po_dataset(dataset_raw)
        run_name = infer_dir.name

        ate_path = infer_dir / "plot" / "stats_external_infer.json"
        eval_data = safe_load_json(eval_path)
        ate_data = safe_load_json(ate_path)

        if eval_data is None or ate_data is None:
            continue

        rows.append(
            {
                "model": "s3po",
                "dataset": dataset,
                "run_name": run_name,
                "eval_scope": "full_test_external_infer",
                "num_eval_frames": as_int(eval_data.get("num_frames")),
                "num_ate_pairs": as_int(eval_data.get("num_frames")),
                "avg_psnr": as_float(eval_data.get("avg_psnr")),
                "avg_ssim": as_float(eval_data.get("avg_ssim")),
                "avg_lpips": as_float(eval_data.get("avg_lpips")),
                "ate_rmse": as_float(ate_data.get("rmse")),
                "ate_mean": as_float(ate_data.get("mean")),
                "ate_median": as_float(ate_data.get("median")),
                "ate_std": as_float(ate_data.get("std")),
                "ate_min": as_float(ate_data.get("min")),
                "ate_max": as_float(ate_data.get("max")),
            }
        )

    return rows


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    rows = collect_reggs_rows() + collect_s3po_rows()
    df = pd.DataFrame(rows)

    col_order = [
        "model",
        "dataset",
        "run_name",
        "eval_scope",
        "num_eval_frames",
        "num_ate_pairs",
        "avg_psnr",
        "avg_ssim",
        "avg_lpips",
        "ate_rmse",
        "ate_mean",
        "ate_median",
        "ate_std",
        "ate_min",
        "ate_max",
    ]

    if df.empty:
        df = pd.DataFrame(columns=col_order)
    else:
        for c in col_order:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[col_order].sort_values(["dataset", "model", "run_name"]).reset_index(drop=True)

    out_csv = RESULTS_ROOT / "part2_full_test_model_compare.csv"
    df.to_csv(out_csv, index=False)

    # Small aggregated table by model + dataset.
    agg = pd.DataFrame(columns=["model", "dataset", "num_rows", "avg_psnr_mean", "avg_ssim_mean", "avg_lpips_mean", "ate_rmse_mean"])
    if not df.empty:
        agg = (
            df.groupby(["model", "dataset"], dropna=False)
            .agg(
                num_rows=("run_name", "count"),
                avg_psnr_mean=("avg_psnr", "mean"),
                avg_ssim_mean=("avg_ssim", "mean"),
                avg_lpips_mean=("avg_lpips", "mean"),
                ate_rmse_mean=("ate_rmse", "mean"),
            )
            .reset_index()
            .sort_values(["dataset", "model"])
            .reset_index(drop=True)
        )

    agg_csv = RESULTS_ROOT / "part2_full_test_model_compare_summary.csv"
    agg.to_csv(agg_csv, index=False)

    print("reggs rows =", len([r for r in rows if r.get("model") == "reggs"]))
    print("s3po rows =", len([r for r in rows if r.get("model") == "s3po"]))
    print("total rows =", len(df))
    print("out_csv =", out_csv)
    print("agg_csv =", agg_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
