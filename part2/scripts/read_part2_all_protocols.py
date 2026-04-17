#!/usr/bin/env python3
"""Merge RegGS + S3PO external + S3PO internal results into unified comparison table.

Outputs:
- /home/bzhang512/CV_Project/results/part2/part2_all_protocols_compare.csv
"""

import json
from pathlib import Path
import pandas as pd

RESULTS_ROOT = Path("/home/bzhang512/CV_Project/results/part2")

def load_reggs_external():
    """Load RegGS results from existing full_test_model_compare.csv"""
    csv_path = RESULTS_ROOT / "part2_full_test_model_compare.csv"
    if not csv_path.exists():
        print("Warning: RegGS results CSV not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    df_reggs = df[df["model"] == "reggs"].copy()
    df_reggs["protocol"] = "external"
    return df_reggs

def load_s3po_external():
    """Load S3PO external results from existing full_test_model_compare.csv"""
    csv_path = RESULTS_ROOT / "part2_full_test_model_compare.csv"
    if not csv_path.exists():
        print("Warning: S3PO external results CSV not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    df_s3po = df[df["model"] == "s3po"].copy()
    df_s3po["protocol"] = "external"
    return df_s3po

def load_s3po_internal():
    """Load S3PO internal results from newly created CSV"""
    csv_path = RESULTS_ROOT / "part2_s3po_internal.csv"
    if not csv_path.exists():
        print("Warning: S3PO internal results CSV not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    # rename protocol: internal_before -> internal, internal_after -> internal_after
    df["protocol"] = df["protocol"].replace({
        "internal_before": "internal",
        "internal_after": "internal_after",
    })
    return df

def merge_results():
    df_reggs = load_reggs_external()
    df_s3po_ext = load_s3po_external()
    df_s3po_int = load_s3po_internal()
    
    # unify column names
    col_map = {
        "avg_psnr": "psnr",
        "avg_ssim": "ssim",
        "avg_lpips": "lpips",
    }
    
    for df in [df_reggs, df_s3po_ext]:
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
    
    # add mode column for RegGS (all runs are full)
    if "mode" not in df_reggs.columns:
        df_reggs["mode"] = "full"
    if "mode" not in df_s3po_ext.columns:
        df_s3po_ext["mode"] = "unknown"  # external eval doesn't track mode
    
    # add model column for internal
    if "model" not in df_s3po_int.columns:
        df_s3po_int["model"] = "s3po"
    
    # combine all
    df_all = pd.concat([df_reggs, df_s3po_ext, df_s3po_int], ignore_index=True)
    
    # standard columns
    std_cols = [
        "model", "dataset", "protocol", "mode", "run_name", 
        "num_frames", "psnr", "ssim", "lpips",
        "ate_rmse", "ate_mean", "ate_median", "ate_std",
    ]
    
    for c in std_cols:
        if c not in df_all.columns:
            df_all[c] = pd.NA
    
    df_all = df_all[std_cols].sort_values(["dataset", "model", "protocol"]).reset_index(drop=True)
    
    return df_all

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    df_all = merge_results()
    
    out_csv = RESULTS_ROOT / "part2_all_protocols_compare.csv"
    df_all.to_csv(out_csv, index=False)
    
    print(f"\nTotal rows: {len(df_all)}")
    print(f"Output: {out_csv}")
    
    # summary by model + protocol
    summary = df_all.groupby(["model", "protocol"]).agg({
        "dataset": "count",
        "psnr": "mean",
        "ssim": "mean",
        "lpips": "mean",
        "ate_rmse": "mean",
    }).round(2)
    print("\nSummary by model + protocol:")
    print(summary.to_string())

if __name__ == "__main__":
    main()
