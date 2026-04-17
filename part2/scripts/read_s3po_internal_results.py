#!/usr/bin/env python3
"""Read S3PO internal eval results from full runs.

Internal eval = before_opt/after_opt PSNR + ATE from SLAM internal tracked cameras.
External eval (00_external_eval) is handled separately.

Outputs:
- /home/bzhang512/CV_Project/results/part2/part2_s3po_internal.csv
"""

import json
from pathlib import Path
import pandas as pd

S3PO_ROOT = Path("/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po")
RESULTS_ROOT = Path("/home/bzhang512/CV_Project/results/part2")

# Dataset paths: mapping from standard name to actual directory names
DATASET_PATHS = {
    "re10k_1": {
        "full": "re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po",
        "sparse": "re10k-1/s3po_re10k-1_sparse/Re10k-1_part2_s3po",
    },
    "dl3dv_2": {
        "full": "dl3dv-2/s3po_dl3dv-2_full/DL3DV-2_part2_s3po",
        "sparse": "dl3dv-2/s3po_dl3dv-2_sparse/DL3DV-2_part2_s3po",
    },
    "405841": {
        "full": "405841/s3po_405841_full/405841_part2_s3po",
        "sparse": "405841/s3po_405841_sparse/405841_part2_s3po",
    },
}

def safe_load_json(path):
    if not path.is_file():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except:
        return None

def collect_internal_results():
    rows = []
    
    for dataset_std, paths in DATASET_PATHS.items():
        for mode in ["full", "sparse"]:
            subpath = paths[mode]
            internal_dir = S3PO_ROOT / subpath
            
            if not internal_dir.exists():
                print(f"Warning: {internal_dir} not found, skipping")
                continue
            
            for run_dir in sorted(internal_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                
                run_name = run_dir.name
                
                # before_opt
                before_json = run_dir / "psnr" / "before_opt" / "final_result.json"
                before_data = safe_load_json(before_json)
                
                # after_opt
                after_json = run_dir / "psnr" / "after_opt" / "final_result.json"
                after_data = safe_load_json(after_json)
                
                # ATE stats
                stats_json = run_dir / "plot" / "stats_final.json"
                stats_data = safe_load_json(stats_json)
                
                if before_data:
                    row = {
                        "model": "s3po",
                        "dataset": dataset_std,
                        "protocol": "internal_before",
                        "mode": mode,
                        "run_name": run_name,
                        "num_frames": before_data.get("num_frames", before_data.get("n_test_frames", "NA")),
                        "psnr": before_data.get("mean_psnr", before_data.get("avg_psnr")),
                        "ssim": before_data.get("mean_ssim", before_data.get("avg_ssim")),
                        "lpips": before_data.get("mean_lpips", before_data.get("avg_lpips")),
                        "ate_rmse": stats_data.get("rmse") if stats_data else None,
                        "ate_mean": stats_data.get("mean") if stats_data else None,
                        "ate_median": stats_data.get("median") if stats_data else None,
                        "ate_std": stats_data.get("std") if stats_data else None,
                    }
                    rows.append(row)
                
                if after_data:
                    row = {
                        "model": "s3po",
                        "dataset": dataset_std,
                        "protocol": "internal_after",
                        "mode": mode,
                        "run_name": run_name,
                        "num_frames": after_data.get("num_frames", after_data.get("n_test_frames", "NA")),
                        "psnr": after_data.get("mean_psnr", after_data.get("avg_psnr")),
                        "ssim": after_data.get("mean_ssim", after_data.get("avg_ssim")),
                        "lpips": after_data.get("mean_lpips", after_data.get("avg_lpips")),
                        "ate_rmse": stats_data.get("rmse") if stats_data else None,
                        "ate_mean": stats_data.get("mean") if stats_data else None,
                        "ate_median": stats_data.get("median") if stats_data else None,
                        "ate_std": stats_data.get("std") if stats_data else None,
                    }
                    rows.append(row)
    
    return rows

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    rows = collect_internal_results()
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("Warning: No data collected")
    else:
        df = df.sort_values(["dataset", "mode", "protocol"]).reset_index(drop=True)
    
    out_csv = RESULTS_ROOT / "part2_s3po_internal.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\nCollected {len(df)} rows")
    print(f"Output: {out_csv}")
    if not df.empty:
        print("\nPreview:")
        print(df[["dataset", "mode", "protocol", "psnr", "ssim", "lpips", "ate_rmse"]].to_string())

if __name__ == "__main__":
    main()
