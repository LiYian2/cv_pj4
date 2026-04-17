#!/usr/bin/env python3
"""Generate comparison plots for Part2.

Outputs saved to: /home/bzhang512/CV_Project/plots/part2/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_ROOT = Path("/home/bzhang512/CV_Project/results/part2")
PLOTS_ROOT = Path("/home/bzhang512/CV_Project/plots/part2")

def load_data():
    csv_path = RESULTS_ROOT / "part2_all_protocols_compare.csv"
    return pd.read_csv(csv_path)

def plot_protocol_psnr_compare(df):
    """PSNR comparison across protocols by dataset"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = df["dataset"].unique()
    protocols = ["external", "internal", "internal_after"]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = {
        "external": "#FF6B6B",
        "internal": "#4ECDC4",
        "internal_after": "#45B7D1",
    }
    
    for i, protocol in enumerate(protocols):
        subset = df[(df["model"] == "s3po") & (df["protocol"] == protocol)]
        psnr_vals = []
        for ds in datasets:
            ds_data = subset[subset["dataset"] == ds]
            if len(ds_data) > 0:
                psnr_vals.append(ds_data["psnr"].mean())
            else:
                psnr_vals.append(0)
        
        bars = ax.bar(x + i * width, psnr_vals, width, 
                     label=f"S3PO {protocol.replace('_', ' ')}", 
                     color=colors.get(protocol, "gray"))
        
        for bar, val in zip(bars, psnr_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    
    # Add RegGS external
    reggs_ext = df[(df["model"] == "reggs") & (df["protocol"] == "external")]
    for i, ds in enumerate(datasets):
        ds_data = reggs_ext[reggs_ext["dataset"] == ds]
        if len(ds_data) > 0:
            val = ds_data["psnr"].mean()
            ax.scatter(x[i] + 0.375, val, color="#2C3E50", s=100, marker="D", 
                      zorder=5, label="RegGS external" if i == 0 else "")
            ax.text(x[i] + 0.375, val + 0.5, f"{val:.1f}", ha="center", fontsize=8)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Part2: PSNR Comparison by Protocol", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = PLOTS_ROOT / "part2_protocol_psnr_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_protocol_ate_compare(df):
    """ATE comparison across protocols by dataset"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = df["dataset"].unique()
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # S3PO internal (before)
    s3po_int = df[(df["model"] == "s3po") & (df["protocol"] == "internal")]
    ate_vals_int = []
    for ds in datasets:
        ds_data = s3po_int[s3po_int["dataset"] == ds]
        if len(ds_data) > 0:
            ate_vals_int.append(ds_data["ate_rmse"].mean())
        else:
            ate_vals_int.append(0)
    
    bars1 = ax.bar(x - width/2, ate_vals_int, width, label="S3PO internal", color="#4ECDC4")
    for bar, val in zip(bars1, ate_vals_int):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    
    # RegGS external
    reggs_ext = df[(df["model"] == "reggs") & (df["protocol"] == "external")]
    ate_vals_reggs = []
    for ds in datasets:
        ds_data = reggs_ext[reggs_ext["dataset"] == ds]
        if len(ds_data) > 0:
            ate_vals_reggs.append(ds_data["ate_rmse"].mean())
        else:
            ate_vals_reggs.append(0)
    
    bars2 = ax.bar(x + width/2, ate_vals_reggs, width, label="RegGS external", color="#2C3E50")
    for bar, val in zip(bars2, ate_vals_reggs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("ATE RMSE", fontsize=12)
    ax.set_title("Part2: ATE Comparison by Protocol", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = PLOTS_ROOT / "part2_protocol_ate_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_model_external_compare(df):
    """RegGS vs S3PO external protocol comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = df["dataset"].unique()
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # S3PO external
    s3po_ext = df[(df["model"] == "s3po") & (df["protocol"] == "external")]
    psnr_s3po = []
    for ds in datasets:
        ds_data = s3po_ext[s3po_ext["dataset"] == ds]
        if len(ds_data) > 0:
            psnr_s3po.append(ds_data["psnr"].mean())
        else:
            psnr_s3po.append(0)
    
    bars1 = ax.bar(x - width/2, psnr_s3po, width, label="S3PO", color="#FF6B6B")
    
    # RegGS external
    reggs_ext = df[(df["model"] == "reggs") & (df["protocol"] == "external")]
    psnr_reggs = []
    for ds in datasets:
        ds_data = reggs_ext[reggs_ext["dataset"] == ds]
        if len(ds_data) > 0:
            psnr_reggs.append(ds_data["psnr"].mean())
        else:
            psnr_reggs.append(0)
    
    bars2 = ax.bar(x + width/2, psnr_reggs, width, label="RegGS", color="#2C3E50")
    
    # add value labels
    for bars, vals in [(bars1, psnr_s3po), (bars2, psnr_reggs)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Part2: RegGS vs S3PO (External Protocol)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = PLOTS_ROOT / "part2_model_external_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_s3po_internal_improvement(df):
    """S3PO before_opt vs after_opt improvement"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = df["dataset"].unique()
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # before_opt
    s3po_before = df[(df["model"] == "s3po") & (df["protocol"] == "internal")]
    psnr_before = []
    for ds in datasets:
        ds_data = s3po_before[s3po_before["dataset"] == ds]
        if len(ds_data) > 0:
            psnr_before.append(ds_data["psnr"].mean())
        else:
            psnr_before.append(0)
    
    bars1 = ax.bar(x - width/2, psnr_before, width, label="before_opt", color="#4ECDC4")
    
    # after_opt
    s3po_after = df[(df["model"] == "s3po") & (df["protocol"] == "internal_after")]
    psnr_after = []
    for ds in datasets:
        ds_data = s3po_after[s3po_after["dataset"] == ds]
        if len(ds_data) > 0:
            psnr_after.append(ds_data["psnr"].mean())
        else:
            psnr_after.append(0)
    
    bars2 = ax.bar(x + width/2, psnr_after, width, label="after_opt", color="#45B7D1")
    
    # add value labels
    for bars, vals in [(bars1, psnr_before), (bars2, psnr_after)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Part2: S3PO Internal Protocol Improvement", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = PLOTS_ROOT / "part2_s3po_internal_improvement.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    
    plot_protocol_psnr_compare(df)
    plot_protocol_ate_compare(df)
    plot_model_external_compare(df)
    plot_s3po_internal_improvement(df)
    
    print(f"\nAll plots saved to: {PLOTS_ROOT}")

if __name__ == "__main__":
    main()
