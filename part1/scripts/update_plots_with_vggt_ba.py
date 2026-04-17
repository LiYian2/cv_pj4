#!/usr/bin/env python3
"""
Part1 画图脚本更新版
主图：colmap_full / colmap_108 / vggt_ba_108 比较
额外图：vggt w/o BA vs vggt BA 对比
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CV_ROOT = Path('/home/bzhang512/CV_Project')
RESULTS_ROOT = CV_ROOT / 'results' / 'part1'
PLOTS_ROOT = CV_ROOT / 'plots' / 'part1'
MAIN_DIR = PLOTS_ROOT / 'main'
APP_DIR = PLOTS_ROOT / 'appendix'
TABLE_DIR = PLOTS_ROOT / 'tables'

for d in [MAIN_DIR, APP_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FINAL_CSV = RESULTS_ROOT / 'final' / 'final_metrics_24.csv'
CONV_CSV = RESULTS_ROOT / 'convergence' / 'all_convergence_metrics_24.csv'

final_df = pd.read_csv(FINAL_CSV)
conv_df = pd.read_csv(CONV_CSV)

print(f'final_df shape = {final_df.shape}')
print(f'conv_df shape = {conv_df.shape}')

# Orders and labels
SCENE_ORDER = ['Re10k-1', 'DL3DV-2', '405841']
# 主图只比较这三个
MAIN_SETTING_ORDER = ['colmap_full', 'colmap_108', 'vggt_ba_108']
MODEL_ORDER = ['3dgs', 'scaffold']

METRICS = [
    ('ours_40000.PSNR', 'PSNR (dB)', True),
    ('ours_40000.SSIM', 'SSIM', True),
    ('ours_40000.LPIPS', 'LPIPS', False),
]

MAIN_SETTING_LABELS = {
    'colmap_full': 'COLMAP-full',
    'colmap_108': 'COLMAP-108',
    'vggt_ba_108': 'VGGT+BA-108',
}

VGGT_SETTING_LABELS = {
    'vggt_108': 'VGGT w/o BA',
    'vggt_ba_108': 'VGGT+BA',
}

MODEL_LABELS = {
    '3dgs': '3DGS',
    'scaffold': 'Scaffold-GS',
}

SCENE_LABELS = {
    'Re10k-1': 'Re10k-1',
    'DL3DV-2': 'DL3DV-2',
    '405841': '405841',
}

MAIN_SETTING_COLORS = {
    'colmap_full': '#1f4e79',
    'colmap_108': '#2e8b57',
    'vggt_ba_108': '#c41e3a',
}

VGGT_COLORS = {
    'vggt_108': '#8b8b8b',
    'vggt_ba_108': '#c41e3a',
}

MODEL_MARKERS = {
    '3dgs': 'o',
    'scaffold': 's',
}

MODEL_LINESTYLES = {
    '3dgs': '-',
    'scaffold': '--',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titleweight': 'semibold',
})

def add_setting_key(df):
    df = df.copy()
    df['setting'] = df['variant']
    return df

# =========================================================================
# 1. 主图：Final Performance (colmap_full / colmap_108 / vggt_ba_108)
# =========================================================================

def plot_main_final_performance():
    df = add_setting_key(final_df)
    df = df[df['setting'].isin(MAIN_SETTING_ORDER)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    
    for ax_idx, (metric_col, metric_label, higher_better) in enumerate(METRICS):
        ax = axes[ax_idx]
        
        for scene_idx, scene in enumerate(SCENE_ORDER):
            scene_df = df[df['scene'] == scene]
            
            for model_idx, model in enumerate(MODEL_ORDER):
                subset = scene_df[scene_df['model'] == model]
                
                x_base = scene_idx * 0.5 + model_idx * 0.15
                
                for setting_idx, setting in enumerate(MAIN_SETTING_ORDER):
                    row = subset[subset['setting'] == setting]
                    if row.empty:
                        continue
                    
                    val = row[metric_col].values[0]
                    x = x_base + setting_idx * 0.08
                    
                    color = MAIN_SETTING_COLORS[setting]
                    marker = MODEL_MARKERS[model]
                    
                    ax.scatter(x, val, c=color, marker=marker, s=80, 
                              edgecolors='white', linewidth=0.5, zorder=3)
        
        # X-axis labels
        x_labels = []
        x_positions = []
        for scene_idx, scene in enumerate(SCENE_ORDER):
            for model_idx, model in enumerate(MODEL_ORDER):
                x_pos = scene_idx * 0.5 + model_idx * 0.15 + 0.04
                x_positions.append(x_pos)
                x_labels.append(f'{SCENE_LABELS[scene]}\n{MODEL_LABELS[model]}')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        
        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_handles = []
    for setting in MAIN_SETTING_ORDER:
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=MAIN_SETTING_COLORS[setting],
                                     markersize=10, label=MAIN_SETTING_LABELS[setting]))
    
    axes[0].legend(handles=legend_handles, loc='upper left', frameon=False)
    
    plt.tight_layout()
    out_path = MAIN_DIR / 'part1_final_metrics_main_3settings.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# =========================================================================
# 2. 主图：PSNR Convergence (colmap_full / colmap_108 / vggt_ba_108)
# =========================================================================

def plot_main_psnr_convergence():
    df = add_setting_key(conv_df)
    df = df[df['setting'].isin(MAIN_SETTING_ORDER)].copy()
    df = df[df['split'] == 'test'].copy()
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey='row')
    
    for scene_idx, scene in enumerate(SCENE_ORDER):
        for model_idx, model in enumerate(MODEL_ORDER):
            ax = axes[scene_idx, model_idx]
            
            subset = df[(df['scene'] == scene) & (df['model'] == model)]
            
            for setting in MAIN_SETTING_ORDER:
                setting_df = subset[subset['setting'] == setting]
                if setting_df.empty:
                    continue
                
                setting_df = setting_df.sort_values('iter')
                color = MAIN_SETTING_COLORS[setting]
                linestyle = MODEL_LINESTYLES[model]
                
                ax.plot(setting_df['iter'], setting_df['psnr'],
                       color=color, linestyle=linestyle, linewidth=2,
                       label=MAIN_SETTING_LABELS[setting])
            
            ax.set_ylabel('PSNR (dB)')
            ax.set_title(f'{SCENE_LABELS[scene]} - {MODEL_LABELS[model]}')
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis label only on bottom row
    for model_idx in range(2):
        axes[2, model_idx].set_xlabel('Iteration')
    
    # Legend on first panel
    axes[0, 0].legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    out_path = MAIN_DIR / 'part1_test_psnr_convergence_main_3settings.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# =========================================================================
# 3. 主图：L1 Convergence (colmap_full / colmap_108 / vggt_ba_108)
# =========================================================================

def plot_main_l1_convergence():
    df = add_setting_key(conv_df)
    df = df[df['setting'].isin(MAIN_SETTING_ORDER)].copy()
    df = df[df['split'] == 'test'].copy()
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey='row')
    
    for scene_idx, scene in enumerate(SCENE_ORDER):
        for model_idx, model in enumerate(MODEL_ORDER):
            ax = axes[scene_idx, model_idx]
            
            subset = df[(df['scene'] == scene) & (df['model'] == model)]
            
            for setting in MAIN_SETTING_ORDER:
                setting_df = subset[subset['setting'] == setting]
                if setting_df.empty:
                    continue
                
                setting_df = setting_df.sort_values('iter')
                color = MAIN_SETTING_COLORS[setting]
                linestyle = MODEL_LINESTYLES[model]
                
                ax.plot(setting_df['iter'], setting_df['l1'],
                       color=color, linestyle=linestyle, linewidth=2,
                       label=MAIN_SETTING_LABELS[setting])
            
            ax.set_ylabel('L1')
            ax.set_title(f'{SCENE_LABELS[scene]} - {MODEL_LABELS[model]}')
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    for model_idx in range(2):
        axes[2, model_idx].set_xlabel('Iteration')
    
    axes[0, 0].legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    out_path = MAIN_DIR / 'part1_test_l1_convergence_main_3settings.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# =========================================================================
# 4. 额外图：VGGT w/o BA vs VGGT+BA Final Performance
# =========================================================================

def plot_vggt_ba_comparison():
    df = add_setting_key(final_df)
    vggt_settings = ['vggt_108', 'vggt_ba_108']
    df = df[df['setting'].isin(vggt_settings)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    
    for ax_idx, (metric_col, metric_label, higher_better) in enumerate(METRICS):
        ax = axes[ax_idx]
        
        for scene_idx, scene in enumerate(SCENE_ORDER):
            scene_df = df[df['scene'] == scene]
            
            for model_idx, model in enumerate(MODEL_ORDER):
                subset = scene_df[scene_df['model'] == model]
                
                x_base = scene_idx * 0.5 + model_idx * 0.2
                
                for setting_idx, setting in enumerate(vggt_settings):
                    row = subset[subset['setting'] == setting]
                    if row.empty:
                        continue
                    
                    val = row[metric_col].values[0]
                    x = x_base + setting_idx * 0.1
                    
                    color = VGGT_COLORS[setting]
                    marker = MODEL_MARKERS[model]
                    
                    ax.scatter(x, val, c=color, marker=marker, s=80,
                              edgecolors='white', linewidth=0.5, zorder=3)
        
        # X-axis labels
        x_labels = []
        x_positions = []
        for scene_idx, scene in enumerate(SCENE_ORDER):
            for model_idx, model in enumerate(MODEL_ORDER):
                x_pos = scene_idx * 0.5 + model_idx * 0.2 + 0.05
                x_positions.append(x_pos)
                x_labels.append(f'{SCENE_LABELS[scene]}\n{MODEL_LABELS[model]}')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_handles = []
    for setting in vggt_settings:
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=VGGT_COLORS[setting],
                                     markersize=10, label=VGGT_SETTING_LABELS[setting]))
    
    axes[0].legend(handles=legend_handles, loc='upper left', frameon=False)
    
    plt.tight_layout()
    out_path = APP_DIR / 'part1_vggt_ba_vs_woba_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# =========================================================================
# 5. Summary Tables
# =========================================================================

def export_summary_tables():
    df = add_setting_key(final_df)
    
    # Main summary (only 3 main settings)
    main_df = df[df['setting'].isin(MAIN_SETTING_ORDER)].copy()
    
    # Long format
    long_rows = []
    for _, row in main_df.iterrows():
        for metric_col, metric_label, _ in METRICS:
            long_rows.append({
                'scene': row['scene'],
                'setting': row['setting'],
                'model': row['model'],
                'metric': metric_label,
                'value': row[metric_col],
            })
    long_df = pd.DataFrame(long_rows)
    long_csv = TABLE_DIR / 'part1_main_metrics_long.csv'
    long_df.to_csv(long_csv, index=False)
    print(f'Saved: {long_csv}')
    
    # Wide table
    wide_df = main_df.pivot_table(
        index=['scene', 'model'],
        columns='setting',
        values=['ours_40000.PSNR', 'ours_40000.SSIM', 'ours_40000.LPIPS'],
        aggfunc='first'
    ).round(4)
    wide_csv = TABLE_DIR / 'part1_main_metrics_wide.csv'
    wide_df.to_csv(wide_csv)
    print(f'Saved: {wide_csv}')
    
    # Aggregate mean
    agg_df = main_df.groupby(['setting', 'model']).agg({
        'ours_40000.PSNR': 'mean',
        'ours_40000.SSIM': 'mean',
        'ours_40000.LPIPS': 'mean',
    }).round(4).reset_index()
    agg_csv = TABLE_DIR / 'part1_main_aggregate_mean.csv'
    agg_df.to_csv(agg_csv, index=False)
    print(f'Saved: {agg_csv}')
    
    # VGGT comparison table
    vggt_df = df[df['setting'].isin(['vggt_108', 'vggt_ba_108'])].copy()
    vggt_agg = vggt_df.groupby(['setting', 'model']).agg({
        'ours_40000.PSNR': 'mean',
        'ours_40000.SSIM': 'mean',
        'ours_40000.LPIPS': 'mean',
    }).round(4).reset_index()
    vggt_csv = TABLE_DIR / 'part1_vggt_ba_comparison_mean.csv'
    vggt_agg.to_csv(vggt_csv, index=False)
    print(f'Saved: {vggt_csv}')

# =========================================================================
# Run all
# =========================================================================

if __name__ == '__main__':
    print('\n=== Generating Main Figures ===')
    plot_main_final_performance()
    plot_main_psnr_convergence()
    plot_main_l1_convergence()
    
    print('\n=== Generating Extra Figures ===')
    plot_vggt_ba_comparison()
    
    print('\n=== Exporting Summary Tables ===')
    export_summary_tables()
    
    print('\n=== Done! ===')