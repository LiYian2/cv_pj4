#!/usr/bin/env python3
"""
执行 part1 绘图（使用原有风格，数据改为 vggt_ba）
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

# Orders and labels (用 vggt_ba_108 替换 vggt_108)
SCENE_ORDER = ['Re10k-1', 'DL3DV-2', '405841']
SETTING_ORDER = ['planA_colmap_full', 'planA_colmap_108', 'planB_vggt_ba_108']
MODEL_ORDER = ['3dgs', 'scaffold']

METRICS = [
    ('ours_40000.PSNR', 'PSNR ↑', True),
    ('ours_40000.SSIM', 'SSIM ↑', True),
    ('ours_40000.LPIPS', 'LPIPS ↓', False),
]

SETTING_LABELS = {
    'planA_colmap_full': 'COLMAP full',
    'planA_colmap_108': 'COLMAP-108',
    'planB_vggt_ba_108': 'VGGT+BA-108',
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

SETTING_COLORS = {
    'planA_colmap_full': '#1f4e79',
    'planA_colmap_108': '#2e8b57',
    'planB_vggt_ba_108': '#8b1e3f',
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
    df['setting'] = df['plan'] + '_' + df['variant']
    return df

def apply_orders(df):
    df = df.copy()
    df['scene'] = pd.Categorical(df['scene'], categories=SCENE_ORDER, ordered=True)
    df['setting'] = pd.Categorical(df['setting'], categories=SETTING_ORDER, ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    return df

def metric_best_mask(df, metric, higher_is_better=True):
    df = df.copy()
    best_vals = (
        df.groupby('scene')[metric].max()
        if higher_is_better
        else df.groupby('scene')[metric].min()
    )
    df['_is_best'] = df.apply(lambda r: np.isclose(r[metric], best_vals[r['scene']]), axis=1)
    return df['_is_best']

def flatten_columns(df):
    df = df.copy()
    df.columns = ['__'.join([str(x) for x in col if str(x) != '']) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df

final_df = apply_orders(add_setting_key(final_df))
conv_df = apply_orders(add_setting_key(conv_df))

# =========================================================
# 主图1: paired compact (原有风格)
# =========================================================

print('\n=== 生成主图: paired compact ===')

plot_df = final_df[
    ['scene', 'setting', 'model', 'ours_40000.PSNR', 'ours_40000.SSIM', 'ours_40000.LPIPS']
].copy()

plot_df['scene'] = plot_df['scene'].astype(str).str.strip()
plot_df['setting'] = plot_df['setting'].astype(str).str.strip().str.lower()
plot_df['model'] = plot_df['model'].astype(str).str.strip().str.lower()

# alias mapping
setting_alias = {
    'plana_colmap_full': 'planA_colmap_full',
    'plana_colmap_108': 'planA_colmap_108',
    'planb_vggt_ba_108': 'planB_vggt_ba_108',
    'colmap_full': 'planA_colmap_full',
    'colmap_108': 'planA_colmap_108',
    'vggt_ba_108': 'planB_vggt_ba_108',
}
plot_df['setting'] = plot_df['setting'].replace(setting_alias)

SETTING_ORDER_LOCAL = ['planA_colmap_full', 'planA_colmap_108', 'planB_vggt_ba_108']
SETTING_LABELS_LOCAL = {
    'planA_colmap_full': 'COLMAP-full',
    'planA_colmap_108': 'COLMAP-108',
    'planB_vggt_ba_108': 'VGGT+BA-108',
}
SETTING_COLORS_LOCAL = {
    'planA_colmap_full': '#1f4e79',
    'planA_colmap_108': '#2e8b57',
    'planB_vggt_ba_108': '#9b1b45',
}
MODEL_ORDER_LOCAL = ['3dgs', 'scaffold']
MODEL_LABELS_LOCAL = {'3dgs': '3DGS', 'scaffold': 'Scaffold-GS'}
MODEL_MARKERS_LOCAL = {'3dgs': 'o', 'scaffold': 's'}

valid_settings = set(SETTING_ORDER_LOCAL)
valid_models = set(MODEL_ORDER_LOCAL)

plot_df = plot_df[plot_df['scene'].isin(SCENE_ORDER)].copy()
plot_df = plot_df[plot_df['setting'].isin(valid_settings)].copy()
plot_df = plot_df[plot_df['model'].isin(valid_models)].copy()

row_defs = []
for scene in SCENE_ORDER:
    for setting in SETTING_ORDER_LOCAL:
        sub = plot_df[(plot_df['scene'] == scene) & (plot_df['setting'] == setting)]
        if len(sub) > 0:
            row_defs.append({'scene': scene, 'setting': setting})

for i, rd in enumerate(row_defs):
    rd['y'] = i

yticklabels = [SETTING_LABELS_LOCAL[rd['setting']] for rd in row_defs]

scene_to_rows = {}
for scene in SCENE_ORDER:
    rows = [rd['y'] for rd in row_defs if rd['scene'] == scene]
    if rows:
        scene_to_rows[scene] = rows

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)

for ax, (metric, metric_label, higher_is_better) in zip(axes, METRICS):
    for j, scene in enumerate(SCENE_ORDER):
        rows = scene_to_rows.get(scene, [])
        if not rows:
            continue
        ymin = min(rows) - 0.5
        ymax = max(rows) + 0.5
        ax.axhspan(ymin, ymax, color='0.985' if j % 2 == 0 else '0.965', zorder=0)

    for scene in SCENE_ORDER[:-1]:
        rows = scene_to_rows.get(scene, [])
        if rows:
            ax.axhline(max(rows) + 0.5, color='0.72', lw=1.0, zorder=1)

    for rd in row_defs:
        scene = rd['scene']
        setting = rd['setting']
        y = rd['y']

        sub = plot_df[
            (plot_df['scene'] == scene) &
            (plot_df['setting'] == setting)
        ][['model', metric]].dropna()

        if len(sub) == 0:
            continue

        vals = {}
        for model in MODEL_ORDER_LOCAL:
            model_sub = sub[sub['model'] == model][metric]
            if len(model_sub) > 0:
                vals[model] = float(model_sub.iloc[0])

        if len(vals) == 2:
            ax.plot([vals['3dgs'], vals['scaffold']], [y, y], color='0.72', lw=1.4, zorder=2)

        if len(vals) > 0:
            best_model = max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)

            for model in MODEL_ORDER_LOCAL:
                if model not in vals:
                    continue
                x = vals[model]
                ax.scatter(x, y, s=80, marker=MODEL_MARKERS_LOCAL[model],
                          color=SETTING_COLORS_LOCAL[setting], edgecolors='white',
                          linewidths=0.8, zorder=3)
                if model == best_model:
                    ax.scatter(x, y, s=155, marker='o', facecolors='none',
                              edgecolors='black', linewidths=1, zorder=4)

    ax.set_title(metric_label, fontsize=16, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.55, alpha=0.30)
    ax.set_axisbelow(True)
    ax.set_xlabel(metric_label, fontsize=13)

    ax.set_yticks(range(len(row_defs)))
    if ax is axes[0]:
        ax.set_yticklabels(yticklabels, fontsize=11)
        ax.tick_params(axis='y', length=0, pad=6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    ax.invert_yaxis()

ax0 = axes[0]
for scene in SCENE_ORDER:
    rows = scene_to_rows.get(scene, [])
    if not rows:
        continue
    y_center = 0.5 * (min(rows) + max(rows))
    ax0.text(-0.17, y_center, SCENE_LABELS.get(scene, scene),
            transform=ax0.get_yaxis_transform(), ha='right', va='center',
            fontsize=12.5, fontweight='bold')

setting_handles = [
    Patch(facecolor=SETTING_COLORS_LOCAL[s], edgecolor='none', label=SETTING_LABELS_LOCAL[s])
    for s in SETTING_ORDER_LOCAL
]

model_handles = [
    Line2D([0], [0], marker=MODEL_MARKERS_LOCAL[m], color='black', linestyle='none',
          markerfacecolor='black', markersize=8, label=MODEL_LABELS_LOCAL[m])
    for m in MODEL_ORDER_LOCAL
]

best_handle = Line2D([0], [0], marker='o', color='black', linestyle='none',
                     markerfacecolor='none', markersize=10, markeredgewidth=1.6,
                     label='Better of the two')

fig.legend(handles=setting_handles, frameon=False, loc='upper center',
          bbox_to_anchor=(0.33, 1.04), ncol=3, fontsize=12,
          title='Derived setting', title_fontsize=12.5)

fig.legend(handles=model_handles + [best_handle], frameon=False, loc='upper center',
          bbox_to_anchor=(0.76, 1.04), ncol=3, fontsize=12,
          title='Model / highlight', title_fontsize=12.5)

fig.suptitle('Part 1 Final Test Metrics (Paired Comparison, VGGT+BA)', y=1.11, fontsize=17)
fig.tight_layout(rect=[0.08, 0.02, 1.00, 0.92])

out = MAIN_DIR / 'part1_final_test_metrics_paired_compact_vggt_ba.png'
fig.savefig(out, dpi=260, bbox_inches='tight')
plt.close(fig)
print(f'saved {out}')

# =========================================================
# 主图2: PSNR convergence (原有风格)
# =========================================================

print('\n=== 生成主图: PSNR convergence ===')

fig, axes = plt.subplots(len(SCENE_ORDER), len(MODEL_ORDER), figsize=(12.5, 10), sharex=True)

for r, scene in enumerate(SCENE_ORDER):
    for c, model in enumerate(MODEL_ORDER):
        ax = axes[r, c]
        sub = conv_df[
            (conv_df['scene'] == scene) &
            (conv_df['model'] == model) &
            (conv_df['split'] == 'test')
        ].copy()

        for setting in SETTING_ORDER:
            cur = sub[sub['setting'] == setting].sort_values('iter')
            if cur.empty:
                continue
            ax.plot(cur['iter'], cur['psnr'], color=SETTING_COLORS[setting],
                   linestyle='-', linewidth=2.2, label=SETTING_LABELS[setting])

        if r == 0:
            ax.set_title(MODEL_LABELS[model])
        if c == 0:
            ax.set_ylabel(f'{SCENE_LABELS[scene]}\nPSNR ↑')

        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.28)
        ax.set_axisbelow(True)

for ax in axes[-1, :]:
    ax.set_xlabel('Iteration')

setting_handles = [
    Line2D([0], [0], color=SETTING_COLORS[s], lw=2.5, label=SETTING_LABELS[s])
    for s in SETTING_ORDER
]
fig.legend(handles=setting_handles, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02))
fig.suptitle('Part 1 Test PSNR Convergence (VGGT+BA)', y=1.05, fontsize=14)
fig.tight_layout()

out = MAIN_DIR / 'part1_test_psnr_convergence_scene_by_model_vggt_ba.png'
fig.savefig(out, dpi=260, bbox_inches='tight')
plt.close(fig)
print(f'saved {out}')

# =========================================================
# 主图3: L1 convergence (原有风格)
# =========================================================

print('\n=== 生成主图: L1 convergence ===')

fig, axes = plt.subplots(len(SCENE_ORDER), len(MODEL_ORDER), figsize=(12.5, 10), sharex=True)

for r, scene in enumerate(SCENE_ORDER):
    for c, model in enumerate(MODEL_ORDER):
        ax = axes[r, c]
        sub = conv_df[
            (conv_df['scene'] == scene) &
            (conv_df['model'] == model) &
            (conv_df['split'] == 'test')
        ].copy()

        for setting in SETTING_ORDER:
            cur = sub[sub['setting'] == setting].sort_values('iter')
            if cur.empty:
                continue
            ax.plot(cur['iter'], cur['l1'], color=SETTING_COLORS[setting],
                   linestyle='-', linewidth=2.2, label=SETTING_LABELS[setting])

        if r == 0:
            ax.set_title(MODEL_LABELS[model])
        if c == 0:
            ax.set_ylabel(f'{SCENE_LABELS[scene]}\nL1 ↓')

        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.28)
        ax.set_axisbelow(True)

for ax in axes[-1, :]:
    ax.set_xlabel('Iteration')

setting_handles = [
    Line2D([0], [0], color=SETTING_COLORS[s], lw=2.5, label=SETTING_LABELS[s])
    for s in SETTING_ORDER
]
fig.legend(handles=setting_handles, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02))
fig.suptitle('Part 1 Test L1 Convergence (VGGT+BA)', y=1.05, fontsize=14)
fig.tight_layout()

out = MAIN_DIR / 'part1_test_l1_convergence_scene_by_model_vggt_ba.png'
fig.savefig(out, dpi=260, bbox_inches='tight')
plt.close(fig)
print(f'saved {out}')

# =========================================================
# 额外图: VGGT w/o BA vs VGGT+BA (对比图)
# =========================================================

print('\n=== 生成额外图: VGGT BA vs w/o BA ===')

vggt_settings = ['planB_vggt_108', 'planB_vggt_ba_108']
vggt_labels = {
    'planB_vggt_108': 'VGGT w/o BA',
    'planB_vggt_ba_108': 'VGGT+BA',
}
vggt_colors = {
    'planB_vggt_108': '#8b8b8b',
    'planB_vggt_ba_108': '#8b1e3f',
}

vggt_df = final_df[final_df['setting'].isin(vggt_settings)].copy()

row_defs_vggt = []
for scene in SCENE_ORDER:
    for setting in vggt_settings:
        sub = vggt_df[(vggt_df['scene'] == scene) & (vggt_df['setting'] == setting)]
        if len(sub) > 0:
            row_defs_vggt.append({'scene': scene, 'setting': setting})

for i, rd in enumerate(row_defs_vggt):
    rd['y'] = i

yticklabels_vggt = [vggt_labels[rd['setting']] for rd in row_defs_vggt]

scene_to_rows_vggt = {}
for scene in SCENE_ORDER:
    rows = [rd['y'] for rd in row_defs_vggt if rd['scene'] == scene]
    if rows:
        scene_to_rows_vggt[scene] = rows

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)

for ax, (metric, metric_label, higher_is_better) in zip(axes, METRICS):
    for j, scene in enumerate(SCENE_ORDER):
        rows = scene_to_rows_vggt.get(scene, [])
        if not rows:
            continue
        ymin = min(rows) - 0.5
        ymax = max(rows) + 0.5
        ax.axhspan(ymin, ymax, color='0.985' if j % 2 == 0 else '0.965', zorder=0)

    for scene in SCENE_ORDER[:-1]:
        rows = scene_to_rows_vggt.get(scene, [])
        if rows:
            ax.axhline(max(rows) + 0.5, color='0.72', lw=1.0, zorder=1)

    for rd in row_defs_vggt:
        scene = rd['scene']
        setting = rd['setting']
        y = rd['y']

        sub = vggt_df[
            (vggt_df['scene'] == scene) &
            (vggt_df['setting'] == setting)
        ][['model', metric]].dropna()

        if len(sub) == 0:
            continue

        vals = {}
        for model in MODEL_ORDER_LOCAL:
            model_sub = sub[sub['model'] == model][metric]
            if len(model_sub) > 0:
                vals[model] = float(model_sub.iloc[0])

        if len(vals) == 2:
            ax.plot([vals['3dgs'], vals['scaffold']], [y, y], color='0.72', lw=1.4, zorder=2)

        if len(vals) > 0:
            best_model = max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)

            for model in MODEL_ORDER_LOCAL:
                if model not in vals:
                    continue
                x = vals[model]
                ax.scatter(x, y, s=80, marker=MODEL_MARKERS_LOCAL[model],
                          color=vggt_colors[setting], edgecolors='white',
                          linewidths=0.8, zorder=3)
                if model == best_model:
                    ax.scatter(x, y, s=155, marker='o', facecolors='none',
                              edgecolors='black', linewidths=1, zorder=4)

    ax.set_title(metric_label, fontsize=16, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.55, alpha=0.30)
    ax.set_axisbelow(True)
    ax.set_xlabel(metric_label, fontsize=13)

    ax.set_yticks(range(len(row_defs_vggt)))
    if ax is axes[0]:
        ax.set_yticklabels(yticklabels_vggt, fontsize=11)
        ax.tick_params(axis='y', length=0, pad=6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    ax.invert_yaxis()

ax0 = axes[0]
for scene in SCENE_ORDER:
    rows = scene_to_rows_vggt.get(scene, [])
    if not rows:
        continue
    y_center = 0.5 * (min(rows) + max(rows))
    ax0.text(-0.17, y_center, SCENE_LABELS.get(scene, scene),
            transform=ax0.get_yaxis_transform(), ha='right', va='center',
            fontsize=12.5, fontweight='bold')

setting_handles = [
    Patch(facecolor=vggt_colors[s], edgecolor='none', label=vggt_labels[s])
    for s in vggt_settings
]

model_handles = [
    Line2D([0], [0], marker=MODEL_MARKERS_LOCAL[m], color='black', linestyle='none',
          markerfacecolor='black', markersize=8, label=MODEL_LABELS_LOCAL[m])
    for m in MODEL_ORDER_LOCAL
]

best_handle = Line2D([0], [0], marker='o', color='black', linestyle='none',
                     markerfacecolor='none', markersize=10, markeredgewidth=1.6,
                     label='Better of the two')

fig.legend(handles=setting_handles, frameon=False, loc='upper center',
          bbox_to_anchor=(0.33, 1.04), ncol=2, fontsize=12,
          title='VGGT setting', title_fontsize=12.5)

fig.legend(handles=model_handles + [best_handle], frameon=False, loc='upper center',
          bbox_to_anchor=(0.76, 1.04), ncol=3, fontsize=12,
          title='Model / highlight', title_fontsize=12.5)

fig.suptitle('VGGT w/o BA vs VGGT+BA Comparison', y=1.11, fontsize=17)
fig.tight_layout(rect=[0.08, 0.02, 1.00, 0.92])

out = APP_DIR / 'part1_vggt_ba_vs_woba_paired.png'
fig.savefig(out, dpi=260, bbox_inches='tight')
plt.close(fig)
print(f'saved {out}')

print('\n=== 完成 ===')
print(f'主图目录: {MAIN_DIR}')
print(f'额外图目录: {APP_DIR}')