#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path('/home/bzhang512/CV_Project/plots/part1')
CSV_PATH = BASE / 'convergence_long.csv'

OUT_TEST_PSNR = BASE / 'part1_test_psnr_curve_clean.png'
OUT_TEST_L1 = BASE / 'part1_test_l1_curve_clean.png'
OUT_COMBINED = BASE / 'part1_combined_psnr_l1.png'

rows = []
with CSV_PATH.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['iteration'] = int(row['iteration'])
        row['L1'] = float(row['L1'])
        row['PSNR'] = float(row['PSNR'])
        rows.append(row)

series_order = ['PlanA-279', 'PlanA-96', 'PlanB-96']
colors = {
    'PlanA-279': '#1f4e79',  # deep blue
    'PlanA-96': '#2e8b57',   # sea green
    'PlanB-96': '#8b1e3f',   # muted dark red
}
linestyles = {'test': '-', 'train': '--'}
markers = {'PlanA-279': 'o', 'PlanA-96': 's', 'PlanB-96': '^'}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def get_points(series, split):
    pts = [r for r in rows if r['series'] == series and r['split'] == split]
    pts.sort(key=lambda r: r['iteration'])
    return pts

def style_axis(ax, ylabel, title=None):
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.28)
    ax.set_axisbelow(True)

# 1) Clean test PSNR only
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for series in series_order:
    pts = get_points(series, 'test')
    ax.plot([p['iteration'] for p in pts], [p['PSNR'] for p in pts],
            color=colors[series], linewidth=2.2, marker=markers[series],
            markersize=4.5, label=series)
style_axis(ax, 'PSNR ↑', 'Part 1: Test PSNR Convergence')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_TEST_PSNR, dpi=220, bbox_inches='tight')
plt.close(fig)

# 2) Clean test L1 only
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for series in series_order:
    pts = get_points(series, 'test')
    ax.plot([p['iteration'] for p in pts], [p['L1'] for p in pts],
            color=colors[series], linewidth=2.2, marker=markers[series],
            markersize=4.5, label=series)
style_axis(ax, 'L1 ↓', 'Part 1: Test L1 Convergence')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_TEST_L1, dpi=220, bbox_inches='tight')
plt.close(fig)

# 3) Combined figure: PSNR and L1 side by side, each with train+test
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
metrics = [('PSNR', 'PSNR ↑'), ('L1', 'L1 ↓')]
for ax, (metric, ylabel) in zip(axes, metrics):
    for series in series_order:
        for split in ['test', 'train']:
            pts = get_points(series, split)
            ax.plot([p['iteration'] for p in pts], [p[metric] for p in pts],
                    color=colors[series], linestyle=linestyles[split], linewidth=2.0,
                    marker=markers[series], markersize=3.6 if split == 'test' else 0,
                    markevery=2)
    style_axis(ax, ylabel)

axes[0].set_title('PSNR Curves')
axes[1].set_title('L1 Curves')

method_handles = [
    Line2D([0], [0], color=colors[s], lw=2.2, marker=markers[s], markersize=5, label=s)
    for s in series_order
]
split_handles = [
    Line2D([0], [0], color='black', lw=2.0, linestyle='-', label='Test'),
    Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='Train'),
]

leg1 = axes[0].legend(handles=method_handles, loc='lower right', frameon=False, title='Method')
axes[0].add_artist(leg1)
axes[1].legend(handles=split_handles, loc='upper right', frameon=False, title='Split')

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT_COMBINED, dpi=240, bbox_inches='tight')
plt.close(fig)

print('Saved:', OUT_TEST_PSNR)
print('Saved:', OUT_TEST_L1)
print('Saved:', OUT_COMBINED)
