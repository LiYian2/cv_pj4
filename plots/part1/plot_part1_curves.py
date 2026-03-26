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

base_order = ['PlanA-279', 'PlanA-96', 'PlanB-96']
method_order = ['3DGS', 'ScaffoldGS']
base_colors = {
    'PlanA-279': '#1f4e79',
    'PlanA-96': '#2e8b57',
    'PlanB-96': '#8b1e3f',
}
method_linestyles = {
    '3DGS': '-',
    'ScaffoldGS': '--',
}
method_markers = {
    '3DGS': 'o',
    'ScaffoldGS': 'D',
}
split_alpha = {'test': 1.0, 'train': 0.58}
split_width_scale = {'test': 1.0, 'train': 0.9}

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

def parse_series_name(series_name: str):
    if '_' in series_name:
        base, method = series_name.rsplit('_', 1)
    else:
        base, method = series_name, 'Unknown'
    return base, method

def series_present():
    seen = []
    for r in rows:
        s = r['series']
        if s not in seen:
            seen.append(s)
    ordered = []
    for base in base_order:
        for method in method_order:
            name = f'{base}_{method}'
            if name in seen:
                ordered.append(name)
    for s in seen:
        if s not in ordered:
            ordered.append(s)
    return ordered

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

series_list = series_present()

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for series in series_list:
    base, method = parse_series_name(series)
    pts = get_points(series, 'test')
    if not pts:
        continue
    ax.plot([p['iteration'] for p in pts], [p['PSNR'] for p in pts],
            color=base_colors.get(base, '#444444'), linewidth=2.2,
            linestyle=method_linestyles.get(method, '-'),
            marker=method_markers.get(method, 'o'), markersize=4.5,
            label=series)
style_axis(ax, 'PSNR ↑', 'Part 1: Test PSNR Convergence')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_TEST_PSNR, dpi=220, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for series in series_list:
    base, method = parse_series_name(series)
    pts = get_points(series, 'test')
    if not pts:
        continue
    ax.plot([p['iteration'] for p in pts], [p['L1'] for p in pts],
            color=base_colors.get(base, '#444444'), linewidth=2.2,
            linestyle=method_linestyles.get(method, '-'),
            marker=method_markers.get(method, 'o'), markersize=4.5,
            label=series)
style_axis(ax, 'L1 ↓', 'Part 1: Test L1 Convergence')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_TEST_L1, dpi=220, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
metrics = [('PSNR', 'PSNR ↑'), ('L1', 'L1 ↓')]
for ax, (metric, ylabel) in zip(axes, metrics):
    for series in series_list:
        base, method = parse_series_name(series)
        for split in ['test', 'train']:
            pts = get_points(series, split)
            if not pts:
                continue
            ax.plot([p['iteration'] for p in pts], [p[metric] for p in pts],
                    color=base_colors.get(base, '#444444'),
                    linestyle=method_linestyles.get(method, '-'),
                    linewidth=2.0 * split_width_scale[split],
                    alpha=split_alpha[split],
                    marker=method_markers.get(method, 'o') if split == 'test' else None,
                    markersize=3.8, markevery=2)
    style_axis(ax, ylabel)

axes[0].set_title('PSNR Curves')
axes[1].set_title('L1 Curves')

base_handles = [
    Line2D([0], [0], color=base_colors[b], lw=2.2, label=b)
    for b in base_order if any(parse_series_name(s)[0] == b for s in series_list)
]
method_handles = [
    Line2D([0], [0], color='black', lw=2.0, linestyle=method_linestyles[m], marker=method_markers[m], markersize=5, label=m)
    for m in method_order if any(parse_series_name(s)[1] == m for s in series_list)
]
split_handles = [
    Line2D([0], [0], color='black', lw=2.0, alpha=split_alpha['test'], label='Test'),
    Line2D([0], [0], color='black', lw=2.0, alpha=split_alpha['train'], label='Train'),
]

leg1 = axes[0].legend(handles=base_handles, loc='lower right', frameon=False, title='Init / Data')
axes[0].add_artist(leg1)
leg2 = axes[1].legend(handles=method_handles, loc='upper right', frameon=False, title='Method')
axes[1].add_artist(leg2)
axes[1].legend(handles=split_handles, loc='lower right', frameon=False, title='Split')

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT_COMBINED, dpi=240, bbox_inches='tight')
plt.close(fig)

print('Saved:', OUT_TEST_PSNR)
print('Saved:', OUT_TEST_L1)
print('Saved:', OUT_COMBINED)
