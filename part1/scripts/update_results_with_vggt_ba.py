#!/usr/bin/env python3
"""
更新 part1 读结果脚本，包含 vggt_ba_108
直接运行此脚本生成新的 CSV 文件
"""

from pathlib import Path
import json
import re
import pandas as pd

CV_ROOT = Path('/home/bzhang512/CV_Project')
OUTPUT_ROOT = CV_ROOT / 'output' / 'part1'
RESULTS_ROOT = CV_ROOT / 'results' / 'part1'
FINAL_DIR = RESULTS_ROOT / 'final'
CONV_DIR = RESULTS_ROOT / 'convergence'

for d in [FINAL_DIR, CONV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print('OUTPUT_ROOT =', OUTPUT_ROOT)

# 1. Scan all results.json
result_files = sorted(OUTPUT_ROOT.rglob('results.json'))
print(f'num results.json = {len(result_files)}')

# 统计各 variant
variant_counts = {}
for p in result_files:
    parts = p.relative_to(OUTPUT_ROOT).parts
    if len(parts) >= 4:
        variant = parts[3] if parts[1] == 'planA' else parts[3]
        # 正确提取 variant
        # path: scene/plan/variant/model/run_name/results.json
        if len(parts) >= 6:
            variant = parts[2] if parts[1] == 'planA' else parts[2]
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

print('variant counts:', variant_counts)

# 2. Parse final metrics
def flatten_dict(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out

def parse_result_identity(path: Path):
    rel = path.relative_to(OUTPUT_ROOT)
    parts = rel.parts
    # expected: scene / plan / variant / model / run_name / results.json
    if len(parts) < 6:
        return None
    scene, plan, variant, model, run_name = parts[:5]
    experiment_name = '__'.join([scene, plan, variant, model, run_name])
    return {
        'scene': scene,
        'plan': plan,
        'variant': variant,
        'model': model,
        'run_name': run_name,
        'experiment_name': experiment_name,
        'result_path': str(path),
    }

final_rows = []
for path in result_files:
    identity = parse_result_identity(path)
    if identity is None:
        continue
    payload = json.loads(path.read_text(encoding='utf-8'))
    row = dict(identity)
    row.update(flatten_dict(payload))
    final_rows.append(row)

final_df = pd.DataFrame(final_rows)
print(f'final_df shape = {final_df.shape}')

# 3. Find matching log files
def find_log_file(row):
    log_dir = OUTPUT_ROOT / row['scene'] / row['plan'] / row['variant'] / row['model'] / 'logs'
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob('*.log'))
    if not candidates:
        return None
    tokens = [row['scene'], row['plan'], row['variant'], row['model']]
    for c in candidates:
        name = c.name
        if all(tok in name for tok in tokens):
            return c
    if len(candidates) == 1:
        return candidates[0]
    return None

if not final_df.empty:
    final_df['log_path'] = final_df.apply(find_log_file, axis=1)
    final_df['log_path'] = final_df['log_path'].astype('string')

# 4. Read convergence metrics from logs
pattern = re.compile(r'\[ITER\s+(\d+)\]\s+Evaluating\s+(test|train):\s+L1\s+([0-9eE.+-]+)\s+PSNR\s+([0-9eE.+-]+)')

conv_rows = []
for _, row in final_df.iterrows():
    log_path = row.get('log_path')
    if pd.isna(log_path) or not log_path:
        continue
    log_path = Path(log_path)
    if not log_path.exists():
        continue
    with log_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            conv_rows.append({
                'experiment_name': row['experiment_name'],
                'scene': row['scene'],
                'plan': row['plan'],
                'variant': row['variant'],
                'model': row['model'],
                'run_name': row['run_name'],
                'iter': int(m.group(1)),
                'split': m.group(2),
                'l1': float(m.group(3)),
                'psnr': float(m.group(4)),
                'log_path': str(log_path),
            })

conv_df = pd.DataFrame(conv_rows)
print(f'conv_df shape = {conv_df.shape}')

# 5. Export csv summaries
final_csv = FINAL_DIR / 'final_metrics_24.csv'
conv_all_csv = CONV_DIR / 'all_convergence_metrics_24.csv'

if not final_df.empty:
    final_df.to_csv(final_csv, index=False)
if not conv_df.empty:
    conv_df.to_csv(conv_all_csv, index=False)

# Per-experiment convergence files
if not conv_df.empty:
    for exp_name, subdf in conv_df.groupby('experiment_name'):
        out = CONV_DIR / f'{exp_name}.csv'
        subdf.sort_values(['split', 'iter']).to_csv(out, index=False)

print(f'final csv = {final_csv}')
print(f'convergence all csv = {conv_all_csv}')
print(f'Done! Total experiments: {len(final_df)}')

# 6. Quick summary
print('\n=== Variant Summary ===')
for variant in sorted(final_df['variant'].unique()):
    count = len(final_df[final_df['variant'] == variant])
    print(f'{variant}: {count} experiments')