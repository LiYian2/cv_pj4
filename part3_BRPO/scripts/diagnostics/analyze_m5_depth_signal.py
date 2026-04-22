#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Analyze current BRPO depth signal dilution before M5 densify')
    p.add_argument('--pseudo-cache', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    pseudo_cache = Path(args.pseudo_cache)
    manifest = load_json(pseudo_cache / 'manifest.json')
    out = {
        'pseudo_cache': str(pseudo_cache),
        'samples': [],
    }
    agg = {
        'train_mask_coverage': [],
        'verified_depth_coverage': [],
        'verified_within_train_mask_ratio': [],
        'render_fallback_within_train_mask_ratio': [],
        'weighted_depth_l1_total': [],
        'weighted_depth_l1_verified_only': [],
        'weighted_depth_l1_fallback_only': [],
    }

    for sid in manifest['sample_ids']:
        sdir = pseudo_cache / 'samples' / str(sid)
        train = np.load(sdir / 'train_confidence_mask_brpo_fused.npy').astype(np.float32)
        render = np.load(sdir / 'render_depth.npy').astype(np.float32)
        target = np.load(sdir / 'target_depth_for_refine.npy').astype(np.float32)
        source = np.load(sdir / 'target_depth_for_refine_source_map.npy')

        train_pos = train > 0
        verified = source != 0
        fallback = source == 0
        abs_diff = np.abs(target - render)
        weighted = abs_diff * train

        denom_total = float(train.sum() + 1e-8)
        denom_verified = float((train * verified.astype(np.float32)).sum() + 1e-8)
        denom_fallback = float((train * fallback.astype(np.float32)).sum() + 1e-8)

        sample = {
            'sample_id': int(sid),
            'train_mask_coverage': float(train_pos.sum() / train.size),
            'verified_depth_coverage': float(verified.sum() / verified.size),
            'verified_within_train_mask_ratio': float((train_pos & verified).sum() / max(train_pos.sum(), 1)),
            'render_fallback_within_train_mask_ratio': float((train_pos & fallback).sum() / max(train_pos.sum(), 1)),
            'weighted_depth_l1_total': float(weighted.sum() / denom_total),
            'weighted_depth_l1_verified_only': float((weighted * verified.astype(np.float32)).sum() / denom_verified),
            'weighted_depth_l1_fallback_only': float((weighted * fallback.astype(np.float32)).sum() / denom_fallback),
            'weighted_verified_contribution_fraction': float((weighted * verified.astype(np.float32)).sum() / max(weighted.sum(), 1e-8)),
            'weighted_fallback_contribution_fraction': float((weighted * fallback.astype(np.float32)).sum() / max(weighted.sum(), 1e-8)),
        }
        out['samples'].append(sample)
        for k in agg:
            agg[k].append(sample[k])

    out['summary'] = {f'mean_{k}': float(np.mean(v)) for k, v in agg.items()}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
