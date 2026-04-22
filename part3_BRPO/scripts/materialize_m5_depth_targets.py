#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pseudo_branch.target.brpo_depth_target import build_blended_target_depth_v2


def parse_args():
    p = argparse.ArgumentParser(description='Materialize M5 densified depth targets from existing pseudo_cache')
    p.add_argument('--pseudo-cache', required=True)
    p.add_argument('--output-summary', required=True)
    p.add_argument('--patch-size', type=int, default=11)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--min-seed-count', type=int, default=6)
    p.add_argument('--max-seed-delta-std', type=float, default=0.08)
    p.add_argument('--use-continuous-confidence', action='store_true')
    p.add_argument('--min-patch-confidence', type=float, default=0.0)
    p.add_argument('--both-seed-count-relax', type=int, default=0)
    p.add_argument('--single-std-tighten', type=float, default=1.0)
    return p.parse_args()


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def main():
    args = parse_args()
    pseudo_cache = Path(args.pseudo_cache)
    manifest = load_json(pseudo_cache / 'manifest.json')
    summary = {
        'pseudo_cache': str(pseudo_cache),
        'params': {
            'patch_size': int(args.patch_size),
            'stride': int(args.stride),
            'min_seed_count': int(args.min_seed_count),
            'max_seed_delta_std': float(args.max_seed_delta_std),
            'use_continuous_confidence': bool(args.use_continuous_confidence),
            'min_patch_confidence': float(args.min_patch_confidence),
            'both_seed_count_relax': int(args.both_seed_count_relax),
            'single_std_tighten': float(args.single_std_tighten),
        },
        'samples': [],
    }

    for sid in manifest['sample_ids']:
        sdir = pseudo_cache / 'samples' / str(sid)
        render_depth = np.load(sdir / 'render_depth.npy').astype(np.float32)
        projected_left = np.load(sdir / 'projected_depth_left.npy').astype(np.float32)
        projected_right = np.load(sdir / 'projected_depth_right.npy').astype(np.float32)
        valid_left = np.load(sdir / 'projected_depth_valid_left.npy').astype(np.float32)
        valid_right = np.load(sdir / 'projected_depth_valid_right.npy').astype(np.float32)
        train_mask = np.load(sdir / 'train_confidence_mask_brpo_fused.npy').astype(np.float32)
        continuous_confidence = None
        cont_path = sdir / 'raw_confidence_mask_brpo_cont_fused.npy'
        if args.use_continuous_confidence and cont_path.exists():
            continuous_confidence = np.load(cont_path).astype(np.float32)
        support_both = np.load(sdir / 'seed_support_both.npy').astype(np.float32) if (sdir / 'seed_support_both.npy').exists() else None
        support_single = np.load(sdir / 'seed_support_single.npy').astype(np.float32) if (sdir / 'seed_support_single.npy').exists() else None

        result = build_blended_target_depth_v2(
            render_depth=render_depth,
            projected_depth_left=projected_left,
            projected_depth_right=projected_right,
            train_confidence_mask=train_mask,
            valid_left=valid_left,
            valid_right=valid_right,
            patch_size=args.patch_size,
            stride=args.stride,
            min_seed_count=args.min_seed_count,
            max_seed_delta_std=args.max_seed_delta_std,
            continuous_confidence=continuous_confidence,
            support_both_mask=support_both,
            support_single_mask=support_single,
            min_patch_confidence=args.min_patch_confidence,
            both_seed_count_relax=args.both_seed_count_relax,
            single_std_tighten=args.single_std_tighten,
        )

        np.save(sdir / 'target_depth_for_refine_v2.npy', result['target_depth_for_refine_v2'])
        np.save(sdir / 'target_depth_dense_source_map.npy', result['target_depth_dense_source_map'])
        np.save(sdir / 'depth_correction_seed.npy', result['depth_correction_seed'])
        np.save(sdir / 'depth_correction_dense.npy', result['depth_correction_dense'])
        np.save(sdir / 'depth_seed_valid_mask.npy', result['depth_seed_valid_mask'])
        np.save(sdir / 'depth_dense_valid_mask.npy', result['depth_dense_valid_mask'])
        meta = {
            **result['summary'],
            'target_depth_for_refine_v2_path': f'samples/{sid}/target_depth_for_refine_v2.npy',
            'target_depth_dense_source_map_path': f'samples/{sid}/target_depth_dense_source_map.npy',
            'depth_correction_seed_path': f'samples/{sid}/depth_correction_seed.npy',
            'depth_correction_dense_path': f'samples/{sid}/depth_correction_dense.npy',
            'depth_seed_valid_mask_path': f'samples/{sid}/depth_seed_valid_mask.npy',
            'depth_dense_valid_mask_path': f'samples/{sid}/depth_dense_valid_mask.npy',
        }
        write_json(sdir / 'depth_densify_meta.json', meta)
        summary['samples'].append({'sample_id': int(sid), **result['summary']})

    if summary['samples']:
        numeric_keys = [
            k for k, v in summary['samples'][0].items()
            if k != 'sample_id' and _is_number(v)
        ]
        summary['aggregate'] = {
            f'mean_{k}': float(np.mean([float(s[k]) for s in summary['samples']]))
            for k in numeric_keys
        }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

