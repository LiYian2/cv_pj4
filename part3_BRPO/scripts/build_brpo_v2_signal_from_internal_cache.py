#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pseudo_branch.flow_matcher import FlowMatcher
from pseudo_branch.brpo_reprojection_verify import find_neighbor_kfs
from pseudo_branch.brpo_v2_signal.rgb_mask_inference import (
    build_rgb_mask_from_correspondences,
    write_rgb_mask_outputs,
)
from pseudo_branch.brpo_v2_signal.depth_supervision_v2 import (
    build_depth_supervision_v2,
    write_depth_supervision_outputs,
)


def parse_args():
    p = argparse.ArgumentParser(description='Build isolated BRPO v2 RGB-mask + depth-supervision artifacts from fused outputs')
    p.add_argument('--internal-cache-root', required=True)
    p.add_argument('--prepare-root', required=True, help='Run root containing manifests/ and fusion/samples/<frame_id>/')
    p.add_argument('--stage-tag', choices=['before_opt', 'after_opt'], default='after_opt')
    p.add_argument('--output-root', default=None, help='Defaults to <prepare-root>/signal_v2')
    p.add_argument('--frame-ids', type=int, nargs='*', default=None)
    p.add_argument('--matcher-size', type=int, default=512)
    p.add_argument('--min-rgb-conf-for-depth', type=float, default=0.5)
    p.add_argument('--fallback-mode', choices=['render_depth', 'none'], default='render_depth')
    p.add_argument('--both-mode', choices=['weighted_by_fusion'], default='weighted_by_fusion')
    p.add_argument('--single-mode', choices=['single_branch_projected'], default='single_branch_projected')
    p.add_argument('--disable-continuous-depth-reweight', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_records(prepare_root: Path) -> List[Dict]:
    return load_json(prepare_root / 'manifests' / 'pseudo_selection_manifest.json')


def main():
    args = parse_args()
    internal_cache_root = Path(args.internal_cache_root)
    prepare_root = Path(args.prepare_root)
    output_root = Path(args.output_root) if args.output_root else (prepare_root / 'signal_v2')
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'diag').mkdir(parents=True, exist_ok=True)

    camera_states = load_json(internal_cache_root / 'camera_states.json')
    manifest = load_json(internal_cache_root / 'manifest.json')
    states_by_id = {int(s['frame_id']): s for s in camera_states}
    kf_indices = [int(x) for x in manifest['kf_indices']]

    records = load_records(prepare_root)
    if args.frame_ids:
        wanted = {int(x) for x in args.frame_ids}
        records = [r for r in records if int(r['frame_id']) in wanted]
    if not records:
        raise RuntimeError('No records selected for BRPO v2 signal build')

    if args.dry_run:
        print(json.dumps({
            'prepare_root': str(prepare_root),
            'output_root': str(output_root),
            'num_records': len(records),
            'frame_ids': [int(r['frame_id']) for r in records],
        }, indent=2))
        return

    matcher = FlowMatcher()
    summary = []

    for rec in records:
        frame_id = int(rec['frame_id'])
        image_name = rec.get('image_name', f'{frame_id:05d}.png')
        frame_out = output_root / f'frame_{frame_id:04d}'
        frame_out.mkdir(parents=True, exist_ok=True)

        pseudo_state = states_by_id[frame_id]
        left_ref_id = int(rec.get('left_ref_frame_id') or find_neighbor_kfs(frame_id, kf_indices)[0])
        right_ref_id = int(rec.get('right_ref_frame_id') or find_neighbor_kfs(frame_id, kf_indices)[1])
        left_ref_state = states_by_id[left_ref_id]
        right_ref_state = states_by_id[right_ref_id]

        fused_root = prepare_root / 'fusion' / 'samples' / str(frame_id)
        fused_rgb_path = fused_root / 'target_rgb_fused.png'
        render_depth_path = Path(rec['render_depth_path'])
        if not fused_rgb_path.exists():
            raise FileNotFoundError(f'Missing fused RGB for frame_id={frame_id}: {fused_rgb_path}')
        if not render_depth_path.exists():
            raise FileNotFoundError(f'Missing render depth for frame_id={frame_id}: {render_depth_path}')

        required_fusion = [
            fused_root / 'projected_depth_left.npy',
            fused_root / 'projected_depth_right.npy',
            fused_root / 'fusion_weight_left.npy',
            fused_root / 'fusion_weight_right.npy',
            fused_root / 'overlap_mask_left.npy',
            fused_root / 'overlap_mask_right.npy',
        ]
        for p in required_fusion:
            if not p.exists():
                raise FileNotFoundError(f'Missing fusion artifact for frame_id={frame_id}: {p}')

        rgb_result = build_rgb_mask_from_correspondences(
            fused_rgb_path=str(fused_rgb_path),
            left_ref_rgb_path=str(left_ref_state['image_path']),
            right_ref_rgb_path=str(right_ref_state['image_path']),
            matcher=matcher,
            size=int(args.matcher_size),
        )
        rgb_meta = {
            'frame_id': frame_id,
            'image_name': image_name,
            'left_ref_frame_id': left_ref_id,
            'right_ref_frame_id': right_ref_id,
            'summary': rgb_result['summary'],
            'matcher_meta': rgb_result['matcher_meta'],
            'signal_pipeline': 'brpo_v2_rgbmask_from_fused_rgb',
        }
        write_rgb_mask_outputs(frame_out, rgb_result, rgb_meta)

        render_depth = np.load(render_depth_path).astype(np.float32)
        projected_depth_left = np.load(fused_root / 'projected_depth_left.npy').astype(np.float32)
        projected_depth_right = np.load(fused_root / 'projected_depth_right.npy').astype(np.float32)
        fusion_weight_left = np.load(fused_root / 'fusion_weight_left.npy').astype(np.float32)
        fusion_weight_right = np.load(fused_root / 'fusion_weight_right.npy').astype(np.float32)
        overlap_mask_left = np.load(fused_root / 'overlap_mask_left.npy').astype(np.float32)
        overlap_mask_right = np.load(fused_root / 'overlap_mask_right.npy').astype(np.float32)

        depth_result = build_depth_supervision_v2(
            render_depth=render_depth,
            projected_depth_left=projected_depth_left,
            projected_depth_right=projected_depth_right,
            fusion_weight_left=fusion_weight_left,
            fusion_weight_right=fusion_weight_right,
            raw_rgb_confidence=rgb_result['raw_rgb_confidence_v2'],
            raw_rgb_confidence_cont=rgb_result['raw_rgb_confidence_cont_v2'],
            projected_valid_left=overlap_mask_left,
            projected_valid_right=overlap_mask_right,
            min_rgb_conf_for_depth=float(args.min_rgb_conf_for_depth),
            fallback_mode=args.fallback_mode,
            both_mode=args.both_mode,
            single_mode=args.single_mode,
            use_continuous_reweight=not args.disable_continuous_depth_reweight,
        )
        depth_meta = {
            'frame_id': frame_id,
            'image_name': image_name,
            'summary': depth_result['summary'],
            'inputs': {
                'render_depth_path': str(render_depth_path),
                'projected_depth_left_path': str(fused_root / 'projected_depth_left.npy'),
                'projected_depth_right_path': str(fused_root / 'projected_depth_right.npy'),
                'fusion_weight_left_path': str(fused_root / 'fusion_weight_left.npy'),
                'fusion_weight_right_path': str(fused_root / 'fusion_weight_right.npy'),
                'raw_rgb_confidence_v2_path': str(frame_out / 'raw_rgb_confidence_v2.npy'),
                'raw_rgb_confidence_cont_v2_path': str(frame_out / 'raw_rgb_confidence_cont_v2.npy'),
            },
            'signal_pipeline': 'brpo_v2_depth_supervision_from_fused_mask_plus_projected_depth',
        }
        write_depth_supervision_outputs(frame_out, depth_result, depth_meta)

        frame_summary = {
            'frame_id': frame_id,
            'left_ref_frame_id': left_ref_id,
            'right_ref_frame_id': right_ref_id,
            'rgb_mask_summary': rgb_result['summary'],
            'depth_summary': depth_result['summary'],
        }
        summary.append(frame_summary)

    write_json(output_root / 'summary.json', summary)
    write_json(output_root / 'summary_meta.json', {
        'signal_pipeline': 'brpo_v2',
        'prepare_root': str(prepare_root.resolve()),
        'internal_cache_root': str(internal_cache_root.resolve()),
        'stage_tag': args.stage_tag,
        'frame_ids': [int(r['frame_id']) for r in records],
        'matcher_size': int(args.matcher_size),
        'min_rgb_conf_for_depth': float(args.min_rgb_conf_for_depth),
        'fallback_mode': args.fallback_mode,
        'both_mode': args.both_mode,
        'single_mode': args.single_mode,
        'use_continuous_depth_reweight': not args.disable_continuous_depth_reweight,
        'num_frames': len(summary),
    })
    print(json.dumps({
        'output_root': str(output_root),
        'num_frames': len(summary),
        'frames': summary,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
