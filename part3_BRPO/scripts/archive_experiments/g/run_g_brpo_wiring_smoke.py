#!/usr/bin/env python3
"""G-BRPO-1 wiring smoke: minimal delayed direct BRPO path validation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='G-BRPO-1 wiring smoke')
    p.add_argument('--output-root', required=True)
    p.add_argument('--py', default='/home/bzhang512/miniconda3/envs/s3po-gs/bin/python')
    p.add_argument('--project-root', default='/home/bzhang512/CV_Project/part3_BRPO')
    p.add_argument('--ply-path', required=True)
    p.add_argument('--pseudo-cache', required=True)
    p.add_argument('--signal-v2-root', required=True)
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--train-rgb-dir', required=True)
    p.add_argument('--init-pseudo-camera-states-json', required=True)
    p.add_argument('--stageB-iters', type=int, default=5, help='Minimal iterations for wiring smoke')
    p.add_argument('--num-real-views', type=int, default=2)
    p.add_argument('--num-pseudo-views', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def run_cmd(cmd, cwd, env, dry_run=False):
    print('[CMD]', ' '.join(cmd[:8]), '...')
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env['PYTHONPATH'] = '/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO'

    # Minimal wiring smoke: 5 iterations, no replay, only check path execution + history output
    base_cmd = [
        args.py, 'scripts/run_pseudo_refinement_v2.py',
        '--ply_path', args.ply_path,
        '--pseudo_cache', args.pseudo_cache,
        '--signal_pipeline', 'brpo_v2',
        '--signal_v2_root', args.signal_v2_root,
        '--joint_topology_mode', 'brpo_joint_v1',
        '--stage_mode', 'stageB',
        '--stageA_iters', '0',
        '--stageB_iters', str(args.stageB_iters),
        '--stageB_post_switch_iter', '0',  # No post-switch for smoke
        '--stageA_depth_loss_mode', 'source_aware',
        '--stageA_lambda_depth_seed', '1.0',
        '--stageA_lambda_depth_dense', '0.35',
        '--lambda_real', '1.0',
        '--lambda_pseudo', '1.0',
        '--num_real_views', str(args.num_real_views),
        '--num_pseudo_views', str(args.num_pseudo_views),
        '--train_manifest', args.train_manifest,
        '--train_rgb_dir', args.train_rgb_dir,
        '--init_pseudo_camera_states_json', args.init_pseudo_camera_states_json,
        '--init_pseudo_reference_mode', 'keep',
        '--seed', str(args.seed),
        '--pseudo_local_gating', 'spgm_keep',
        '--pseudo_local_gating_params', 'xyz',
    ]

    arms = [
        {
            'name': 'legacy_delayed_b3_opacity',
            'note': 'Legacy B3 delayed deterministic opacity (baseline)',
            'extra_args': [
                '--pseudo_local_gating_spgm_manager_mode', 'deterministic_opacity_participation',
                '--pseudo_local_gating_spgm_score_semantics', 'legacy_v1',
                '--pseudo_local_gating_spgm_control_universe', 'active',
                '--pseudo_local_gating_spgm_action_semantics', 'deterministic_opacity_participation',
                '--pseudo_local_gating_spgm_timing_mode', 'delayed',
            ],
        },
        {
            'name': 'direct_brpo_delayed_stochastic',
            'note': 'Direct BRPO delayed stochastic Bernoulli opacity (G-BRPO-0)',
            'extra_args': [
                '--pseudo_local_gating_spgm_manager_mode', 'summary_only',  # Manager mode fallback
                '--pseudo_local_gating_spgm_score_semantics', 'brpo_unified_v1',
                '--pseudo_local_gating_spgm_control_universe', 'population_active',
                '--pseudo_local_gating_spgm_action_semantics', 'stochastic_bernoulli_opacity',
                '--pseudo_local_gating_spgm_timing_mode', 'delayed',
                '--pseudo_local_gating_spgm_drop_rate_global', '0.05',
                '--pseudo_local_gating_spgm_cluster_weight_near', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_mid', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_far', '1.0',
                '--pseudo_local_gating_spgm_sample_seed_mode', 'per_iter',
                '--pseudo_local_gating_spgm_brpo_alpha', '0.5',
            ],
        },
    ]

    summary = {
        'output_root': str(output_root),
        'smoke_type': 'G-BRPO-1_wiring_smoke',
        'stageB_iters': args.stageB_iters,
        'arms': {},
    }

    for arm in arms:
        out_dir = output_root / arm['name']
        cmd = base_cmd + ['--output_dir', str(out_dir)] + arm['extra_args']
        run_cmd(cmd, cwd=args.project_root, env=env, dry_run=args.dry_run)
        
        if not args.dry_run:
            # Check if history has direct BRPO fields
            history_path = out_dir / 'stageB_history.json'
            if history_path.exists():
                history = json.load(open(history_path))
                # Check for direct BRPO fields in first iteration
                if history.get('iterations'):
                    first_iter_idx = 0
                    direct_fields = [
                        'spgm_score_semantics', 'spgm_control_universe', 'spgm_control_ratio',
                        'spgm_unified_score_mean', 'spgm_unified_score_p50',
                        'spgm_drop_prob_mean', 'spgm_sampled_keep_ratio',
                    ]
                    found_fields = {}
                    for field in direct_fields:
                        values = history.get(field, [])
                        if values and len(values) > first_iter_idx:
                            found_fields[field] = values[first_iter_idx]
                    
                    summary['arms'][arm['name']] = {
                        'name': arm['name'],
                        'note': arm['note'],
                        'history_exists': True,
                        'direct_brpo_fields_found': found_fields,
                        'output_dir': str(out_dir),
                    }
                else:
                    summary['arms'][arm['name']] = {
                        'name': arm['name'],
                        'note': arm['note'],
                        'history_exists': True,
                        'error': 'No iterations in history',
                    }
            else:
                summary['arms'][arm['name']] = {
                    'name': arm['name'],
                    'note': arm['note'],
                    'history_exists': False,
                }

    if not args.dry_run:
        with open(output_root / 'wiring_smoke_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print('[SMOKE SUMMARY]')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()