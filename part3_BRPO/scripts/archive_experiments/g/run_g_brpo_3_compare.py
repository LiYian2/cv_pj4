#!/usr/bin/env python3
"""G-BRPO-3: formal 5-arm compare with replay eval."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='G-BRPO-3 formal compare')
    p.add_argument('--output-root', required=True)
    p.add_argument('--py', default='/home/bzhang512/miniconda3/envs/s3po-gs/bin/python')
    p.add_argument('--project-root', default='/home/bzhang512/CV_Project/part3_BRPO')
    p.add_argument('--ply-path', required=True)
    p.add_argument('--pseudo-cache', required=True)
    p.add_argument('--signal-v2-root', required=True)
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--train-rgb-dir', required=True)
    p.add_argument('--init-pseudo-camera-states-json', required=True)
    p.add_argument('--stageB-iters', type=int, default=40)
    p.add_argument('--num-real-views', type=int, default=2)
    p.add_argument('--num-pseudo-views', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-replay', action='store_true', help='Run replay eval after StageB')
    p.add_argument('--internal-cache-root', default='')
    p.add_argument('--skip-arms', nargs='*', default=[], help='Skip specific arms by name')
    return p.parse_args()


def run_cmd(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env['PYTHONPATH'] = '/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO'

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
        '--stageB_post_switch_iter', '0',
        '--stageA_depth_loss_mode', 'source_aware',
        '--lambda_real', '1.0',
        '--lambda_pseudo', '1.0',
        '--num_real_views', str(args.num_real_views),
        '--num_pseudo_views', str(args.num_pseudo_views),
        '--train_manifest', args.train_manifest,
        '--train_rgb_dir', args.train_rgb_dir,
        '--init_pseudo_camera_states_json', args.init_pseudo_camera_states_json,
        '--init_pseudo_reference_mode', 'keep',
        '--seed', str(args.seed),
    ]

    # 5 arms
    arms = [
        {
            'name': 'baseline_summary_only',
            'note': 'Baseline: old A1 + new T1 + true summary_only control (no grad-mask or state action)',
            'extra_args': [
                '--pseudo_local_gating', 'spgm_keep',
                '--pseudo_local_gating_params', 'xyz',
                '--pseudo_local_gating_spgm_manager_mode', 'summary_only',
                '--pseudo_local_gating_spgm_action_semantics', 'summary_only',
            ],
        },
        {
            'name': 'legacy_delayed_opacity',
            'note': 'Legacy B3: delayed deterministic opacity + legacy_v1 score',
            'extra_args': [
                '--pseudo_local_gating', 'spgm_keep',
                '--pseudo_local_gating_params', 'xyz',
                '--pseudo_local_gating_spgm_manager_mode', 'deterministic_opacity_participation',
                '--pseudo_local_gating_spgm_score_semantics', 'legacy_v1',
                '--pseudo_local_gating_spgm_control_universe', 'active',
                '--pseudo_local_gating_spgm_action_semantics', 'deterministic_opacity_participation',
                '--pseudo_local_gating_spgm_timing_mode', 'delayed',
            ],
        },
        {
            'name': 'direct_brpo_delayed',
            'note': 'Direct BRPO delayed: brpo_unified_v1 + stochastic_bernoulli_opacity + delayed',
            'extra_args': [
                '--pseudo_local_gating', 'spgm_keep',
                '--pseudo_local_gating_params', 'xyz',
                '--pseudo_local_gating_spgm_score_semantics', 'brpo_unified_v1',
                '--pseudo_local_gating_spgm_control_universe', 'population_active',
                '--pseudo_local_gating_spgm_action_semantics', 'stochastic_bernoulli_opacity',
                '--pseudo_local_gating_spgm_timing_mode', 'delayed',
                '--pseudo_local_gating_spgm_drop_rate_global', '0.05',
                '--pseudo_local_gating_spgm_brpo_alpha', '0.5',
                '--pseudo_local_gating_spgm_cluster_weight_near', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_mid', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_far', '1.0',
                '--pseudo_local_gating_spgm_sample_seed_mode', 'per_iter',
            ],
        },
        {
            'name': 'direct_brpo_current_step',
            'note': 'Direct BRPO current_step: brpo_unified_v1 + stochastic_bernoulli_opacity + current_step_probe_loss',
            'extra_args': [
                '--pseudo_local_gating', 'spgm_keep',
                '--pseudo_local_gating_params', 'xyz',
                '--pseudo_local_gating_spgm_score_semantics', 'brpo_unified_v1',
                '--pseudo_local_gating_spgm_control_universe', 'population_active',
                '--pseudo_local_gating_spgm_action_semantics', 'stochastic_bernoulli_opacity',
                '--pseudo_local_gating_spgm_timing_mode', 'current_step_probe_loss',
                '--pseudo_local_gating_spgm_drop_rate_global', '0.05',
                '--pseudo_local_gating_spgm_brpo_alpha', '0.5',
                '--pseudo_local_gating_spgm_cluster_weight_near', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_mid', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_far', '1.0',
                '--pseudo_local_gating_spgm_sample_seed_mode', 'per_iter',
            ],
        },
        {
            'name': 'direct_brpo_current_step_aggressive',
            'note': 'Direct BRPO current_step aggressive: higher drop_rate_global=0.10',
            'extra_args': [
                '--pseudo_local_gating', 'spgm_keep',
                '--pseudo_local_gating_params', 'xyz',
                '--pseudo_local_gating_spgm_score_semantics', 'brpo_unified_v1',
                '--pseudo_local_gating_spgm_control_universe', 'population_active',
                '--pseudo_local_gating_spgm_action_semantics', 'stochastic_bernoulli_opacity',
                '--pseudo_local_gating_spgm_timing_mode', 'current_step_probe_loss',
                '--pseudo_local_gating_spgm_drop_rate_global', '0.10',
                '--pseudo_local_gating_spgm_brpo_alpha', '0.5',
                '--pseudo_local_gating_spgm_cluster_weight_near', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_mid', '1.0',
                '--pseudo_local_gating_spgm_cluster_weight_far', '1.0',
                '--pseudo_local_gating_spgm_sample_seed_mode', 'per_iter',
            ],
        },
    ]

    results = {
        'output_root': str(output_root),
        'stageB_iters': args.stageB_iters,
        'arms': {},
    }

    for arm in arms:
        if arm['name'] in args.skip_arms:
            print(f'[SKIP] {arm["name"]}')
            continue

        out_dir = output_root / arm['name']
        cmd = base_cmd + ['--output_dir', str(out_dir)] + arm['extra_args']
        print(f'[RUN] {arm["name"]}: {arm["note"]}')
        run_cmd(cmd, cwd=args.project_root, env=env)

        # Read history for quick summary
        history_path = out_dir / 'stageB_history.json'
        replay_path = out_dir / 'replay_eval/psnr/replay_internal/final_result.json'
        if history_path.exists():
            history = json.load(open(history_path))
            stageb_iter_count = history.get('iterations', [])
            if isinstance(stageb_iter_count, list):
                stageb_iter_count = len(stageb_iter_count)
            else:
                stageb_iter_count = int(history.get('iters', 0) or 0)
            arm_summary = {
                'name': arm['name'],
                'note': arm['note'],
                'output_dir': str(out_dir),
                'stageB_iters': stageb_iter_count,
                'final_loss_total': history.get('loss_total', [None])[-1],
                'final_loss_pseudo': history.get('loss_pseudo', [None])[-1],
            }
            if replay_path.exists():
                replay = json.load(open(replay_path))
                arm_summary.update({
                    'replay_avg_psnr': replay.get('avg_psnr'),
                    'replay_avg_ssim': replay.get('avg_ssim'),
                    'replay_avg_lpips': replay.get('avg_lpips'),
                    'replay_num_frames': replay.get('num_frames'),
                })
            results['arms'][arm['name']] = arm_summary

    # Save summary
    summary_path = output_root / 'g_brpo_3_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'[DONE] Summary saved to {summary_path}')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()