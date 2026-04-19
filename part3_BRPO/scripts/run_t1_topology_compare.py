#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='T1 topology compare: old topology vs new brpo_joint_v1 topology')
    p.add_argument('--output-root', required=True)
    p.add_argument('--py', default='/home/bzhang512/miniconda3/envs/s3po-gs/bin/python')
    p.add_argument('--project-root', default='/home/bzhang512/CV_Project/part3_BRPO')
    p.add_argument('--internal-cache-root', default='/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache')
    p.add_argument('--ply-path', default='/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply')
    p.add_argument('--pseudo-cache', default='/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline')
    p.add_argument('--signal-v2-root', default='/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_joint_observation_signalv2')
    p.add_argument('--train-manifest', default='/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json')
    p.add_argument('--train-rgb-dir', default='/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb')
    p.add_argument('--init-pseudo-camera-states-json', default='/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json')
    p.add_argument('--num-real-views', type=int, default=2)
    p.add_argument('--num-pseudo-views', type=int, default=4)
    p.add_argument('--stageB-iters', type=int, default=120)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def run_cmd(cmd, cwd, env, dry_run=False):
    print('CMD:', ' '.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def read_replay_metrics(meta_path: Path):
    x = json.load(open(meta_path))
    r = x['compare_to_internal']['replay_eval']
    return {
        'psnr': float(r['avg_psnr']),
        'ssim': float(r['avg_ssim']),
        'lpips': float(r['avg_lpips']),
        'num_frames': int(r['num_frames']),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env['PYTHONPATH'] = '/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO'

    common = [
        args.py, 'scripts/run_pseudo_refinement_v2.py',
        '--ply_path', args.ply_path,
        '--pseudo_cache', args.pseudo_cache,
        '--signal_pipeline', 'brpo_v2',
        '--signal_v2_root', args.signal_v2_root,
        '--pseudo_observation_mode', 'brpo_joint_v1',
        '--stage_mode', 'stageB',
        '--stageA_iters', '0',
        '--stageB_iters', str(args.stageB_iters),
        '--stageB_post_switch_iter', '40',
        '--stageB_post_lr_scale_xyz', '0.3',
        '--stageB_post_lr_scale_opacity', '1.0',
        '--stageA_depth_loss_mode', 'source_aware',
        '--stageA_lambda_depth_seed', '1.0',
        '--stageA_lambda_depth_dense', '0.35',
        '--stageA_lambda_depth_fallback', '0.0',
        '--lambda_real', '1.0',
        '--lambda_pseudo', '1.0',
        '--num_real_views', str(args.num_real_views),
        '--num_pseudo_views', str(args.num_pseudo_views),
        '--train_manifest', args.train_manifest,
        '--train_rgb_dir', args.train_rgb_dir,
        '--init_pseudo_camera_states_json', args.init_pseudo_camera_states_json,
        '--init_pseudo_reference_mode', 'keep',
        '--pseudo_local_gating', 'spgm_keep',
        '--pseudo_local_gating_params', 'xyz',
        '--pseudo_local_gating_spgm_policy_mode', 'dense_keep',
        '--pseudo_local_gating_spgm_ranking_mode', 'v1',
        '--pseudo_local_gating_spgm_density_mode', 'opacity_support',
        '--pseudo_local_gating_spgm_cluster_keep_near', '1.0',
        '--pseudo_local_gating_spgm_cluster_keep_mid', '1.0',
        '--pseudo_local_gating_spgm_cluster_keep_far', '1.0',
        '--pseudo_local_gating_spgm_support_eta', '0.0',
        '--pseudo_local_gating_spgm_weight_floor', '0.25',
        '--seed', str(args.seed),
    ]

    arms = [
        {
            'name': 'old_topology_stageA5_mainline_control',
            'joint_topology_mode': 'off',
            'stageA5_role': 'legacy_mainline',
            'note': 'StageA.5 is treated as required mainline predecessor / control',
        },
        {
            'name': 'new_topology_brpo_joint_v1',
            'joint_topology_mode': 'brpo_joint_v1',
            'stageA5_role': 'optional_warmup_or_control',
            'note': 'StageA.5 handoff is reused only as optional warmup/control anchor; joint loop is the main topology',
        },
    ]

    summary = {
        'output_root': str(output_root),
        'compare_type': 'T1_topology_old_vs_new',
        'observation_mode': 'new_a1_brpo_joint_v1',
        'arms': {},
    }

    for arm in arms:
        out_dir = output_root / arm['name']
        if out_dir.exists() and not args.dry_run:
            shutil.rmtree(out_dir)
        cmd = common + ['--output_dir', str(out_dir), '--joint_topology_mode', arm['joint_topology_mode']]
        run_cmd(cmd, cwd=args.project_root, env=env, dry_run=args.dry_run)
        replay_cmd = [
            args.py, 'scripts/replay_internal_eval.py',
            '--internal-cache-root', args.internal_cache_root,
            '--stage-tag', 'after_opt',
            '--ply-path', str(out_dir / 'refined_gaussians.ply'),
            '--label', arm['name'],
            '--save-dir', str(out_dir / 'replay_eval'),
        ]
        run_cmd(replay_cmd, cwd=args.project_root, env=env, dry_run=args.dry_run)
        if not args.dry_run:
            metrics = read_replay_metrics(out_dir / 'replay_eval' / 'replay_eval_meta.json')
            summary['arms'][arm['name']] = {
                **arm,
                **metrics,
                'output_dir': str(out_dir),
            }

    if not args.dry_run:
        old_arm = summary['arms']['old_topology_stageA5_mainline_control']
        new_arm = summary['arms']['new_topology_brpo_joint_v1']
        summary['delta_new_vs_old'] = {
            'psnr': new_arm['psnr'] - old_arm['psnr'],
            'ssim': new_arm['ssim'] - old_arm['ssim'],
            'lpips': new_arm['lpips'] - old_arm['lpips'],
        }
        with open(output_root / 'compare_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
