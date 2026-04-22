#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, S3PO_ROOT)
sys.path.insert(0, f"{S3PO_ROOT}/gaussian_splatting")

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from pseudo_branch.refine.pseudo_loss_v2 import build_stageA_loss, build_stageA_loss_source_aware
from pseudo_branch.refine.pseudo_refine_scheduler import StageAConfig
from scripts.run_pseudo_refinement_v2 import load_stageA_pseudo_views, sample_indices, diagnose_grad_contrib_for_sampled_views
from types import SimpleNamespace


def parse_args():
    p = argparse.ArgumentParser(description='Single-step StageA loss contribution diagnosis')
    p.add_argument('--ply_path', required=True)
    p.add_argument('--pseudo_cache', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--target_side', choices=['left', 'right', 'fused'], default='fused')
    p.add_argument('--confidence_mask_source', choices=['legacy', 'brpo'], default='brpo')
    p.add_argument('--brpo_mask_root', default=None)
    p.add_argument('--stageA_mask_mode', choices=['auto', 'train_mask', 'seed_support_only', 'legacy'], default='train_mask')
    p.add_argument('--stageA_target_depth_mode', choices=['auto', 'blended_depth', 'blended_depth_m5', 'render_depth_only', 'target_depth_for_refine', 'target_depth_for_refine_v2', 'target_depth', 'render_depth'], default='blended_depth')
    p.add_argument('--stageA_depth_loss_mode', choices=['legacy', 'source_aware'], default='source_aware')
    p.add_argument('--stageA_disable_depth', action='store_true')
    p.add_argument('--stageA_num_views', type=int, default=3)
    p.add_argument('--stageA_beta_rgb', type=float, default=0.7)
    p.add_argument('--stageA_lambda_pose', type=float, default=0.01)
    p.add_argument('--stageA_lambda_exp', type=float, default=0.001)
    p.add_argument('--stageA_lambda_abs_pose', type=float, default=0.0)
    p.add_argument('--stageA_lambda_abs_t', type=float, default=0.0)
    p.add_argument('--stageA_lambda_abs_r', type=float, default=0.0)
    p.add_argument('--stageA_abs_pose_robust', choices=['charbonnier', 'huber', 'l2'], default='charbonnier')
    p.add_argument('--stageA_abs_pose_scale_source', choices=['render_depth_trainmask_median', 'render_depth_valid_median', 'fixed'], default='render_depth_trainmask_median')
    p.add_argument('--stageA_abs_pose_fixed_scale', type=float, default=1.0)
    p.add_argument('--stageA_trans_reg_weight', type=float, default=1.0)
    p.add_argument('--stageA_lambda_depth_seed', type=float, default=1.0)
    p.add_argument('--stageA_lambda_depth_dense', type=float, default=0.35)
    p.add_argument('--stageA_lambda_depth_fallback', type=float, default=0.0)
    p.add_argument('--sh_degree', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pipeline_params = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')

    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply_path)

    pseudo_views, pseudo_manifest_info = load_stageA_pseudo_views(args)
    if not pseudo_views:
        raise RuntimeError('No pseudo views loaded')

    idxs = sample_indices(len(pseudo_views), args.stageA_num_views, rng)
    sampled_views = [pseudo_views[i] for i in idxs]

    cfg = StageAConfig(
        num_iterations=1,
        beta_rgb=args.stageA_beta_rgb,
        lambda_pose=args.stageA_lambda_pose,
        lambda_exp=args.stageA_lambda_exp,
        lambda_abs_pose=args.stageA_lambda_abs_pose,
        lambda_abs_t=args.stageA_lambda_abs_t,
        lambda_abs_r=args.stageA_lambda_abs_r,
        abs_pose_robust=args.stageA_abs_pose_robust,
        trans_reg_weight=args.stageA_trans_reg_weight,
        num_pseudo_views=args.stageA_num_views,
    )

    grad = diagnose_grad_contrib_for_sampled_views(args, cfg, sampled_views, gaussians, pipeline_params, bg)

    out = {
        'args': vars(args),
        'num_loaded_views': len(pseudo_views),
        'sampled_indices': idxs,
        'sampled_sample_ids': [int(v['sample_id']) for v in sampled_views],
        'sampled_scene_scales': [float(v.get('stageA_scene_scale', 1.0)) for v in sampled_views],
        'grad_contrib': grad,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f'[diagnose_stageA_loss_contrib] saved: {out_path}')
    for k in ['rgb', 'depth_total', 'depth_seed', 'depth_dense', 'pose_reg', 'abs_pose_trans', 'abs_pose_rot']:
        if k in grad:
            g = grad[k]
            print(f"  {k:14s} rot={g['grad_norm_rot']:.3e} trans={g['grad_norm_trans']:.3e} expA={g['grad_norm_exp_a']:.3e} expB={g['grad_norm_exp_b']:.3e} loss={g['loss_value']:.4e}")


if __name__ == '__main__':
    main()
