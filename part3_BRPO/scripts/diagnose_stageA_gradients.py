#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, S3PO_ROOT)
sys.path.insert(0, f"{S3PO_ROOT}/gaussian_splatting")

from pseudo_branch.pseudo_camera_state import make_viewpoint_trainable
from pseudo_branch.pseudo_loss_v2 import (
    build_stageA_loss,
    build_stageA_loss_source_aware,
)


def load_v1_module():
    script = Path(__file__).parent / 'run_pseudo_refinement.py'
    spec = importlib.util.spec_from_file_location('run_pseudo_refinement_v1', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description='Diagnose Stage A loss connectivity and gradients')
    p.add_argument('--ply_path', required=True)
    p.add_argument('--pseudo_cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--target-side', default='fused', choices=['left', 'right', 'fused'])
    p.add_argument('--sample-ids', type=int, nargs='*', default=None)
    return p.parse_args()


def _grad_norm(param):
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().cpu().item())


def _zero_grads(vp):
    for p in [vp.cam_rot_delta, vp.cam_trans_delta, vp.exposure_a, vp.exposure_b]:
        if p.grad is not None:
            p.grad.zero_()


def _to_python(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def main():
    args = parse_args()
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render

    v1 = load_v1_module()
    pseudo_views, manifest_info = v1.load_pseudo_viewpoints(args.pseudo_cache, args.target_side, 'brpo', None)
    if args.sample_ids:
        keep = set(int(x) for x in args.sample_ids)
        pseudo_views = [v for v in pseudo_views if int(v['sample_id']) in keep]
    if not pseudo_views:
        raise RuntimeError('No pseudo views selected for diagnosis')

    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(args.ply_path)
    bg = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)

    out = {'manifest_info': manifest_info, 'samples': []}
    for view in pseudo_views:
        make_viewpoint_trainable(view['vp'])
        vp = view['vp']
        sample_dir = Path(view['target_rgb_path']).parent

        # Load M5 assets if present.
        target_depth_v2 = sample_dir / 'target_depth_for_refine_v2.npy'
        source_map_v2 = sample_dir / 'target_depth_dense_source_map.npy'
        if target_depth_v2.exists():
            depth_target = np.load(target_depth_v2).astype(np.float32)
            source_map = np.load(source_map_v2)
            depth_mode = 'm5_v2'
        else:
            depth_target = np.load(sample_dir / 'target_depth_for_refine.npy').astype(np.float32)
            source_map = np.load(sample_dir / 'target_depth_for_refine_source_map.npy')
            depth_mode = 'm3_v1'

        conf = np.load(sample_dir / 'train_confidence_mask_brpo_fused.npy').astype(np.float32)

        base_pkg = render(vp, gaussians, pipe, bg)
        base_rgb = base_pkg['render'].detach().clone()
        base_depth = base_pkg['depth'].detach().clone()

        # Finite-change probe with manual perturbation.
        with torch.no_grad():
            vp.cam_rot_delta[:] = torch.tensor([0.01, 0.0, 0.0], device=vp.cam_rot_delta.device)
            vp.cam_trans_delta[:] = torch.tensor([0.0, 0.0, 0.01], device=vp.cam_trans_delta.device)
        pert_pkg = render(vp, gaussians, pipe, bg)
        rgb_delta = float(torch.abs(pert_pkg['render'] - base_rgb).mean().detach().cpu().item())
        depth_delta = float(torch.abs(pert_pkg['depth'] - base_depth).mean().detach().cpu().item())
        with torch.no_grad():
            vp.cam_rot_delta.zero_()
            vp.cam_trans_delta.zero_()
            vp.exposure_a.zero_()
            vp.exposure_b.zero_()

        # RGB-only gradient
        pkg = render(vp, gaussians, pipe, bg)
        loss_rgb, stats_rgb = build_stageA_loss(
            render_rgb=pkg['render'], render_depth=pkg['depth'],
            target_rgb=view['rgb'], target_depth=depth_target,
            confidence_mask=conf, viewpoint=vp,
            beta_rgb=1.0, lambda_pose=0.0, lambda_exp=0.0, trans_weight=1.0,
            use_depth=False,
        )
        loss_rgb.backward(retain_graph=True)
        grad_rgb = {
            'rot': _grad_norm(vp.cam_rot_delta),
            'trans': _grad_norm(vp.cam_trans_delta),
            'exp_a': _grad_norm(vp.exposure_a),
            'exp_b': _grad_norm(vp.exposure_b),
            'loss_rgb': stats_rgb['loss_rgb'],
        }
        _zero_grads(vp)

        # Depth legacy gradient
        loss_d_legacy, stats_d_legacy = build_stageA_loss(
            render_rgb=pkg['render'], render_depth=pkg['depth'],
            target_rgb=view['rgb'], target_depth=depth_target,
            confidence_mask=conf, viewpoint=vp,
            beta_rgb=0.0, lambda_pose=0.0, lambda_exp=0.0, trans_weight=1.0,
            use_depth=True,
        )
        loss_d_legacy.backward(retain_graph=True)
        grad_d_legacy = {
            'rot': _grad_norm(vp.cam_rot_delta),
            'trans': _grad_norm(vp.cam_trans_delta),
            'exp_a': _grad_norm(vp.exposure_a),
            'exp_b': _grad_norm(vp.exposure_b),
            'loss_depth': stats_d_legacy['loss_depth'],
        }
        _zero_grads(vp)

        # Depth source-aware gradient.
        loss_d_src, stats_d_src = build_stageA_loss_source_aware(
            render_rgb=pkg['render'], render_depth=pkg['depth'],
            target_rgb=view['rgb'], target_depth=depth_target,
            confidence_mask=conf, depth_source_map=source_map,
            viewpoint=vp, beta_rgb=0.0, lambda_pose=0.0, lambda_exp=0.0,
            trans_weight=1.0, lambda_depth_seed=1.0, lambda_depth_dense=0.35,
            lambda_depth_fallback=0.0, use_depth=True,
        )
        loss_d_src.backward(retain_graph=True)
        grad_d_src = {
            'rot': _grad_norm(vp.cam_rot_delta),
            'trans': _grad_norm(vp.cam_trans_delta),
            'exp_a': _grad_norm(vp.exposure_a),
            'exp_b': _grad_norm(vp.exposure_b),
            'loss_depth': stats_d_src['loss_depth'],
            'loss_depth_seed': stats_d_src['loss_depth_seed'],
            'loss_depth_dense': stats_d_src['loss_depth_dense'],
            'loss_depth_fallback': stats_d_src['loss_depth_fallback'],
        }
        _zero_grads(vp)

        # Regularization gradients.
        loss_reg, stats_reg = build_stageA_loss(
            render_rgb=pkg['render'], render_depth=pkg['depth'],
            target_rgb=view['rgb'], target_depth=depth_target,
            confidence_mask=conf, viewpoint=vp,
            beta_rgb=0.0, lambda_pose=1.0, lambda_exp=1.0, trans_weight=1.0,
            use_depth=False,
        )
        loss_reg.backward()
        grad_reg = {
            'rot': _grad_norm(vp.cam_rot_delta),
            'trans': _grad_norm(vp.cam_trans_delta),
            'exp_a': _grad_norm(vp.exposure_a),
            'exp_b': _grad_norm(vp.exposure_b),
            'loss_pose_reg': stats_reg['loss_pose_reg'],
            'loss_exp_reg': stats_reg['loss_exp_reg'],
        }
        _zero_grads(vp)

        out['samples'].append({
            'sample_id': int(view['sample_id']),
            'frame_id': int(view['frame_id']),
            'depth_mode': depth_mode,
            'finite_change_probe': {
                'rgb_mean_abs_change_at_pose_delta': rgb_delta,
                'depth_mean_abs_change_at_pose_delta': depth_delta,
            },
            'grad_rgb_only': grad_rgb,
            'grad_depth_legacy_only': grad_d_legacy,
            'grad_depth_source_aware_only': grad_d_src,
            'grad_regularization_only': grad_reg,
        })

    # aggregate means
    def mean_key(path):
        vals = []
        for s in out['samples']:
            cur = s
            for k in path:
                cur = cur[k]
            vals.append(float(cur))
        return float(np.mean(vals))

    out['summary'] = {
        'mean_rgb_probe_change': mean_key(['finite_change_probe', 'rgb_mean_abs_change_at_pose_delta']),
        'mean_depth_probe_change': mean_key(['finite_change_probe', 'depth_mean_abs_change_at_pose_delta']),
        'mean_grad_rgb_rot': mean_key(['grad_rgb_only', 'rot']),
        'mean_grad_rgb_trans': mean_key(['grad_rgb_only', 'trans']),
        'mean_grad_depth_legacy_rot': mean_key(['grad_depth_legacy_only', 'rot']),
        'mean_grad_depth_legacy_trans': mean_key(['grad_depth_legacy_only', 'trans']),
        'mean_grad_depth_src_rot': mean_key(['grad_depth_source_aware_only', 'rot']),
        'mean_grad_depth_src_trans': mean_key(['grad_depth_source_aware_only', 'trans']),
        'mean_grad_reg_rot': mean_key(['grad_regularization_only', 'rot']),
        'mean_grad_reg_trans': mean_key(['grad_regularization_only', 'trans']),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
