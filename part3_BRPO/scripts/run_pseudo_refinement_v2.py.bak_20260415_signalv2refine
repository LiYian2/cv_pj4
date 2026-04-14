#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, S3PO_ROOT)
sys.path.insert(0, f"{S3PO_ROOT}/gaussian_splatting")

from pseudo_branch.pseudo_camera_state import (
    make_viewpoint_trainable,
    export_view_state,
    apply_pose_residual_,
    load_exported_view_states,
    apply_loaded_view_state_,
    summarize_true_pose_deltas,
)
from pseudo_branch.pseudo_loss_v2 import build_stageA_loss, build_stageA_loss_source_aware
from pseudo_branch.pseudo_refine_scheduler import StageAConfig, StageA5Config, build_stageA_optimizer, build_stageA5_optimizers
from pseudo_branch.gaussian_param_groups import build_micro_gaussian_param_groups


def load_v1_module():
    script = Path(__file__).parent / 'run_pseudo_refinement.py'
    spec = importlib.util.spec_from_file_location('run_pseudo_refinement_v1', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description='Part3 BRPO refine v2 (Stage A first)')
    p.add_argument('--ply_path', required=True)
    p.add_argument('--pseudo_cache', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--target_side', choices=['left', 'right', 'fused'], default='fused')
    p.add_argument('--confidence_mask_source', choices=['legacy', 'brpo'], default='brpo')
    p.add_argument('--brpo_mask_root', default=None)
    p.add_argument('--stage_mode', choices=['stageA', 'stageA5', 'stageB'], default='stageA')
    p.add_argument('--stageA_iters', type=int, default=300)
    p.add_argument('--stageA_beta_rgb', type=float, default=0.7)
    p.add_argument('--stageA_lambda_pose', type=float, default=0.01)
    p.add_argument('--stageA_lambda_exp', type=float, default=0.001)
    p.add_argument('--stageA_lambda_abs_pose', type=float, default=0.0, help='Legacy single-scalar abs pose prior weight')
    p.add_argument('--stageA_lambda_abs_t', type=float, default=0.0)
    p.add_argument('--stageA_lambda_abs_r', type=float, default=0.0)
    p.add_argument('--stageA_abs_pose_robust', choices=['charbonnier', 'huber', 'l2'], default='charbonnier')
    p.add_argument('--stageA_abs_pose_scale_source', choices=['render_depth_trainmask_median', 'render_depth_valid_median', 'fixed'], default='render_depth_trainmask_median')
    p.add_argument('--stageA_abs_pose_fixed_scale', type=float, default=1.0)
    p.add_argument('--stageA_log_grad_contrib', action='store_true')
    p.add_argument('--stageA_log_grad_interval', type=int, default=20)
    p.add_argument('--stageA_trans_reg_weight', type=float, default=1.0)
    p.add_argument('--stageA_lr_rot', type=float, default=0.003)
    p.add_argument('--stageA_lr_trans', type=float, default=0.001)
    p.add_argument('--stageA_lr_exp', type=float, default=0.01)
    p.add_argument('--stageA_disable_depth', action='store_true')
    p.add_argument('--stageA_apply_pose_update', action='store_true', default=True)
    p.add_argument('--stageA_no_apply_pose_update', dest='stageA_apply_pose_update', action='store_false')
    p.add_argument(
        '--stageA_mask_mode',
        choices=['auto', 'train_mask', 'seed_support_only', 'legacy'],
        default='train_mask',
        help='Legacy shared mask selector; kept for backward compatibility when rgb/depth mask modes stay auto.',
    )
    p.add_argument('--stageA_rgb_mask_mode', choices=['auto', 'raw_confidence', 'seed_support_only', 'train_mask', 'legacy'], default='auto')
    p.add_argument('--stageA_depth_mask_mode', choices=['auto', 'seed_support_only', 'train_mask', 'legacy'], default='auto')
    p.add_argument('--stageA_confidence_variant', choices=['discrete', 'continuous'], default='discrete')
    p.add_argument(
        '--stageA_target_depth_mode',
        choices=['auto', 'blended_depth', 'blended_depth_m5', 'render_depth_only', 'target_depth_for_refine', 'target_depth_for_refine_v2', 'target_depth', 'render_depth'],
        default='blended_depth',
        help='Which depth target Stage A should actually consume.',
    )
    p.add_argument('--stageA_depth_loss_mode', choices=['legacy', 'source_aware'], default='legacy')
    p.add_argument('--stageA_lambda_depth_seed', type=float, default=1.0)
    p.add_argument('--stageA_lambda_depth_dense', type=float, default=0.35)
    p.add_argument('--stageA_lambda_depth_fallback', type=float, default=0.0)
    p.add_argument('--num_pseudo_views', type=int, default=4)
    p.add_argument('--stageA5_trainable_params', choices=['xyz', 'xyz_opacity'], default='xyz')
    p.add_argument('--stageA5_lr_xyz', type=float, default=1e-4)
    p.add_argument('--stageA5_lr_opacity', type=float, default=5e-4)
    p.add_argument('--stageA5_disable_densify', action='store_true', default=True)
    p.add_argument('--stageA5_disable_prune', action='store_true', default=True)
    p.add_argument('--stageB_iters', type=int, default=120)
    p.add_argument('--init_pseudo_camera_states_json', default=None, help='Optional pseudo_camera_states_*.json from a previous stage for sequential handoff')
    p.add_argument('--init_pseudo_reference_mode', choices=['keep', 'reset_to_loaded'], default='keep', help='Whether to keep original R0/T0 anchors or reset them to the loaded handoff state')
    p.add_argument('--train_manifest', default=None, help='split_manifest.json for sparse train views (real anchor)')
    p.add_argument('--train_rgb_dir', default=None, help='RGB dir for sparse train views; defaults to manifest files.rgb')
    p.add_argument('--lambda_real', type=float, default=1.0)
    p.add_argument('--lambda_pseudo', type=float, default=1.0)
    p.add_argument('--num_real_views', type=int, default=2)
    p.add_argument('--sh_degree', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def sample_indices(n_total: int, n_sample: int, rng: np.random.Generator):
    n_sample = max(0, min(n_total, n_sample))
    if n_sample == 0:
        return []
    return rng.choice(n_total, n_sample, replace=False).tolist()


def _resolve_depth_mode_alias(mode: str):
    if mode == 'blended_depth':
        return 'target_depth_for_refine'
    if mode == 'blended_depth_m5':
        return 'target_depth_for_refine_v2'
    if mode == 'render_depth_only':
        return 'render_depth'
    return mode


def resolve_depth_target_path(sample_dir: Path, mode: str):
    resolved_mode = _resolve_depth_mode_alias(mode)
    candidates = []
    if resolved_mode == 'target_depth_for_refine_v2':
        candidates = [('target_depth_for_refine_v2', sample_dir / 'target_depth_for_refine_v2.npy', sample_dir / 'target_depth_dense_source_map.npy')]
    elif resolved_mode == 'target_depth_for_refine':
        candidates = [('target_depth_for_refine', sample_dir / 'target_depth_for_refine.npy', sample_dir / 'target_depth_for_refine_source_map.npy')]
    elif resolved_mode == 'target_depth':
        candidates = [('target_depth', sample_dir / 'target_depth.npy', None)]
    elif resolved_mode == 'render_depth':
        candidates = [('render_depth', sample_dir / 'render_depth.npy', None)]
    else:
        candidates = [
            ('target_depth_for_refine_v2', sample_dir / 'target_depth_for_refine_v2.npy', sample_dir / 'target_depth_dense_source_map.npy'),
            ('target_depth_for_refine', sample_dir / 'target_depth_for_refine.npy', sample_dir / 'target_depth_for_refine_source_map.npy'),
            ('target_depth', sample_dir / 'target_depth.npy', None),
            ('render_depth', sample_dir / 'render_depth.npy', None),
        ]
    for kind, path, source_map_path in candidates:
        if path.exists():
            return kind, path, source_map_path
    raise FileNotFoundError(f'No depth target candidate found under {sample_dir} for mode={mode}')


def _mask_train_candidates(target_side: str):
    if target_side == 'fused':
        return ['train_confidence_mask_brpo_fused.npy', 'confidence_mask_brpo_fused.npy', 'confidence_mask_brpo.npy']
    if target_side == 'right':
        return ['train_confidence_mask_brpo_right.npy', 'confidence_mask_brpo_right.npy', 'confidence_mask_brpo.npy']
    return ['train_confidence_mask_brpo_left.npy', 'confidence_mask_brpo_left.npy', 'confidence_mask_brpo.npy']


def _compose_seed_support_mask(sample_dir: Path, target_side: str):
    seed_left = np.load(sample_dir / 'seed_support_left.npy').astype(np.float32)
    seed_right = np.load(sample_dir / 'seed_support_right.npy').astype(np.float32)
    seed_both = np.load(sample_dir / 'seed_support_both.npy').astype(np.float32)
    seed_single = np.load(sample_dir / 'seed_support_single.npy').astype(np.float32)

    if target_side == 'fused':
        conf = np.zeros_like(seed_both, dtype=np.float32)
        conf[seed_both > 0.5] = 1.0
        conf[seed_single > 0.5] = 0.5
        return conf, 'generated::seed_support_fused'

    if target_side == 'right':
        right_only = (seed_right > 0.5) & ~(seed_both > 0.5)
        conf = np.zeros_like(seed_both, dtype=np.float32)
        conf[seed_both > 0.5] = 1.0
        conf[right_only] = 0.5
        return conf, 'generated::seed_support_right'

    left_only = (seed_left > 0.5) & ~(seed_both > 0.5)
    conf = np.zeros_like(seed_both, dtype=np.float32)
    conf[seed_both > 0.5] = 1.0
    conf[left_only] = 0.5
    return conf, 'generated::seed_support_left'


def _mask_raw_candidates(target_side: str, confidence_variant: str):
    if confidence_variant == 'continuous':
        if target_side == 'fused':
            return ['raw_confidence_mask_brpo_cont_fused.npy']
        if target_side == 'right':
            return ['raw_confidence_mask_brpo_cont_right.npy']
        return ['raw_confidence_mask_brpo_cont_left.npy']
    if target_side == 'fused':
        return ['raw_confidence_mask_brpo_fused.npy', 'confidence_mask_brpo_fused.npy', 'confidence_mask_brpo.npy']
    if target_side == 'right':
        return ['raw_confidence_mask_brpo_right.npy', 'confidence_mask_brpo_right.npy', 'confidence_mask_brpo.npy']
    return ['raw_confidence_mask_brpo_left.npy', 'confidence_mask_brpo_left.npy', 'confidence_mask_brpo.npy']


def _resolve_effective_mask_mode(requested_mode: str, legacy_mode: str):
    if requested_mode != 'auto':
        return requested_mode
    if legacy_mode and legacy_mode != 'auto':
        return legacy_mode
    return 'train_mask'


def resolve_stageA_mask(sample_dir: Path, target_side: str, requested_mode: str, confidence_variant: str, default_conf: np.ndarray, default_path: str | None, default_kind: str | None, legacy_mode: str = 'train_mask'):
    requested_mode = _resolve_effective_mask_mode(requested_mode, legacy_mode)

    if requested_mode == 'legacy':
        if default_conf is None:
            raise FileNotFoundError(f'No legacy/default confidence mask available under {sample_dir}')
        return default_conf.astype(np.float32), default_path, default_kind or 'legacy_default', 'legacy', 'legacy'

    if requested_mode == 'train_mask':
        for name in _mask_train_candidates(target_side):
            path = sample_dir / name
            if path.exists():
                return np.load(path).astype(np.float32), str(path), f'sample_explicit_{name}', 'train_mask', 'discrete'
        raise FileNotFoundError(f'No train-mask candidate found under {sample_dir} for target_side={target_side}')

    if requested_mode == 'raw_confidence':
        for variant in ([confidence_variant, 'discrete'] if confidence_variant == 'continuous' else ['discrete']):
            for name in _mask_raw_candidates(target_side, variant):
                path = sample_dir / name
                if path.exists():
                    return np.load(path).astype(np.float32), str(path), f'sample_explicit_{name}', 'raw_confidence', variant
        raise FileNotFoundError(f'No raw-confidence candidate found under {sample_dir} for target_side={target_side}, variant={confidence_variant}')

    if requested_mode == 'seed_support_only':
        conf, kind = _compose_seed_support_mask(sample_dir, target_side)
        return conf, kind, kind, 'seed_support_only', 'discrete'

    raise ValueError(f'Unsupported stageA mask mode={requested_mode}')


def _summarize_source_map(source_map: np.ndarray | None):
    if source_map is None:
        return {'verified_ratio': None, 'render_fallback_ratio': None, 'seed_ratio': None, 'dense_ratio': None}
    source_map = np.asarray(source_map)
    total = float(source_map.size)
    return {
        'verified_ratio': float((source_map != 0).sum() / total),
        'render_fallback_ratio': float((source_map == 0).sum() / total),
        'seed_ratio': float(np.isin(source_map, [1, 2, 3]).sum() / total),
        'dense_ratio': float((source_map == 4).sum() / total),
    }


def compute_stageA_scene_scale(sample_dir: Path, conf_arr: np.ndarray, scale_source: str, fixed_scale: float):
    if scale_source == 'fixed':
        return max(float(fixed_scale), 1e-6), 'fixed'

    render_depth_path = sample_dir / 'render_depth.npy'
    if not render_depth_path.exists():
        return max(float(fixed_scale), 1e-6), 'fixed_no_render_depth'

    render_depth = np.load(render_depth_path).astype(np.float32)
    valid = render_depth > 1e-6
    if scale_source == 'render_depth_trainmask_median':
        valid = valid & (conf_arr > 0)

    vals = render_depth[valid]
    if vals.size == 0:
        return max(float(fixed_scale), 1e-6), 'fixed_empty_valid'

    return max(float(np.median(vals)), 1e-6), scale_source


def _grad_group_norm(params):
    total = 0.0
    for p in params:
        if p is None or p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return float(total ** 0.5)


def _collect_view_param_groups(sampled_views):
    rot = []
    trans = []
    exp_a = []
    exp_b = []
    seen = set()
    for view in sampled_views:
        sid = int(view['sample_id'])
        if sid in seen:
            continue
        seen.add(sid)
        vp = view['vp']
        rot.append(vp.cam_rot_delta)
        trans.append(vp.cam_trans_delta)
        exp_a.append(vp.exposure_a)
        exp_b.append(vp.exposure_b)
    return {'rot': rot, 'trans': trans, 'exp_a': exp_a, 'exp_b': exp_b}


def diagnose_grad_contrib_for_sampled_views(args, cfg, sampled_views, gaussians, pipeline_params, bg):
    from gaussian_renderer import render as gs_render
    term_accum = {}
    for view in sampled_views:
        pkg = gs_render(view['vp'], gaussians, pipeline_params, bg)
        if args.stageA_depth_loss_mode == 'source_aware' and view.get('target_depth_source_map') is not None:
            _, _, terms = build_stageA_loss_source_aware(
                render_rgb=pkg['render'],
                render_depth=pkg['depth'],
                target_rgb=view['rgb'],
                target_depth=view['depth_for_refine'],
                confidence_mask=view['conf'],
                depth_source_map=view['target_depth_source_map'],
                viewpoint=view['vp'],
                beta_rgb=cfg.beta_rgb,
                lambda_pose=cfg.lambda_pose,
                lambda_exp=cfg.lambda_exp,
                trans_weight=cfg.trans_reg_weight,
                lambda_depth_seed=args.stageA_lambda_depth_seed,
                lambda_depth_dense=args.stageA_lambda_depth_dense,
                lambda_depth_fallback=args.stageA_lambda_depth_fallback,
                use_depth=not args.stageA_disable_depth,
                lambda_abs_pose=cfg.lambda_abs_pose,
                lambda_abs_t=cfg.lambda_abs_t,
                lambda_abs_r=cfg.lambda_abs_r,
                abs_pose_robust=cfg.abs_pose_robust,
                scene_scale=view.get('stageA_scene_scale', 1.0),
                return_terms=True,
            )
        else:
            _, _, terms = build_stageA_loss(
                render_rgb=pkg['render'],
                render_depth=pkg['depth'],
                target_rgb=view['rgb'],
                target_depth=view['depth_for_refine'],
                confidence_mask=view['conf'],
                viewpoint=view['vp'],
                beta_rgb=cfg.beta_rgb,
                lambda_pose=cfg.lambda_pose,
                lambda_exp=cfg.lambda_exp,
                trans_weight=cfg.trans_reg_weight,
                use_depth=not args.stageA_disable_depth,
                lambda_abs_pose=cfg.lambda_abs_pose,
                lambda_abs_t=cfg.lambda_abs_t,
                lambda_abs_r=cfg.lambda_abs_r,
                abs_pose_robust=cfg.abs_pose_robust,
                scene_scale=view.get('stageA_scene_scale', 1.0),
                return_terms=True,
            )
        for k, v in terms.items():
            term_accum.setdefault(k, []).append(v)

    term_mean = {k: sum(v) / float(len(v)) for k, v in term_accum.items()}
    groups = _collect_view_param_groups(sampled_views)
    out = {}
    for term_name, term_loss in term_mean.items():
        for plist in groups.values():
            for pp in plist:
                if pp.grad is not None:
                    pp.grad.zero_()
        if term_loss.requires_grad:
            term_loss.backward(retain_graph=True)
        out[term_name] = {
            'grad_norm_rot': _grad_group_norm(groups['rot']),
            'grad_norm_trans': _grad_group_norm(groups['trans']),
            'grad_norm_exp_a': _grad_group_norm(groups['exp_a']),
            'grad_norm_exp_b': _grad_group_norm(groups['exp_b']),
            'loss_value': float(term_loss.detach().item()),
        }
    for plist in groups.values():
        for pp in plist:
            if pp.grad is not None:
                pp.grad.zero_()
    return out


def load_stageA_pseudo_views(args):
    v1 = load_v1_module()
    pseudo_views, pseudo_manifest_info = v1.load_pseudo_viewpoints(args.pseudo_cache, args.target_side, args.confidence_mask_source, args.brpo_mask_root)
    for view in pseudo_views:
        sample_dir = Path(view['target_rgb_path']).parent
        source_meta_path = sample_dir / 'source_meta.json'
        source_meta = json.load(open(source_meta_path)) if source_meta_path.exists() else {}

        rgb_conf, rgb_conf_path, rgb_conf_kind, rgb_conf_mode, rgb_conf_variant = resolve_stageA_mask(
            sample_dir, args.target_side, args.stageA_rgb_mask_mode, args.stageA_confidence_variant,
            view.get('conf'), view.get('confidence_path'), view.get('confidence_source_kind'), legacy_mode=args.stageA_mask_mode,
        )
        depth_conf, depth_conf_path, depth_conf_kind, depth_conf_mode, depth_conf_variant = resolve_stageA_mask(
            sample_dir, args.target_side, args.stageA_depth_mask_mode, 'discrete',
            view.get('conf'), view.get('confidence_path'), view.get('confidence_source_kind'), legacy_mode=args.stageA_mask_mode,
        )
        rgb_positive = rgb_conf[rgb_conf > 0]
        depth_positive = depth_conf[depth_conf > 0]
        view['conf'] = rgb_conf.astype(np.float32)
        view['conf_rgb'] = rgb_conf.astype(np.float32)
        view['conf_depth'] = depth_conf.astype(np.float32)
        view['confidence_path'] = rgb_conf_path
        view['confidence_source_kind'] = rgb_conf_kind
        view['stageA_mask_mode_effective'] = rgb_conf_mode
        view['stageA_rgb_mask_mode_effective'] = rgb_conf_mode
        view['stageA_depth_mask_mode_effective'] = depth_conf_mode
        view['stageA_rgb_confidence_variant_effective'] = rgb_conf_variant
        view['stageA_depth_confidence_variant_effective'] = depth_conf_variant
        view['rgb_confidence_path'] = rgb_conf_path
        view['depth_confidence_path'] = depth_conf_path
        view['rgb_confidence_source_kind'] = rgb_conf_kind
        view['depth_confidence_source_kind'] = depth_conf_kind
        view['confidence_nonzero_ratio'] = float((rgb_conf > 0).sum() / rgb_conf.size)
        view['confidence_mean_positive'] = float(rgb_positive.mean()) if rgb_positive.size > 0 else 0.0
        view['rgb_confidence_nonzero_ratio'] = float((rgb_conf > 0).sum() / rgb_conf.size)
        view['rgb_confidence_mean_positive'] = float(rgb_positive.mean()) if rgb_positive.size > 0 else 0.0
        view['depth_confidence_nonzero_ratio'] = float((depth_conf > 0).sum() / depth_conf.size)
        view['depth_confidence_mean_positive'] = float(depth_positive.mean()) if depth_positive.size > 0 else 0.0

        depth_kind, depth_for_refine, source_map_path = resolve_depth_target_path(sample_dir, args.stageA_target_depth_mode)
        depth_arr = np.load(depth_for_refine).astype(np.float32)
        depth_meta_path = sample_dir / 'target_depth_for_refine_meta.json'
        depth_meta = json.load(open(depth_meta_path)) if depth_meta_path.exists() else {}
        source_map = np.load(source_map_path) if source_map_path is not None and source_map_path.exists() else None
        source_summary = _summarize_source_map(source_map)

        view['source_meta'] = source_meta
        view['depth_meta'] = depth_meta
        view['target_depth_for_refine_kind'] = depth_kind
        view['stageA_target_depth_mode_effective'] = _resolve_depth_mode_alias(args.stageA_target_depth_mode)
        view['target_depth_for_refine_path'] = str(depth_for_refine)
        view['target_depth_for_refine_source_map_path'] = str(source_map_path) if source_map_path is not None and source_map_path.exists() else None
        view['target_depth_source_map'] = source_map
        view['depth_for_refine'] = depth_arr
        scene_scale, scene_scale_src = compute_stageA_scene_scale(
            sample_dir=sample_dir,
            conf_arr=depth_conf,
            scale_source=args.stageA_abs_pose_scale_source,
            fixed_scale=args.stageA_abs_pose_fixed_scale,
        )
        view['stageA_scene_scale'] = float(scene_scale)
        view['stageA_scene_scale_source_effective'] = scene_scale_src
        view['target_depth_nonzero_ratio'] = float((depth_arr > 1e-6).sum() / depth_arr.size)
        view['target_depth_verified_ratio'] = source_summary['verified_ratio'] if source_summary['verified_ratio'] is not None else depth_meta.get('verified_ratio')
        view['target_depth_render_fallback_ratio'] = source_summary['render_fallback_ratio'] if source_summary['render_fallback_ratio'] is not None else depth_meta.get('render_fallback_ratio')
        view['target_depth_seed_ratio'] = source_summary['seed_ratio']
        view['target_depth_dense_ratio'] = source_summary['dense_ratio']
        make_viewpoint_trainable(view['vp'])
    return pseudo_views, pseudo_manifest_info


def initialize_pseudo_views_from_saved_states(pseudo_views, init_states_json: str | None, reference_mode: str):
    summary = {
        'requested': bool(init_states_json),
        'source_json': init_states_json,
        'reference_mode': reference_mode,
        'loaded_count': 0,
        'missing_sample_ids': [],
        'frame_id_mismatches': [],
    }
    if not init_states_json:
        return summary
    loaded = load_exported_view_states(init_states_json)
    for view in pseudo_views:
        sid = int(view['sample_id'])
        if sid not in loaded:
            summary['missing_sample_ids'].append(sid)
            continue
        state = loaded[sid]
        if int(state.get('frame_id', sid)) != int(view.get('frame_id', sid)):
            summary['frame_id_mismatches'].append({
                'sample_id': sid,
                'loaded_frame_id': int(state.get('frame_id', sid)),
                'current_frame_id': int(view.get('frame_id', sid)),
            })
        apply_loaded_view_state_(view['vp'], state, reference_mode=reference_mode)
        summary['loaded_count'] += 1
    return summary


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render

    pipeline_params = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')

    print(f'[Part3 StageA] Loading PLY: {args.ply_path}')
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply_path)
    print(f'  Loaded {len(gaussians.get_xyz)} Gaussians (sh_degree={args.sh_degree})')

    pseudo_views, pseudo_manifest_info = load_stageA_pseudo_views(args)
    init_handoff_summary = initialize_pseudo_views_from_saved_states(
        pseudo_views,
        init_states_json=args.init_pseudo_camera_states_json,
        reference_mode=args.init_pseudo_reference_mode,
    )
    if not pseudo_views:
        raise RuntimeError(f'No pseudo viewpoints found under {args.pseudo_cache}')
    if init_handoff_summary['requested']:
        print(f"  Loaded prior pseudo states: {init_handoff_summary['loaded_count']}/{len(pseudo_views)} from {args.init_pseudo_camera_states_json} (reference_mode={args.init_pseudo_reference_mode})")
        if init_handoff_summary['missing_sample_ids']:
            print(f"  Missing prior states for sample_ids: {init_handoff_summary['missing_sample_ids']}")
        if init_handoff_summary['frame_id_mismatches']:
            print(f"  Frame-id mismatches in prior states: {init_handoff_summary['frame_id_mismatches']}")
    print(f'  Loaded {len(pseudo_views)} pseudo viewpoints (target_side={args.target_side}, confidence_mask_source={args.confidence_mask_source})')
    print(
        '  StageA effective sources: '
        f"mask_mode(legacy)={args.stageA_mask_mode}, rgb_mask_mode={args.stageA_rgb_mask_mode}, depth_mask_mode={args.stageA_depth_mask_mode}, conf_variant={args.stageA_confidence_variant}, depth_mode={args.stageA_target_depth_mode}, depth_loss_mode={args.stageA_depth_loss_mode}, apply_pose_update={args.stageA_apply_pose_update}, lambda_abs_pose(legacy)={args.stageA_lambda_abs_pose}, lambda_abs_t={args.stageA_lambda_abs_t}, lambda_abs_r={args.stageA_lambda_abs_r}, abs_robust={args.stageA_abs_pose_robust}, "
        f"mean_mask_cov={np.mean([v['confidence_nonzero_ratio'] for v in pseudo_views]):.4f}, "
        f"mean_depth_verified={np.mean([v.get('target_depth_verified_ratio') or 0.0 for v in pseudo_views]):.4f}, "
        f"mean_depth_dense={np.mean([v.get('target_depth_dense_ratio') or 0.0 for v in pseudo_views]):.4f}"
    )

    init_states = [export_view_state(v) for v in pseudo_views]
    save_json(output_dir / 'pseudo_camera_states_init.json', init_states)

    cfg_cls = StageA5Config if args.stage_mode == "stageA5" else StageAConfig
    cfg = cfg_cls(
        num_iterations=args.stageA_iters,
        beta_rgb=args.stageA_beta_rgb,
        lambda_pose=args.stageA_lambda_pose,
        lambda_exp=args.stageA_lambda_exp,
        lambda_abs_pose=args.stageA_lambda_abs_pose,
        lambda_abs_t=args.stageA_lambda_abs_t,
        lambda_abs_r=args.stageA_lambda_abs_r,
        abs_pose_robust=args.stageA_abs_pose_robust,
        trans_reg_weight=args.stageA_trans_reg_weight,
        lr_rot=args.stageA_lr_rot,
        lr_trans=args.stageA_lr_trans,
        lr_exp=args.stageA_lr_exp,
        num_pseudo_views=args.num_pseudo_views,
        **({
            "trainable_params": args.stageA5_trainable_params,
            "lr_xyz": args.stageA5_lr_xyz,
            "lr_opacity": args.stageA5_lr_opacity,
            "disable_densify": args.stageA5_disable_densify,
            "disable_prune": args.stageA5_disable_prune,
        } if args.stage_mode == "stageA5" else {}),
    )
    if args.stage_mode == "stageA5":
        optimizer, gaussian_optimizer = build_stageA5_optimizers(pseudo_views, gaussians, cfg)
    else:
        optimizer = build_stageA_optimizer(pseudo_views, cfg)
        gaussian_optimizer = None

    history = {
        'iterations': [],
        'loss_total': [], 'loss_rgb': [], 'loss_depth': [],
        'loss_depth_seed': [], 'loss_depth_dense': [], 'loss_depth_fallback': [],
        'loss_pose_reg': [], 'loss_abs_pose_reg': [],
        'loss_abs_pose_trans': [], 'loss_abs_pose_rot': [],
        'abs_pose_rho_norm': [], 'abs_pose_theta_norm': [], 'scene_scale_used': [],
        'loss_exp_reg': [],
        'sampled_sample_ids': [], 'num_pose_updates_applied': [],
        'grad_contrib': [],
        'grad_norm_xyz': []
    }

    for it in tqdm(range(1, cfg.num_iterations + 1), desc='StageA pseudo pose+exposure'):
        optimizer.zero_grad(set_to_none=True)
        if gaussian_optimizer is not None:
            gaussian_optimizer.zero_grad(set_to_none=True)
        pseudo_indices = sample_indices(len(pseudo_views), cfg.num_pseudo_views, rng)
        if not pseudo_indices:
            raise RuntimeError('No pseudo viewpoints sampled in Stage A')

        per_view_losses = []
        per_view_stats = []
        sampled_ids = []
        for pidx in pseudo_indices:
            view = pseudo_views[pidx]
            sampled_ids.append(int(view['sample_id']))
            pkg = render(view['vp'], gaussians, pipeline_params, bg)
            if args.stageA_depth_loss_mode == 'source_aware' and view.get('target_depth_source_map') is not None:
                loss, stats = build_stageA_loss_source_aware(
                    render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], depth_source_map=view['target_depth_source_map'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, lambda_depth_seed=args.stageA_lambda_depth_seed, lambda_depth_dense=args.stageA_lambda_depth_dense, lambda_depth_fallback=args.stageA_lambda_depth_fallback, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0),
                )
            else:
                loss, stats = build_stageA_loss(
                    render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0),
                )
            per_view_losses.append(loss)
            per_view_stats.append(stats)

        total_loss = torch.stack(per_view_losses).mean()
        total_loss.backward()
        optimizer.step()
        if gaussian_optimizer is not None:
            gaussian_optimizer.step()

        num_pose_updates = 0
        if args.stageA_apply_pose_update:
            seen = set()
            for pidx in pseudo_indices:
                view = pseudo_views[pidx]
                sid = int(view['sample_id'])
                if sid in seen:
                    continue
                apply_pose_residual_(view['vp'])
                seen.add(sid)
            num_pose_updates = len(seen)

        avg_stats = {key: float(np.mean([s[key] for s in per_view_stats])) for key in ['loss_total', 'loss_rgb', 'loss_depth', 'loss_depth_seed', 'loss_depth_dense', 'loss_depth_fallback', 'loss_pose_reg', 'loss_abs_pose_reg', 'loss_abs_pose_trans', 'loss_abs_pose_rot', 'abs_pose_rho_norm', 'abs_pose_theta_norm', 'scene_scale_used', 'loss_exp_reg']}
        for key in avg_stats:
            history[key].append(avg_stats[key])
        history['iterations'].append(it)
        history['sampled_sample_ids'].append(sampled_ids)
        history['num_pose_updates_applied'].append(int(num_pose_updates))
        history['grad_norm_xyz'].append(float(torch.norm(gaussians._xyz.grad).detach().cpu().item()) if gaussians._xyz.grad is not None else 0.0)

        if args.stageA_log_grad_contrib and (it == 1 or it % max(1, int(args.stageA_log_grad_interval)) == 0):
            sampled_views = [pseudo_views[i] for i in pseudo_indices]
            grad_contrib = diagnose_grad_contrib_for_sampled_views(args, cfg, sampled_views, gaussians, pipeline_params, bg)
            history['grad_contrib'].append({'iter': int(it), 'terms': grad_contrib})
            rgb_rot = grad_contrib.get('rgb', {}).get('grad_norm_rot', 0.0)
            depth_rot = grad_contrib.get('depth_total', {}).get('grad_norm_rot', 0.0)
            abs_rot = grad_contrib.get('abs_pose_rot', {}).get('grad_norm_rot', 0.0)
            print(f"    [grad] iter {it}: rot rgb={rgb_rot:.3e}, depth={depth_rot:.3e}, abs_rot={abs_rot:.3e}")

        if it == 1 or it % 20 == 0:
            print(
                f"  iter {it}: total={avg_stats['loss_total']:.4f}, rgb={avg_stats['loss_rgb']:.4f}, depth={avg_stats['loss_depth']:.4f}, "
                f"depth_seed={avg_stats['loss_depth_seed']:.4f}, depth_dense={avg_stats['loss_depth_dense']:.4f}, depth_fb={avg_stats['loss_depth_fallback']:.4f}, "
                f"pose_reg={avg_stats['loss_pose_reg']:.4f}, abs_pose={avg_stats['loss_abs_pose_reg']:.4f} (t={avg_stats['loss_abs_pose_trans']:.4f}, r={avg_stats['loss_abs_pose_rot']:.4f}), "
                f"abs_norm(rho={avg_stats['abs_pose_rho_norm']:.4f}, theta={avg_stats['abs_pose_theta_norm']:.4f}), scale={avg_stats['scene_scale_used']:.4f}, exp_reg={avg_stats['loss_exp_reg']:.4f}, pose_updates={num_pose_updates}"
            )

    stageA_states = [export_view_state(v) for v in pseudo_views]
    save_json(output_dir / 'pseudo_camera_states_stageA.json', stageA_states)

    true_pose_delta_summary = summarize_true_pose_deltas(init_states, stageA_states)

    stageA_history = {
        'args': vars(args), 'stage_mode': args.stage_mode, 'pseudo_manifest_info': pseudo_manifest_info, 'num_pseudo_viewpoints_loaded': len(pseudo_views), 'init_handoff_summary': init_handoff_summary,
        'effective_source_summary': {
            'stageA_mask_mode_requested': args.stageA_mask_mode,
            'stageA_rgb_mask_mode_requested': args.stageA_rgb_mask_mode,
            'stageA_depth_mask_mode_requested': args.stageA_depth_mask_mode,
            'stageA_confidence_variant_requested': args.stageA_confidence_variant,
            'stageA_target_depth_mode_requested': args.stageA_target_depth_mode,
            'stageA_depth_loss_mode': args.stageA_depth_loss_mode,
            'stageA_apply_pose_update': bool(args.stageA_apply_pose_update),
            'stageA_disable_depth': bool(args.stageA_disable_depth),
            'stageA_lambda_abs_pose': float(args.stageA_lambda_abs_pose),
            'stageA_lambda_abs_t': float(args.stageA_lambda_abs_t),
            'stageA_lambda_abs_r': float(args.stageA_lambda_abs_r),
            'stageA_abs_pose_robust': args.stageA_abs_pose_robust,
            'stageA_abs_pose_scale_source': args.stageA_abs_pose_scale_source,
            'stageA_abs_pose_fixed_scale': float(args.stageA_abs_pose_fixed_scale),
            'mean_stageA_scene_scale': float(np.mean([v.get('stageA_scene_scale', 1.0) for v in pseudo_views])),
            'mean_confidence_nonzero_ratio': float(np.mean([v.get('confidence_nonzero_ratio', 0.0) for v in pseudo_views])),
            'mean_target_depth_verified_ratio': float(np.mean([(v.get('target_depth_verified_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_render_fallback_ratio': float(np.mean([(v.get('target_depth_render_fallback_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_seed_ratio': float(np.mean([(v.get('target_depth_seed_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_dense_ratio': float(np.mean([(v.get('target_depth_dense_ratio') or 0.0) for v in pseudo_views])),
        },
        'pseudo_sample_meta': [
            {
                'sample_id': int(v['sample_id']), 'frame_id': int(v['frame_id']), 'target_rgb_path': v.get('target_rgb_path'),
                'target_depth_for_refine_path': v.get('target_depth_for_refine_path'), 'target_depth_for_refine_kind': v.get('target_depth_for_refine_kind'), 'stageA_target_depth_mode_effective': v.get('stageA_target_depth_mode_effective'),
                'target_depth_nonzero_ratio': v.get('target_depth_nonzero_ratio'), 'target_depth_verified_ratio': v.get('target_depth_verified_ratio'), 'target_depth_render_fallback_ratio': v.get('target_depth_render_fallback_ratio'), 'target_depth_seed_ratio': v.get('target_depth_seed_ratio'), 'target_depth_dense_ratio': v.get('target_depth_dense_ratio'),
                'target_depth_for_refine_source': v.get('source_meta', {}).get('target_depth_for_refine_source'), 'target_depth_for_refine_source_map_path': v.get('target_depth_for_refine_source_map_path'), 'confidence_path': v.get('confidence_path'), 'confidence_source_kind': v.get('confidence_source_kind'), 'stageA_mask_mode_effective': v.get('stageA_mask_mode_effective'), 'stageA_rgb_mask_mode_effective': v.get('stageA_rgb_mask_mode_effective'), 'stageA_depth_mask_mode_effective': v.get('stageA_depth_mask_mode_effective'), 'stageA_rgb_confidence_variant_effective': v.get('stageA_rgb_confidence_variant_effective'), 'stageA_depth_confidence_variant_effective': v.get('stageA_depth_confidence_variant_effective'), 'rgb_confidence_path': v.get('rgb_confidence_path'), 'depth_confidence_path': v.get('depth_confidence_path'), 'rgb_confidence_nonzero_ratio': v.get('rgb_confidence_nonzero_ratio'), 'depth_confidence_nonzero_ratio': v.get('depth_confidence_nonzero_ratio'), 'confidence_nonzero_ratio': v.get('confidence_nonzero_ratio'), 'confidence_mean_positive': v.get('confidence_mean_positive'), 'stageA_scene_scale': v.get('stageA_scene_scale'), 'stageA_scene_scale_source_effective': v.get('stageA_scene_scale_source_effective'),
            }
            for v in pseudo_views
        ],
        'history': history, 'stageA_disable_depth': bool(args.stageA_disable_depth), 'stageA_mask_mode': args.stageA_mask_mode, 'stageA_rgb_mask_mode': args.stageA_rgb_mask_mode, 'stageA_depth_mask_mode': args.stageA_depth_mask_mode, 'stageA_confidence_variant': args.stageA_confidence_variant, 'stageA_target_depth_mode': args.stageA_target_depth_mode, 'stageA_depth_loss_mode': args.stageA_depth_loss_mode, 'stageA_apply_pose_update': bool(args.stageA_apply_pose_update), 'stageA_lambda_depth_seed': float(args.stageA_lambda_depth_seed), 'stageA_lambda_depth_dense': float(args.stageA_lambda_depth_dense), 'stageA_lambda_depth_fallback': float(args.stageA_lambda_depth_fallback), 'stageA_lambda_abs_pose': float(args.stageA_lambda_abs_pose), 'stageA_lambda_abs_t': float(args.stageA_lambda_abs_t), 'stageA_lambda_abs_r': float(args.stageA_lambda_abs_r), 'stageA_abs_pose_robust': args.stageA_abs_pose_robust, 'stageA_abs_pose_scale_source': args.stageA_abs_pose_scale_source, 'stageA_abs_pose_fixed_scale': float(args.stageA_abs_pose_fixed_scale), 'stageA_log_grad_contrib': bool(args.stageA_log_grad_contrib), 'stageA_log_grad_interval': int(args.stageA_log_grad_interval), 'final_states': stageA_states,
        'residual_delta_summary_post_foldback': [
            {'sample_id': int(v['sample_id']), 'frame_id': int(v['frame_id']), 'rot_norm': float(torch.norm(v['vp'].cam_rot_delta).detach().cpu().item()), 'trans_norm': float(torch.norm(v['vp'].cam_trans_delta).detach().cpu().item()), 'exposure_a': float(v['vp'].exposure_a.detach().cpu().item()), 'exposure_b': float(v['vp'].exposure_b.detach().cpu().item())}
            for v in pseudo_views
        ],
        'true_pose_delta_summary': true_pose_delta_summary['per_view'],
        'true_pose_delta_aggregate': true_pose_delta_summary['aggregate'],
    }
    save_json(output_dir / 'stageA_history.json', stageA_history)

    stageB_history = None
    final_states = stageA_states
    if args.stage_mode == 'stageB' and args.stageB_iters > 0:
        v1 = load_v1_module()
        real_views, resolved_train_rgb_dir = [], None
        if args.train_manifest:
            real_views, resolved_train_rgb_dir = v1.load_real_viewpoints(args.train_manifest, args.train_rgb_dir)
            print(f"  [StageB] loaded real sparse-train views: {len(real_views)} from {args.train_manifest}")
        has_real_branch = bool(real_views and args.num_real_views > 0 and args.lambda_real > 0)
        if args.lambda_real > 0 and args.train_manifest and not real_views:
            raise RuntimeError('--train_manifest was given but no real viewpoints were loaded')

        gaussian_groups = build_micro_gaussian_param_groups(
            gaussians=gaussians,
            mode=args.stageA5_trainable_params,
            lr_xyz=args.stageA5_lr_xyz,
            lr_opacity=args.stageA5_lr_opacity,
        )
        stageB_gaussian_optimizer = torch.optim.Adam(gaussian_groups)
        stageB_pseudo_optimizer = build_stageA_optimizer(pseudo_views, cfg)

        stageB_history = {
            'iters': int(args.stageB_iters),
            'has_real_branch': bool(has_real_branch),
            'resolved_train_rgb_dir': resolved_train_rgb_dir,
            'iterations': [],
            'loss_total': [],
            'loss_real': [],
            'loss_pseudo': [],
            'loss_rgb': [],
            'loss_depth': [],
            'loss_pose_reg': [],
            'loss_exp_reg': [],
            'grad_norm_xyz': [],
            'sampled_real_frame_ids': [],
            'sampled_pseudo_sample_ids': [],
        }
        cfg_real = {'Training': {'lambda_pseudo': 1.0, 'rgb_boundary_threshold': 0.01, 'monocular': False}}

        for it in tqdm(range(1, int(args.stageB_iters) + 1), desc='StageB conservative joint'):
            stageB_pseudo_optimizer.zero_grad(set_to_none=True)
            stageB_gaussian_optimizer.zero_grad(set_to_none=True)

            real_loss_tensor = torch.zeros((), device='cuda')
            sampled_real_ids = []
            if has_real_branch:
                real_indices = sample_indices(len(real_views), args.num_real_views, rng)
                real_losses = []
                for ridx in real_indices:
                    rv = real_views[ridx]
                    sampled_real_ids.append(int(rv.get('frame_id', ridx)))
                    pkg = render(rv['vp'], gaussians, pipeline_params, bg)
                    real_losses.append(v1.get_loss_mapping_rgb(cfg_real, pkg['render'], rv['vp']))
                if real_losses:
                    real_loss_tensor = torch.stack(real_losses).mean()

            pseudo_indices = sample_indices(len(pseudo_views), cfg.num_pseudo_views, rng)
            if not pseudo_indices:
                raise RuntimeError('No pseudo viewpoints sampled in Stage B')

            pseudo_losses = []
            per_view_stats = []
            sampled_pseudo_ids = []
            for pidx in pseudo_indices:
                view = pseudo_views[pidx]
                sampled_pseudo_ids.append(int(view['sample_id']))
                pkg = render(view['vp'], gaussians, pipeline_params, bg)
                if args.stageA_depth_loss_mode == 'source_aware' and view.get('target_depth_source_map') is not None:
                    loss, stats = build_stageA_loss_source_aware(
                        render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], depth_source_map=view['target_depth_source_map'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, lambda_depth_seed=args.stageA_lambda_depth_seed, lambda_depth_dense=args.stageA_lambda_depth_dense, lambda_depth_fallback=args.stageA_lambda_depth_fallback, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0),
                    )
                else:
                    loss, stats = build_stageA_loss(
                        render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0),
                    )
                pseudo_losses.append(loss)
                per_view_stats.append(stats)

            pseudo_loss_tensor = torch.stack(pseudo_losses).mean()
            total_loss = args.lambda_real * real_loss_tensor + args.lambda_pseudo * pseudo_loss_tensor
            total_loss.backward()
            stageB_pseudo_optimizer.step()
            stageB_gaussian_optimizer.step()

            if args.stageA_apply_pose_update:
                seen = set()
                for pidx in pseudo_indices:
                    sid = int(pseudo_views[pidx]['sample_id'])
                    if sid in seen:
                        continue
                    apply_pose_residual_(pseudo_views[pidx]['vp'])
                    seen.add(sid)

            avg_stats = {key: float(np.mean([s[key] for s in per_view_stats])) for key in ['loss_rgb', 'loss_depth', 'loss_pose_reg', 'loss_exp_reg']}
            stageB_history['iterations'].append(int(it))
            stageB_history['loss_total'].append(float(total_loss.detach().item()))
            stageB_history['loss_real'].append(float(real_loss_tensor.detach().item()))
            stageB_history['loss_pseudo'].append(float(pseudo_loss_tensor.detach().item()))
            stageB_history['loss_rgb'].append(avg_stats['loss_rgb'])
            stageB_history['loss_depth'].append(avg_stats['loss_depth'])
            stageB_history['loss_pose_reg'].append(avg_stats['loss_pose_reg'])
            stageB_history['loss_exp_reg'].append(avg_stats['loss_exp_reg'])
            stageB_history['grad_norm_xyz'].append(float(torch.norm(gaussians._xyz.grad).detach().cpu().item()) if gaussians._xyz.grad is not None else 0.0)
            stageB_history['sampled_real_frame_ids'].append(sampled_real_ids)
            stageB_history['sampled_pseudo_sample_ids'].append(sampled_pseudo_ids)

            if it == 1 or it % 20 == 0:
                print(f"  [StageB] iter {it}: total={stageB_history['loss_total'][-1]:.4f}, real={stageB_history['loss_real'][-1]:.4f}, pseudo={stageB_history['loss_pseudo'][-1]:.4f}, rgb={stageB_history['loss_rgb'][-1]:.4f}, depth={stageB_history['loss_depth'][-1]:.4f}")

        save_json(output_dir / 'stageB_history.json', stageB_history)
        final_states = [export_view_state(v) for v in pseudo_views]

    save_json(output_dir / 'pseudo_camera_states_final.json', final_states)
    final_true_pose_delta_summary = summarize_true_pose_deltas(init_states, final_states)
    if stageB_history is None:
        stageA_history['final_true_pose_delta_summary'] = final_true_pose_delta_summary['per_view']
        stageA_history['final_true_pose_delta_aggregate'] = final_true_pose_delta_summary['aggregate']
        save_json(output_dir / 'stageA_history.json', stageA_history)
    else:
        stageB_history['final_true_pose_delta_summary'] = final_true_pose_delta_summary['per_view']
        stageB_history['final_true_pose_delta_aggregate'] = final_true_pose_delta_summary['aggregate']
        save_json(output_dir / 'stageB_history.json', stageB_history)

    if stageB_history is None:
        save_json(output_dir / 'refinement_history.json', stageA_history)
    else:
        save_json(output_dir / 'refinement_history.json', {'stageA': stageA_history, 'stageB': stageB_history})

    refined_ply_path = output_dir / 'refined_gaussians.ply'
    gaussians.save_ply(str(refined_ply_path))

    print(f'[Part3 StageA] Saved: {output_dir}')
    print(f'  Init states: {output_dir / "pseudo_camera_states_init.json"}')
    print(f'  StageA states: {output_dir / "pseudo_camera_states_stageA.json"}')
    print(f'  Final states: {output_dir / "pseudo_camera_states_final.json"}')
    agg = final_true_pose_delta_summary['aggregate']
    print(f"  True pose delta summary: mean_trans={agg['mean_trans_norm']:.6f}, max_trans={agg['max_trans_norm']:.6f}, mean_rotF={agg['mean_rot_fro_norm']:.6f}, max_rotF={agg['max_rot_fro_norm']:.6f}")
    print(f'  StageA history: {output_dir / "stageA_history.json"}')
    if stageB_history is not None:
        print(f'  StageB history: {output_dir / "stageB_history.json"}')


if __name__ == '__main__':
    main()

