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

from pseudo_branch.pseudo_camera_state import make_viewpoint_trainable, export_view_state
from pseudo_branch.pseudo_loss_v2 import build_stageA_loss, build_stageA_loss_source_aware
from pseudo_branch.pseudo_refine_scheduler import StageAConfig, build_stageA_optimizer


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
    p.add_argument('--stage_mode', choices=['stageA'], default='stageA')
    p.add_argument('--stageA_iters', type=int, default=300)
    p.add_argument('--stageA_beta_rgb', type=float, default=0.7)
    p.add_argument('--stageA_lambda_pose', type=float, default=0.01)
    p.add_argument('--stageA_lambda_exp', type=float, default=0.001)
    p.add_argument('--stageA_trans_reg_weight', type=float, default=1.0)
    p.add_argument('--stageA_lr_rot', type=float, default=0.003)
    p.add_argument('--stageA_lr_trans', type=float, default=0.001)
    p.add_argument('--stageA_lr_exp', type=float, default=0.01)
    p.add_argument('--stageA_disable_depth', action='store_true')
    p.add_argument(
        '--stageA_mask_mode',
        choices=['auto', 'train_mask', 'seed_support_only', 'legacy'],
        default='train_mask',
        help='Which upstream mask layer Stage A should actually consume.',
    )
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


def resolve_stageA_mask(sample_dir: Path, target_side: str, requested_mode: str, default_conf: np.ndarray, default_path: str | None, default_kind: str | None):
    if requested_mode == 'auto':
        if default_conf is not None:
            return default_conf.astype(np.float32), default_path, default_kind, 'auto'
        requested_mode = 'train_mask'

    if requested_mode == 'legacy':
        if default_conf is None:
            raise FileNotFoundError(f'No legacy/default confidence mask available under {sample_dir}')
        return default_conf.astype(np.float32), default_path, default_kind or 'legacy_default', 'legacy'

    if requested_mode == 'train_mask':
        for name in _mask_train_candidates(target_side):
            path = sample_dir / name
            if path.exists():
                return np.load(path).astype(np.float32), str(path), f'sample_explicit_{name}', 'train_mask'
        raise FileNotFoundError(f'No train-mask candidate found under {sample_dir} for target_side={target_side}')

    if requested_mode == 'seed_support_only':
        conf, kind = _compose_seed_support_mask(sample_dir, target_side)
        return conf, kind, kind, 'seed_support_only'

    raise ValueError(f'Unsupported stageA_mask_mode={requested_mode}')


def _summarize_source_map(source_map: np.ndarray | None):
    if source_map is None:
        return {
            'verified_ratio': None,
            'render_fallback_ratio': None,
            'seed_ratio': None,
            'dense_ratio': None,
        }
    source_map = np.asarray(source_map)
    total = float(source_map.size)
    return {
        'verified_ratio': float((source_map != 0).sum() / total),
        'render_fallback_ratio': float((source_map == 0).sum() / total),
        'seed_ratio': float(np.isin(source_map, [1, 2, 3]).sum() / total),
        'dense_ratio': float((source_map == 4).sum() / total),
    }


def load_stageA_pseudo_views(args):
    v1 = load_v1_module()
    pseudo_views, pseudo_manifest_info = v1.load_pseudo_viewpoints(
        args.pseudo_cache,
        args.target_side,
        args.confidence_mask_source,
        args.brpo_mask_root,
    )
    for view in pseudo_views:
        sample_dir = Path(view['target_rgb_path']).parent
        source_meta_path = sample_dir / 'source_meta.json'
        source_meta = json.load(open(source_meta_path)) if source_meta_path.exists() else {}

        conf_arr, conf_path, conf_kind, conf_mode = resolve_stageA_mask(
            sample_dir=sample_dir,
            target_side=args.target_side,
            requested_mode=args.stageA_mask_mode,
            default_conf=view.get('conf'),
            default_path=view.get('confidence_path'),
            default_kind=view.get('confidence_source_kind'),
        )
        positive = conf_arr[conf_arr > 0]
        view['conf'] = conf_arr.astype(np.float32)
        view['confidence_path'] = conf_path
        view['confidence_source_kind'] = conf_kind
        view['stageA_mask_mode_effective'] = conf_mode
        view['confidence_nonzero_ratio'] = float((conf_arr > 0).sum() / conf_arr.size)
        view['confidence_mean_positive'] = float(positive.mean()) if positive.size > 0 else 0.0

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
        view['target_depth_nonzero_ratio'] = float((depth_arr > 1e-6).sum() / depth_arr.size)
        view['target_depth_verified_ratio'] = source_summary['verified_ratio'] if source_summary['verified_ratio'] is not None else depth_meta.get('verified_ratio')
        view['target_depth_render_fallback_ratio'] = source_summary['render_fallback_ratio'] if source_summary['render_fallback_ratio'] is not None else depth_meta.get('render_fallback_ratio')
        view['target_depth_seed_ratio'] = source_summary['seed_ratio']
        view['target_depth_dense_ratio'] = source_summary['dense_ratio']
        make_viewpoint_trainable(view['vp'])
    return pseudo_views, pseudo_manifest_info


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

    pipeline_params = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    bg = torch.zeros(3, device='cuda')

    print(f'[Part3 StageA] Loading PLY: {args.ply_path}')
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply_path)
    print(f'  Loaded {len(gaussians.get_xyz)} Gaussians (sh_degree={args.sh_degree})')

    pseudo_views, pseudo_manifest_info = load_stageA_pseudo_views(args)
    if not pseudo_views:
        raise RuntimeError(f'No pseudo viewpoints found under {args.pseudo_cache}')
    print(f'  Loaded {len(pseudo_views)} pseudo viewpoints (target_side={args.target_side}, confidence_mask_source={args.confidence_mask_source})')
    print(
        '  StageA effective sources: '
        f"mask_mode={args.stageA_mask_mode}, depth_mode={args.stageA_target_depth_mode}, depth_loss_mode={args.stageA_depth_loss_mode}, "
        f"mean_mask_cov={np.mean([v['confidence_nonzero_ratio'] for v in pseudo_views]):.4f}, "
        f"mean_depth_verified={np.mean([v.get('target_depth_verified_ratio') or 0.0 for v in pseudo_views]):.4f}, "
        f"mean_depth_dense={np.mean([v.get('target_depth_dense_ratio') or 0.0 for v in pseudo_views]):.4f}"
    )

    init_states = [export_view_state(v) for v in pseudo_views]
    save_json(output_dir / 'pseudo_camera_states_init.json', init_states)

    cfg = StageAConfig(
        num_iterations=args.stageA_iters,
        beta_rgb=args.stageA_beta_rgb,
        lambda_pose=args.stageA_lambda_pose,
        lambda_exp=args.stageA_lambda_exp,
        trans_reg_weight=args.stageA_trans_reg_weight,
        lr_rot=args.stageA_lr_rot,
        lr_trans=args.stageA_lr_trans,
        lr_exp=args.stageA_lr_exp,
        num_pseudo_views=args.num_pseudo_views,
    )
    optimizer = build_stageA_optimizer(pseudo_views, cfg)

    history = {
        'iterations': [],
        'loss_total': [],
        'loss_rgb': [],
        'loss_depth': [],
        'loss_depth_seed': [],
        'loss_depth_dense': [],
        'loss_depth_fallback': [],
        'loss_pose_reg': [],
        'loss_exp_reg': [],
        'sampled_sample_ids': [],
    }

    for it in tqdm(range(1, cfg.num_iterations + 1), desc='StageA pseudo pose+exposure'):
        optimizer.zero_grad(set_to_none=True)
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
                )
            else:
                loss, stats = build_stageA_loss(
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
                )
            per_view_losses.append(loss)
            per_view_stats.append(stats)

        total_loss = torch.stack(per_view_losses).mean()
        total_loss.backward()
        optimizer.step()

        avg_stats = {}
        for key in ['loss_total', 'loss_rgb', 'loss_depth', 'loss_depth_seed', 'loss_depth_dense', 'loss_depth_fallback', 'loss_pose_reg', 'loss_exp_reg']:
            avg_stats[key] = float(np.mean([s[key] for s in per_view_stats]))

        for key in avg_stats:
            history[key].append(avg_stats[key])
        history['iterations'].append(it)
        history['sampled_sample_ids'].append(sampled_ids)

        if it == 1 or it % 20 == 0:
            print(
                f"  iter {it}: total={avg_stats['loss_total']:.4f}, rgb={avg_stats['loss_rgb']:.4f}, depth={avg_stats['loss_depth']:.4f}, "
                f"depth_seed={avg_stats['loss_depth_seed']:.4f}, depth_dense={avg_stats['loss_depth_dense']:.4f}, depth_fb={avg_stats['loss_depth_fallback']:.4f}, "
                f"pose_reg={avg_stats['loss_pose_reg']:.4f}, exp_reg={avg_stats['loss_exp_reg']:.4f}"
            )

    stageA_states = [export_view_state(v) for v in pseudo_views]
    save_json(output_dir / 'pseudo_camera_states_stageA.json', stageA_states)
    save_json(output_dir / 'pseudo_camera_states_final.json', stageA_states)

    stageA_history = {
        'args': vars(args),
        'stage_mode': args.stage_mode,
        'pseudo_manifest_info': pseudo_manifest_info,
        'num_pseudo_viewpoints_loaded': len(pseudo_views),
        'effective_source_summary': {
            'stageA_mask_mode_requested': args.stageA_mask_mode,
            'stageA_target_depth_mode_requested': args.stageA_target_depth_mode,
            'stageA_depth_loss_mode': args.stageA_depth_loss_mode,
            'stageA_disable_depth': bool(args.stageA_disable_depth),
            'mean_confidence_nonzero_ratio': float(np.mean([v.get('confidence_nonzero_ratio', 0.0) for v in pseudo_views])),
            'mean_target_depth_verified_ratio': float(np.mean([(v.get('target_depth_verified_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_render_fallback_ratio': float(np.mean([(v.get('target_depth_render_fallback_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_seed_ratio': float(np.mean([(v.get('target_depth_seed_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_dense_ratio': float(np.mean([(v.get('target_depth_dense_ratio') or 0.0) for v in pseudo_views])),
        },
        'pseudo_sample_meta': [
            {
                'sample_id': int(v['sample_id']),
                'frame_id': int(v['frame_id']),
                'target_rgb_path': v.get('target_rgb_path'),
                'target_depth_for_refine_path': v.get('target_depth_for_refine_path'),
                'target_depth_for_refine_kind': v.get('target_depth_for_refine_kind'),
                'stageA_target_depth_mode_effective': v.get('stageA_target_depth_mode_effective'),
                'target_depth_nonzero_ratio': v.get('target_depth_nonzero_ratio'),
                'target_depth_verified_ratio': v.get('target_depth_verified_ratio'),
                'target_depth_render_fallback_ratio': v.get('target_depth_render_fallback_ratio'),
                'target_depth_seed_ratio': v.get('target_depth_seed_ratio'),
                'target_depth_dense_ratio': v.get('target_depth_dense_ratio'),
                'target_depth_for_refine_source': v.get('source_meta', {}).get('target_depth_for_refine_source'),
                'target_depth_for_refine_source_map_path': v.get('target_depth_for_refine_source_map_path'),
                'confidence_path': v.get('confidence_path'),
                'confidence_source_kind': v.get('confidence_source_kind'),
                'stageA_mask_mode_effective': v.get('stageA_mask_mode_effective'),
                'confidence_nonzero_ratio': v.get('confidence_nonzero_ratio'),
                'confidence_mean_positive': v.get('confidence_mean_positive'),
            }
            for v in pseudo_views
        ],
        'history': history,
        'stageA_disable_depth': bool(args.stageA_disable_depth),
        'stageA_mask_mode': args.stageA_mask_mode,
        'stageA_target_depth_mode': args.stageA_target_depth_mode,
        'stageA_depth_loss_mode': args.stageA_depth_loss_mode,
        'stageA_lambda_depth_seed': float(args.stageA_lambda_depth_seed),
        'stageA_lambda_depth_dense': float(args.stageA_lambda_depth_dense),
        'stageA_lambda_depth_fallback': float(args.stageA_lambda_depth_fallback),
        'final_states': stageA_states,
        'final_delta_summary': [
            {
                'sample_id': int(v['sample_id']),
                'frame_id': int(v['frame_id']),
                'rot_norm': float(torch.norm(v['vp'].cam_rot_delta).detach().cpu().item()),
                'trans_norm': float(torch.norm(v['vp'].cam_trans_delta).detach().cpu().item()),
                'exposure_a': float(v['vp'].exposure_a.detach().cpu().item()),
                'exposure_b': float(v['vp'].exposure_b.detach().cpu().item()),
            }
            for v in pseudo_views
        ],
    }
    save_json(output_dir / 'stageA_history.json', stageA_history)
    save_json(output_dir / 'refinement_history.json', stageA_history)

    print(f'[Part3 StageA] Saved: {output_dir}')
    print(f'  Init states: {output_dir / "pseudo_camera_states_init.json"}')
    print(f'  Final states: {output_dir / "pseudo_camera_states_stageA.json"}')
    print(f'  History: {output_dir / "stageA_history.json"}')


if __name__ == '__main__':
    main()
