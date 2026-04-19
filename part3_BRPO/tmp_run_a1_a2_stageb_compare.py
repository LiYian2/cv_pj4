#!/usr/bin/env python3
import subprocess, json
from pathlib import Path
from datetime import datetime

PLY_PATH = '/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply'
PSEUDO_CACHE = '/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline'
INIT_STATES = '/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json'
TRAIN_MANIFEST = '/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json'
TRAIN_RGB_DIR = '/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb'
SIGNAL_V2_ROOT = '/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_re10k_signal_v2_from_e15'
OUTPUT_ROOT = Path('/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_a2_stageb_compare_e1')
PY = '/home/bzhang512/miniconda3/envs/s3po-gs/bin/python'
SCRIPT = '/home/bzhang512/CV_Project/part3_BRPO/scripts/run_pseudo_refinement_v2.py'

ARMS = [
    {'name': 'control', 'rgb_mode': 'brpo_v2_raw', 'depth_mode': 'train_mask', 'target_depth_mode': 'target_depth_for_refine_v2_brpo'},
    {'name': 'a1', 'rgb_mode': 'joint_confidence_v2', 'depth_mode': 'joint_confidence_v2', 'target_depth_mode': 'joint_depth_v2'},
    {'name': 'a1_a2', 'rgb_mode': 'joint_confidence_expand_v1', 'depth_mode': 'joint_confidence_expand_v1', 'target_depth_mode': 'joint_depth_expand_v1'},
]

def run_arm(arm_name, rgb_mode, depth_mode, target_depth_mode):
    output_dir = OUTPUT_ROOT / arm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, SCRIPT, '--ply_path', PLY_PATH, '--pseudo_cache', PSEUDO_CACHE, '--output_dir', str(output_dir),
           '--signal_pipeline', 'brpo_v2', '--signal_v2_root', SIGNAL_V2_ROOT, '--stage_mode', 'stageB',
           '--stageA_iters', '0', '--stageA_rgb_mask_mode', rgb_mode, '--stageA_depth_mask_mode', depth_mode,
           '--stageA_target_depth_mode', target_depth_mode, '--stageA_depth_loss_mode', 'source_aware',
           '--stageB_iters', '120', '--stageB_post_switch_iter', '40', '--stageB_post_lr_scale_xyz', '0.3',
           '--init_pseudo_camera_states_json', INIT_STATES, '--init_pseudo_reference_mode', 'keep',
           '--train_manifest', TRAIN_MANIFEST, '--train_rgb_dir', TRAIN_RGB_DIR,
           '--lambda_real', '1.0', '--lambda_pseudo', '1.0', '--num_real_views', '2', '--sh_degree', '0',
           '--pseudo_local_gating', 'spgm_keep', '--pseudo_local_gating_min_rgb_mask_ratio', '0.0192',
           '--pseudo_local_gating_spgm_policy_mode', 'dense_keep', '--pseudo_local_gating_spgm_ranking_mode', 'v1',
           '--pseudo_local_gating_spgm_cluster_keep_near', '1.0', '--pseudo_local_gating_spgm_cluster_keep_mid', '1.0',
           '--pseudo_local_gating_spgm_cluster_keep_far', '1.0']
    print(f'Running {arm_name}: {rgb_mode} / {depth_mode} / {target_depth_mode}')
    result = subprocess.run(cmd, cwd='/home/bzhang512/CV_Project/part3_BRPO', capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f'ERROR: {result.stderr[:500]}')
        return None
    return str(output_dir)

def parse_replay_eval(output_dir):
    replay_dir = Path(output_dir) / 'replay_eval'
    if not replay_dir.exists(): return None
    results_file = replay_dir / 'metrics_summary.json'
    if results_file.exists():
        with open(results_file) as f: return json.load(f)
    return None

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in ARMS:
        output_dir = run_arm(arm['name'], arm['rgb_mode'], arm['depth_mode'], arm['target_depth_mode'])
        if output_dir:
            metrics = parse_replay_eval(output_dir)
            results[arm['name']] = {'rgb_mode': arm['rgb_mode'], 'depth_mode': arm['depth_mode'],
                                   'target_depth_mode': arm['target_depth_mode'], 'output_dir': output_dir, 'metrics': metrics}
    print('SUMMARY:')
    for arm_name, data in results.items():
        if data['metrics']:
            m = data['metrics']
            print(f'{arm_name}: PSNR={m.get(avg_psnr,0):.6f}, SSIM={m.get(avg_ssim,0):.6f}, LPIPS={m.get(avg_lpips,0):.6f}')
    with open(OUTPUT_ROOT / 'compare_summary.json', 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'signal_v2_root': SIGNAL_V2_ROOT, 'arms': results}, f, indent=2)
    print(f'Saved to: {OUTPUT_ROOT / compare_summary.json}')

if __name__ == '__main__': main()
