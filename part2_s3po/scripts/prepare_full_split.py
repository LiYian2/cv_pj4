#!/usr/bin/env python3
"""
Prepare 'full' split for S3PO datasets (Re10k-1, DL3DV-2, 405841).
This script adds a 'full' split containing all frames to each dataset.
"""

from pathlib import Path
import json
import shutil
from datetime import datetime
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path('/home/bzhang512/CV_Project')
S3PO_ROOT = PROJECT_ROOT / 'third_party' / 'S3PO-GS'
CONFIG_DIR = PROJECT_ROOT / 'part2_s3po/configs'


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def list_pngs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.png'])


def intrinsics_norm_to_px(norm_intrinsics: dict, width: int, height: int):
    return {
        'fx': float(norm_intrinsics['fx']) * float(width),
        'fy': float(norm_intrinsics['fy']) * float(height),
        'cx': float(norm_intrinsics['cx']) * float(width),
        'cy': float(norm_intrinsics['cy']) * float(height),
    }


def symlink_files(src_dir: Path, dst_dir: Path, names):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f'Missing source file: {src}')
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def prepare_dl3dv_full():
    """Prepare full split for Re10k-1 and DL3DV-2 (dl3dv loader format)."""
    dl3dv_specs = [
        {
            'dataset_name': 'Re10k-1',
            'scene_key': 're10k1',
            'source_root': PROJECT_ROOT / 'dataset/Re10k-1/part1/shared',
            'target_root': PROJECT_ROOT / 'dataset/Re10k-1/part2_s3po',
        },
        {
            'dataset_name': 'DL3DV-2',
            'scene_key': 'dl3dv2',
            'source_root': PROJECT_ROOT / 'dataset/DL3DV-2/part1/shared',
            'target_root': PROJECT_ROOT / 'dataset/DL3DV-2/part2_s3po',
        },
    ]

    results = []
    
    for spec in dl3dv_specs:
        source_root = spec['source_root']
        target_root = spec['target_root']
        
        src_images = source_root / 'images_512'
        src_cameras = source_root / 'meta/cameras.json'
        src_intr = source_root / 'meta/intrinsics.json'
        
        if not src_images.exists():
            print(f"Skip {spec['dataset_name']}: missing images")
            continue
        if not src_cameras.exists():
            print(f"Skip {spec['dataset_name']}: missing cameras.json")
            continue
        if not src_intr.exists():
            print(f"Skip {spec['dataset_name']}: missing intrinsics.json")
            continue
        
        image_paths = list_pngs(src_images)
        image_names = [p.name for p in image_paths]
        n_total = len(image_names)
        
        if n_total == 0:
            print(f"Skip {spec['dataset_name']}: no images")
            continue
        
        cams = read_json(src_cameras)
        cam_by_name = {c['image_name']: c for c in cams}
        
        # Full split: all indices
        full_ids = list(range(n_total))
        
        # Create full split directory
        split_root = target_root / 'full'
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        selected_names = [image_names[i] for i in full_ids]
        selected_cams = []
        for local_id, name in enumerate(selected_names):
            c = dict(cam_by_name[name])
            c['cam_id'] = local_id
            selected_cams.append(c)
        
        # Symlink RGB files
        symlink_files(src_images, split_root / 'rgb', selected_names)
        
        # Write cameras.json
        write_json(split_root / 'cameras.json', selected_cams)
        
        # Symlink intrinsics_norm.json
        intr_norm_link = split_root / 'intrinsics_norm.json'
        if intr_norm_link.exists() or intr_norm_link.is_symlink():
            intr_norm_link.unlink()
        intr_norm_link.symlink_to(src_intr)
        
        # Write intrinsics_px.json
        width, height = Image.open(image_paths[0]).size
        intr_norm = read_json(src_intr)
        intr_px = intrinsics_norm_to_px(intr_norm, width, height)
        write_json(split_root / 'intrinsics_px.json', intr_px)
        
        # Write split_manifest.json
        manifest = {
            'dataset_name': spec['dataset_name'],
            'split': 'full',
            'loader_type': 'dl3dv',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'n_total': n_total,
            'n_split': n_total,
            'selected_indices': full_ids,
            'width': width,
            'height': height,
            'files': {
                'rgb': 'rgb',
                'cameras': 'cameras.json',
                'intrinsics_norm': 'intrinsics_norm.json',
                'intrinsics_px': 'intrinsics_px.json',
            },
        }
        write_json(split_root / 'split_manifest.json', manifest)
        
        # Write config file
        cfg = yaml.safe_load((S3PO_ROOT / 'configs/mono/dl3dv/base_config.yaml').read_text(encoding='utf-8'))
        cfg['Dataset']['type'] = 'dl3dv'
        cfg['Dataset']['sensor_type'] = 'monocular'
        cfg['Dataset']['dataset_path'] = str(split_root)
        cfg['Dataset']['begin'] = 0
        cfg['Dataset']['end'] = n_total
        cfg['Dataset']['Calibration'] = {
            'fx': float(intr_px['fx']),
            'fy': float(intr_px['fy']),
            'cx': float(intr_px['cx']),
            'cy': float(intr_px['cy']),
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0,
            'width': int(width),
            'height': int(height),
            'depth_scale': 200.0,
            'distorted': False,
        }
        
        cfg_name = f"s3po_{spec['scene_key']}_full.yaml"
        cfg_path = CONFIG_DIR / cfg_name
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        
        results.append({
            'dataset_name': spec['dataset_name'],
            'split': 'full',
            'n_total': n_total,
            'n_split': n_total,
            'target_root': str(split_root),
            'config_path': str(cfg_path),
        })
        
        print(f"{spec['dataset_name']} full: {n_total} frames")
    
    return results


def prepare_waymo_full():
    """Prepare full split for 405841 (waymo loader format)."""
    WAYMO_RGB_SRC = PROJECT_ROOT / 'dataset/405841/part1/shared/images_512'
    WAYMO_DEPTH_SRC = PROJECT_ROOT / 'dataset/405841/FRONT/depth'
    WAYMO_GT_SRC = PROJECT_ROOT / 'dataset/405841/FRONT/gt'
    WAYMO_DST = PROJECT_ROOT / 'dataset/405841/part2_s3po'
    
    if not WAYMO_RGB_SRC.exists():
        print("Skip 405841: missing RGB source")
        return None
    
    rgb_names = sorted([p.name for p in WAYMO_RGB_SRC.glob('*.png')])
    depth_names = sorted([p.name for p in WAYMO_DEPTH_SRC.glob('*.png')])
    gt_names = sorted([p.name for p in WAYMO_GT_SRC.glob('*.txt')])
    
    rgb_stems = {Path(x).stem for x in rgb_names}
    depth_stems = {Path(x).stem for x in depth_names}
    gt_stems = {Path(x).stem for x in gt_names}
    common_stems = sorted(list(rgb_stems & depth_stems & gt_stems))
    
    if len(common_stems) == 0:
        print("Skip 405841: no common stems")
        return None
    
    n_total = len(common_stems)
    
    # Load calibration
    waymo_cfg_file = S3PO_ROOT / 'configs/mono/waymo/405841.yaml'
    waymo_cfg = yaml.safe_load(waymo_cfg_file.read_text(encoding='utf-8'))
    waymo_calib_raw = dict(waymo_cfg['Dataset']['Calibration'])
    
    rgb_sample = Image.open(WAYMO_RGB_SRC / rgb_names[0])
    target_width, target_height = rgb_sample.size
    rgb_sample.close()
    
    src_width = float(waymo_calib_raw['width'])
    src_height = float(waymo_calib_raw['height'])
    sx = float(target_width) / src_width
    sy = float(target_height) / src_height
    
    waymo_calib_sync = {
        'fx': float(waymo_calib_raw['fx']) * sx,
        'fy': float(waymo_calib_raw['fy']) * sy,
        'cx': float(waymo_calib_raw['cx']) * sx,
        'cy': float(waymo_calib_raw['cy']) * sy,
        'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0,
        'width': int(target_width),
        'height': int(target_height),
        'depth_scale': float(waymo_calib_raw.get('depth_scale', 5000.0)),
        'distorted': False,
    }
    
    # Create full split directory
    split_root = WAYMO_DST / 'full'
    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    
    for sub in ['rgb', 'depth', 'mono_depth', 'gt']:
        (split_root / sub).mkdir(parents=True, exist_ok=True)
    
    # Process all frames
    for stem in common_stems:
        rgb_name = f'{stem}.png'
        gt_name = f'{stem}.txt'
        
        # RGB symlink
        rgb_dst = split_root / 'rgb' / rgb_name
        if rgb_dst.exists() or rgb_dst.is_symlink():
            rgb_dst.unlink()
        rgb_dst.symlink_to(WAYMO_RGB_SRC / rgb_name)
        
        # Resize depth
        depth_src = WAYMO_DEPTH_SRC / rgb_name
        depth_dst = split_root / 'depth' / rgb_name
        with Image.open(depth_src) as dep_img:
            dep_512 = dep_img.resize((target_width, target_height), Image.NEAREST)
            dep_512.save(depth_dst)
        
        # mono_depth symlink
        mono_dst = split_root / 'mono_depth' / rgb_name
        if mono_dst.exists() or mono_dst.is_symlink():
            mono_dst.unlink()
        mono_dst.symlink_to(depth_dst)
        
        # gt symlink
        gt_dst = split_root / 'gt' / gt_name
        if gt_dst.exists() or gt_dst.is_symlink():
            gt_dst.unlink()
        gt_dst.symlink_to(WAYMO_GT_SRC / gt_name)
    
    # Write split_manifest.json
    manifest = {
        'dataset_name': '405841',
        'split': 'full',
        'loader_type': 'waymo',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'n_total': n_total,
        'n_split': n_total,
        'selected_indices': list(range(n_total)),
        'selected_stems': common_stems,
        'rgb_source': str(WAYMO_RGB_SRC),
        'depth_source': str(WAYMO_DEPTH_SRC),
        'gt_source': str(WAYMO_GT_SRC),
        'depth_resize': {
            'source_size': [int(src_width), int(src_height)],
            'target_size': [int(target_width), int(target_height)],
            'resample': 'NEAREST',
        },
        'calibration_sync': waymo_calib_sync,
        'mono_depth_placeholder': 'symlink_to_resized_depth',
        'files': {
            'rgb': 'rgb',
            'depth': 'depth',
            'mono_depth': 'mono_depth',
            'gt': 'gt',
        },
    }
    write_json(split_root / 'split_manifest.json', manifest)
    
    # Write config file
    cfg = yaml.safe_load((S3PO_ROOT / 'configs/mono/waymo/base_config.yaml').read_text(encoding='utf-8'))
    cfg['Dataset']['type'] = 'waymo'
    cfg['Dataset']['sensor_type'] = 'monocular'
    cfg['Dataset']['dataset_path'] = str(split_root)
    cfg['Dataset']['Calibration'] = waymo_calib_sync
    
    cfg_name = 's3po_405841_full_waymo.yaml'
    cfg_path = CONFIG_DIR / cfg_name
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    
    print(f"405841 full: {n_total} frames")
    
    return {
        'dataset_name': '405841',
        'split': 'full',
        'n_total': n_total,
        'n_split': n_total,
        'target_root': str(split_root),
        'config_path': str(cfg_path),
    }


def validate_results(results):
    """Validate the created full splits."""
    for row in results:
        root = Path(row['target_root'])
        n_expected = row['n_split']
        
        if 'loader_type' not in row:
            row['loader_type'] = 'dl3dv' if row['dataset_name'] != '405841' else 'waymo'
        
        if row.get('loader_type') == 'dl3dv':
            rgb = list_pngs(root / 'rgb')
            cams = read_json(root / 'cameras.json')
            assert len(rgb) == n_expected, f"{root}: rgb count mismatch"
            assert len(cams) == n_expected, f"{root}: camera count mismatch"
            assert (root / 'intrinsics_norm.json').exists(), f"{root}: missing intrinsics_norm.json"
            assert (root / 'intrinsics_px.json').exists(), f"{root}: missing intrinsics_px.json"
            print(f"[OK] {row['dataset_name']} full: {len(rgb)} frames, cameras.json OK")
        else:
            rgb = list_pngs(root / 'rgb')
            dep = list_pngs(root / 'depth')
            mono = list_pngs(root / 'mono_depth')
            gt = sorted([p for p in (root / 'gt').iterdir() if p.is_file() and p.suffix.lower() == '.txt'])
            assert len(rgb) == n_expected, f"{root}: rgb count mismatch"
            assert len(dep) == n_expected, f"{root}: depth count mismatch"
            assert len(mono) == n_expected, f"{root}: mono_depth count mismatch"
            assert len(gt) == n_expected, f"{root}: gt count mismatch"
            print(f"[OK] {row['dataset_name']} full: {len(rgb)} frames, all directories OK")


if __name__ == '__main__':
    print("=== Preparing full splits for S3PO datasets ===")
    print()
    
    print('--- DL3DV format (Re10k-1, DL3DV-2) ---')
    dl3dv_results = prepare_dl3dv_full()
    
    print()
    print('--- Waymo format (405841) ---')
    waymo_result = prepare_waymo_full()
    
    all_results = dl3dv_results + ([waymo_result] if waymo_result else [])
    
    print()
    print('=== Validation ===')
    validate_results(all_results)
    
    print()
    print('=== Done ===')
