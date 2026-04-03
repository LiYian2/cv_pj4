#!/usr/bin/env python3
"""
Prepare part3 stage1 Difix-ref pseudo-view assets for S3PO backend.

Input:
  - sparse split: selected_indices (train frames)
  - test split: selected_indices (test frames) + render_rgb + render_depth + pose
  - render from external eval

Output:
  - inputs/raw_render/, inputs/left_ref/, inputs/right_ref/
  - difix/left_fixed/, difix/right_fixed/
  - pseudo_cache/samples/{frame_id}/camera.json, render_rgb.png, render_depth.npy
  - augmented_train_left/rgb/, augmented_train_right/rgb/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PseudoRecord:
    frame_id: int
    placement: str
    gap_left_train_id: int
    gap_right_train_id: int
    test_idx: int
    render_src_path: str
    left_ref_src_path: str
    right_ref_src_path: str
    render_input_path: str
    left_ref_input_path: str
    right_ref_input_path: str
    left_fixed_path: str
    right_fixed_path: str
    pseudo_camera_path: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare part3 stage1 Difix assets for S3PO.")
    p.add_argument("--stage", choices=["select", "difix", "pack", "all"], default="all")
    p.add_argument("--scene-name", required=True, help="e.g., 405841, Re10k-1, DL3DV-2")
    p.add_argument("--run-key", required=True, help="subdir under dataset/<scene>/part3_stage1/")
    p.add_argument("--dataset-root", type=str, default="/home/bzhang512/CV_Project/dataset")
    p.add_argument("--sparse-manifest", required=True, help="path to sparse/split_manifest.json")
    p.add_argument("--test-manifest", required=True, help="path to test/split_manifest.json")
    p.add_argument("--sparse-rgb-dir", required=True, help="path to sparse/rgb/")
    p.add_argument("--render-rgb-dir", required=True, help="path to external eval render_rgb/")
    p.add_argument("--render-depth-dir", required=True, help="path to external eval render_depth_npy/")
    p.add_argument("--trj-json", required=True, help="path to trj_external_infer.json")
    p.add_argument("--placement", choices=["midpoint", "tertile", "both"], default="midpoint")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prompt", type=str, default="remove degradation")
    p.add_argument("--difix-model-name", type=str, default="nvidia/difix_ref")
    p.add_argument("--difix-model-path", type=str, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--timestep", type=int, default=199)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def symlink_force(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def copy_force(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def get_sparse_rgb_name(frame_id: int, scene_name: str) -> str:
    """Get sparse rgb filename from frame_id."""
    if scene_name == "DL3DV-2":
        return f"frame_{frame_id + 1:05d}.png"
    elif scene_name == "Re10k-1":
        return f"{frame_id:05d}.png"
    else:  # 405841 and others
        return f"{frame_id:06d}.png"


def get_render_name(test_idx: int) -> str:
    """Get render filename from test index."""
    return f"{test_idx:04d}_pred.png"


def get_render_depth_name(test_idx: int) -> str:
    """Get render depth filename from test index."""
    return f"{test_idx:04d}.npy"


def get_canonical_name(frame_id: int, scene_name: str) -> str:
    """Get canonical output filename (without _pred suffix)."""
    if scene_name == "DL3DV-2":
        return f"{frame_id:05d}.png"
    elif scene_name == "Re10k-1":
        return f"{frame_id:05d}.png"
    else:
        return f"{frame_id:06d}.png"


def pick_pseudo_in_gap(
    left_id: int,
    right_id: int,
    test_indices: List[int],
    test_idx_to_frame_id: Dict[int, int],
    placement: str,
) -> List[Tuple[int, str]]:
    """Select pseudo frames in a sparse gap."""
    out: List[Tuple[int, int, str]] = []  # (frame_id, test_idx, label)
    
    # Get test frames in this gap
    gap_test = [(fid, idx) for idx, fid in test_idx_to_frame_id.items() if left_id < fid < right_id]
    if not gap_test:
        return []
    
    candidates = sorted([fid for fid, _ in gap_test])
    
    def add(target: float, label: str) -> None:
        # Find nearest frame_id to target
        nearest = min(candidates, key=lambda x: abs(x - target))
        test_idx = next((idx for fid, idx in gap_test if fid == nearest), None)
        if test_idx is not None:
            out.append((nearest, test_idx, label))
    
    if placement in ("midpoint", "both"):
        target = left_id + (right_id - left_id) / 2.0
        add(target, "midpoint")
    if placement in ("tertile", "both"):
        add(left_id + (right_id - left_id) / 3.0, "tertile_left")
        add(left_id + 2.0 * (right_id - left_id) / 3.0, "tertile_right")
    
    # Deduplicate by frame_id
    seen: set = set()
    result: List[Tuple[int, int, str]] = []
    for fid, tidx, label in out:
        if fid not in seen:
            seen.add(fid)
            result.append((fid, tidx, label))
    return result


def build_source_manifest(args: argparse.Namespace, run_root: Path) -> dict:
    sparse_manifest = read_json(Path(args.sparse_manifest))
    test_manifest = read_json(Path(args.test_manifest))
    
    sparse_indices = sparse_manifest["selected_indices"]
    test_indices = test_manifest["selected_indices"]
    
    manifest = {
        "scene_name": args.scene_name,
        "run_key": args.run_key,
        "sparse_manifest": args.sparse_manifest,
        "test_manifest": args.test_manifest,
        "sparse_rgb_dir": args.sparse_rgb_dir,
        "render_rgb_dir": args.render_rgb_dir,
        "render_depth_dir": args.render_depth_dir,
        "trj_json": args.trj_json,
        "sparse_indices": sparse_indices,
        "test_indices": test_indices,
        "num_sparse": len(sparse_indices),
        "num_test": len(test_indices),
    }
    write_json(run_root / "manifests" / "source_manifest.json", manifest)
    return manifest


def build_pseudo_records(args: argparse.Namespace, run_root: Path, source: dict) -> List[PseudoRecord]:
    sparse_indices = source["sparse_indices"]
    test_indices = source["test_indices"]
    
    # Build mapping: test_idx -> frame_id and frame_id -> test_idx
    test_idx_to_frame_id = {i: fid for i, fid in enumerate(test_indices)}
    frame_id_to_test_idx = {fid: i for i, fid in enumerate(test_indices)}
    
    sparse_rgb_dir = Path(args.sparse_rgb_dir)
    render_rgb_dir = Path(args.render_rgb_dir)
    render_depth_dir = Path(args.render_depth_dir)
    
    # Load pose
    trj = read_json(Path(args.trj_json))
    trj_id = trj["trj_id"]
    trj_est = trj["trj_est"]  # c2w matrices
    
    records: List[PseudoRecord] = []
    
    for left_id, right_id in zip(sparse_indices[:-1], sparse_indices[1:]):
        chosen = pick_pseudo_in_gap(left_id, right_id, test_indices, test_idx_to_frame_id, args.placement)
        for frame_id, test_idx, label in chosen:
            # Get render paths
            render_name = get_render_name(test_idx)
            depth_name = get_render_depth_name(test_idx)
            render_src = render_rgb_dir / render_name
            depth_src = render_depth_dir / depth_name
            
            # Get left/right ref paths
            left_name = get_sparse_rgb_name(left_id, args.scene_name)
            right_name = get_sparse_rgb_name(right_id, args.scene_name)
            left_ref_src = sparse_rgb_dir / left_name
            right_ref_src = sparse_rgb_dir / right_name
            
            # Output paths
            canonical = get_canonical_name(frame_id, args.scene_name)
            
            # Get pose for this test_idx
            pose_idx = trj_id.index(test_idx) if test_idx in trj_id else -1
            pose_c2w = trj_est[pose_idx] if pose_idx >= 0 else None
            
            rec = PseudoRecord(
                frame_id=frame_id,
                placement=label,
                gap_left_train_id=left_id,
                gap_right_train_id=right_id,
                test_idx=test_idx,
                render_src_path=str(render_src.resolve()),
                left_ref_src_path=str(left_ref_src.resolve()),
                right_ref_src_path=str(right_ref_src.resolve()),
                render_input_path=str((run_root / "inputs" / "raw_render" / canonical).resolve()),
                left_ref_input_path=str((run_root / "inputs" / "left_ref" / canonical).resolve()),
                right_ref_input_path=str((run_root / "inputs" / "right_ref" / canonical).resolve()),
                left_fixed_path=str((run_root / "difix" / "left_fixed" / canonical).resolve()),
                right_fixed_path=str((run_root / "difix" / "right_fixed" / canonical).resolve()),
                pseudo_camera_path=str((run_root / "pseudo_cache" / "samples" / str(frame_id) / "camera.json").resolve()),
            )
            records.append(rec)
    
    records = sorted(records, key=lambda r: r.frame_id)
    if args.limit is not None:
        records = records[:args.limit]
    return records


def stage_select(args: argparse.Namespace, run_root: Path) -> List[PseudoRecord]:
    source = build_source_manifest(args, run_root)
    records = build_pseudo_records(args, run_root, source)
    
    # Create symlinks
    raw_dir = run_root / "inputs" / "raw_render"
    left_dir = run_root / "inputs" / "left_ref"
    right_dir = run_root / "inputs" / "right_ref"
    ensure_dir(raw_dir)
    ensure_dir(left_dir)
    ensure_dir(right_dir)
    
    for rec in records:
        if not args.dry_run:
            symlink_force(Path(rec.render_src_path), Path(rec.render_input_path))
            symlink_force(Path(rec.left_ref_src_path), Path(rec.left_ref_input_path))
            symlink_force(Path(rec.right_ref_src_path), Path(rec.right_ref_input_path))
    
    # Save manifests
    write_json(run_root / "manifests" / "pseudo_selection_manifest.json", [asdict(r) for r in records])
    
    summary = {
        "scene_name": args.scene_name,
        "run_key": args.run_key,
        "placement": args.placement,
        "num_sparse": source["num_sparse"],
        "num_test": source["num_test"],
        "num_selected": len(records),
        "selected_frame_ids": [r.frame_id for r in records],
    }
    write_json(run_root / "manifests" / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return records


def load_pseudo_records(run_root: Path) -> List[PseudoRecord]:
    data = read_json(run_root / "manifests" / "pseudo_selection_manifest.json")
    return [PseudoRecord(**item) for item in data]


def load_difix_model(model_name: Optional[str], model_path: Optional[str], timestep: int):
    if model_name and not model_path:
        from diffusers import DiffusionPipeline
        custom_pipeline = "/home/bzhang512/CV_Project/third_party/Difix3D/src/pipeline_difix.py"
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            custom_pipeline=custom_pipeline,
            trust_remote_code=True,
        )
        pipe = pipe.to("cuda")
        return {"kind": "hf_pipeline", "obj": pipe, "timestep": timestep}
    
    difix_src = Path("/home/bzhang512/CV_Project/third_party/Difix3D/src")
    if str(difix_src) not in sys.path:
        sys.path.insert(0, str(difix_src))
    from model import Difix  # type: ignore
    
    model = Difix(
        pretrained_name=model_name,
        pretrained_path=model_path,
        timestep=timestep,
        mv_unet=True,
    )
    model.set_eval()
    return {"kind": "local_model", "obj": model, "timestep": timestep}


def run_single_difix(model_bundle, image_path: Path, ref_path: Path, output_path: Path, prompt: str, height: int, width: int, overwrite: bool) -> None:
    from PIL import Image
    
    if output_path.exists() and not overwrite:
        return
    image = Image.open(image_path).convert("RGB")
    ref = Image.open(ref_path).convert("RGB")
    ensure_dir(output_path.parent)
    
    if model_bundle["kind"] == "hf_pipeline":
        pipe = model_bundle["obj"]
        out = pipe(
            prompt,
            image=image,
            ref_image=ref,
            height=height,
            width=width,
            num_inference_steps=1,
            timesteps=[model_bundle["timestep"]],
            guidance_scale=0.0,
        ).images[0]
    else:
        model = model_bundle["obj"]
        out = model.sample(image=image, ref_image=ref, prompt=prompt, height=height, width=width)
    
    if out.size != image.size:
        out = out.resize(image.size, Image.LANCZOS)
    out.save(output_path)


def stage_difix(args: argparse.Namespace, run_root: Path) -> List[PseudoRecord]:
    records = load_pseudo_records(run_root)
    if args.dry_run:
        print(f"[dry-run] would run difix on {len(records)} pseudo frames")
        return records
    
    model = load_difix_model(args.difix_model_name, args.difix_model_path, args.timestep)
    ensure_dir(run_root / "difix" / "left_fixed")
    ensure_dir(run_root / "difix" / "right_fixed")
    
    for idx, rec in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] frame_id={rec.frame_id} test_idx={rec.test_idx} placement={rec.placement}")
        run_single_difix(
            model_bundle=model,
            image_path=Path(rec.render_input_path),
            ref_path=Path(rec.left_ref_input_path),
            output_path=Path(rec.left_fixed_path),
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            overwrite=args.overwrite,
        )
        run_single_difix(
            model_bundle=model,
            image_path=Path(rec.render_input_path),
            ref_path=Path(rec.right_ref_input_path),
            output_path=Path(rec.right_fixed_path),
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            overwrite=args.overwrite,
        )
    
    write_json(
        run_root / "manifests" / "difix_run_manifest.json",
        {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "timestep": args.timestep,
            "difix_model_name": args.difix_model_name,
            "num_records": len(records),
            "frame_ids": [r.frame_id for r in records],
        },
    )
    return records


def stage_pack(args: argparse.Namespace, run_root: Path) -> None:
    source = read_json(run_root / "manifests" / "source_manifest.json")
    records = load_pseudo_records(run_root)
    
    # Load pose
    trj = read_json(Path(args.trj_json))
    trj_id = trj["trj_id"]
    trj_est = trj["trj_est"]
    
    left_rgb = run_root / "augmented_train_left" / "rgb"
    right_rgb = run_root / "augmented_train_right" / "rgb"
    ensure_dir(left_rgb)
    ensure_dir(right_rgb)
    
    # Copy/symlink train frames
    sparse_rgb_dir = Path(args.sparse_rgb_dir)
    sparse_indices = source["sparse_indices"]
    
    for frame_id in sparse_indices:
        name = get_sparse_rgb_name(frame_id, args.scene_name)
        canonical = get_canonical_name(frame_id, args.scene_name)
        src = sparse_rgb_dir / name
        if not args.dry_run:
            symlink_force(src, left_rgb / canonical)
            symlink_force(src, right_rgb / canonical)
    
    # Copy difix outputs
    for rec in records:
        canonical = get_canonical_name(rec.frame_id, args.scene_name)
        left_src = Path(rec.left_fixed_path)
        right_src = Path(rec.right_fixed_path)
        if not left_src.is_file() or not right_src.is_file():
            raise FileNotFoundError(f"Difix outputs missing for frame {rec.frame_id}")
        if not args.dry_run:
            copy_force(left_src, left_rgb / canonical)
            copy_force(right_src, right_rgb / canonical)
    
    # Create pseudo_cache
    pseudo_cache = run_root / "pseudo_cache"
    render_depth_dir = Path(args.render_depth_dir)
    
    for rec in records:
        sample_dir = pseudo_cache / "samples" / str(rec.frame_id)
        ensure_dir(sample_dir)
        
        # Save camera.json
        pose_idx = trj_id.index(rec.test_idx) if rec.test_idx in trj_id else -1
        if pose_idx >= 0:
            camera = {
                "frame_id": rec.frame_id,
                "test_idx": rec.test_idx,
                "pose_c2w": trj_est[pose_idx],
                "left_ref_frame_id": rec.gap_left_train_id,
                "right_ref_frame_id": rec.gap_right_train_id,
            }
            write_json(sample_dir / "camera.json", camera)
        
        # Copy render_rgb and render_depth
        canonical = get_canonical_name(rec.frame_id, args.scene_name)
        render_src = Path(rec.render_src_path)
        depth_src = render_depth_dir / get_render_depth_name(rec.test_idx)
        
        if not args.dry_run:
            if render_src.exists():
                copy_force(render_src, sample_dir / "render_rgb.png")
            if depth_src.exists():
                copy_force(depth_src, sample_dir / "render_depth.npy")
    
    # Save pseudo_cache manifest
    cache_manifest = {
        "scene_name": args.scene_name,
        "run_key": args.run_key,
        "num_samples": len(records),
        "sample_ids": [r.frame_id for r in records],
        "samples": [
            {
                "frame_id": r.frame_id,
                "test_idx": r.test_idx,
                "placement": r.placement,
                "left_ref_frame_id": r.gap_left_train_id,
                "right_ref_frame_id": r.gap_right_train_id,
                "camera_path": r.pseudo_camera_path,
                "render_rgb": str((pseudo_cache / "samples" / str(r.frame_id) / "render_rgb.png").resolve()),
                "render_depth": str((pseudo_cache / "samples" / str(r.frame_id) / "render_depth.npy").resolve()),
            }
            for r in records
        ],
    }
    write_json(pseudo_cache / "manifest.json", cache_manifest)
    
    # Save pack manifest
    pack_manifest = {
        "scene_name": args.scene_name,
        "run_key": args.run_key,
        "train_ids": sparse_indices,
        "pseudo_ids": [r.frame_id for r in records],
        "augmented_train_left_rgb": str(left_rgb.resolve()),
        "augmented_train_right_rgb": str(right_rgb.resolve()),
        "num_left_files": len(list(left_rgb.glob("*.png"))),
        "num_right_files": len(list(right_rgb.glob("*.png"))),
    }
    write_json(run_root / "manifests" / "pack_manifest.json", pack_manifest)
    print(json.dumps(pack_manifest, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    run_root = Path(args.dataset_root) / args.scene_name / "part3_stage1" / args.run_key
    ensure_dir(run_root)
    ensure_dir(run_root / "manifests")
    
    if args.stage in ("select", "all"):
        stage_select(args, run_root)
    if args.stage in ("difix", "all"):
        stage_difix(args, run_root)
    if args.stage in ("pack", "all"):
        stage_pack(args, run_root)


if __name__ == "__main__":
    main()
