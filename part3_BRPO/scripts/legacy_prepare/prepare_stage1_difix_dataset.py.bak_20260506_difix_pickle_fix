#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple



@dataclass
class SourceManifest:
    backend: str
    scene_name: str
    train_dir: str
    render_dir: str
    run_root: str
    name_width: int
    train_files: Dict[str, str]
    render_files: Dict[str, str]


@dataclass
class PseudoRecord:
    frame_id: int
    placement: str
    gap_left_train_id: int
    gap_right_train_id: int
    render_src_path: str
    left_ref_src_path: str
    right_ref_src_path: str
    render_input_path: str
    left_ref_input_path: str
    right_ref_input_path: str
    left_fixed_path: str
    right_fixed_path: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare part3 stage1 Difix-ref pseudo-view assets.")
    p.add_argument("--stage", choices=["select", "difix", "pack", "all"], default="all")
    p.add_argument("--backend", choices=["reggs", "s3po"], required=True)
    p.add_argument("--scene-name", required=True, help="e.g. Re10k-1 / DL3DV-2 / 405841")
    p.add_argument("--train-dir", type=str, default=None)
    p.add_argument("--render-dir", type=str, default=None)
    p.add_argument("--dataset-root", type=str, default="/home/bzhang512/CV_Project/dataset")
    p.add_argument("--run-key", required=True, help="subdir name under dataset/<scene>/part3_stage1/")
    p.add_argument("--placement", choices=["midpoint", "tertile", "both", "manual"], default="midpoint")
    p.add_argument("--manual-ids", type=str, default=None, help="comma-separated frame ids, e.g. 17,52,88")
    p.add_argument("--limit", type=int, default=None, help="optional cap on selected pseudo frames")
    p.add_argument("--prompt", type=str, default="remove degradation")
    p.add_argument("--difix-model-name", type=str, default="nvidia/difix_ref")
    p.add_argument("--difix-model-path", type=str, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--timestep", type=int, default=199)
    p.add_argument("--seed", type=int, default=42)
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


def _parse_reggs_id(path: Path) -> Optional[int]:
    m = re.match(r"^color_(\d+)\.png$", path.name)
    return int(m.group(1)) if m else None


def _parse_s3po_train_id(path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)\.png$", path.name)
    return int(m.group(1)) if m else None


def _parse_s3po_render_id(path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)(?:_pred)?\.png$", path.name)
    return int(m.group(1)) if m else None


def parse_frame_id(path: Path, backend: str, kind: str) -> Optional[int]:
    if backend == "reggs":
        return _parse_reggs_id(path)
    if backend == "s3po":
        if kind == "train":
            return _parse_s3po_train_id(path)
        return _parse_s3po_render_id(path)
    raise ValueError(f"Unsupported backend: {backend}")


def canonical_name(frame_id: int, backend: str, name_width: int) -> str:
    if backend == "reggs":
        return f"color_{frame_id:0{name_width}d}.png"
    if backend == "s3po":
        return f"{frame_id:0{name_width}d}.png"
    raise ValueError(f"Unsupported backend: {backend}")


def collect_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def build_source_manifest(args: argparse.Namespace, run_root: Path) -> SourceManifest:
    if not args.train_dir or not args.render_dir:
        raise ValueError("select/all stage requires --train-dir and --render-dir")

    train_dir = Path(args.train_dir)
    render_dir = Path(args.render_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    if not render_dir.is_dir():
        raise FileNotFoundError(f"render dir not found: {render_dir}")

    train_files: Dict[str, str] = {}
    render_files: Dict[str, str] = {}
    name_width = 0

    for p in collect_pngs(train_dir):
        fid = parse_frame_id(p, args.backend, kind="train")
        if fid is None:
            continue
        train_files[str(fid)] = str(p.resolve())
        digits = re.search(r"(\d+)", p.name)
        if digits:
            name_width = max(name_width, len(digits.group(1)))

    for p in collect_pngs(render_dir):
        fid = parse_frame_id(p, args.backend, kind="render")
        if fid is None:
            continue
        render_files[str(fid)] = str(p.resolve())

    if not train_files:
        raise RuntimeError(f"No train png files parsed from: {train_dir}")
    if not render_files:
        raise RuntimeError(f"No render png files parsed from: {render_dir}")
    if name_width <= 0:
        name_width = 4 if args.backend == "reggs" else 5

    manifest = SourceManifest(
        backend=args.backend,
        scene_name=args.scene_name,
        train_dir=str(train_dir.resolve()),
        render_dir=str(render_dir.resolve()),
        run_root=str(run_root.resolve()),
        name_width=name_width,
        train_files=train_files,
        render_files=render_files,
    )
    write_json(run_root / "manifests" / "source_manifest.json", asdict(manifest))
    return manifest


def load_source_manifest(run_root: Path) -> SourceManifest:
    data = read_json(run_root / "manifests" / "source_manifest.json")
    return SourceManifest(**data)


def nearest_render_id(candidates: List[int], target: float) -> int:
    return min(candidates, key=lambda x: (abs(x - target), x))


def pick_gap_targets(left_id: int, right_id: int, candidates: List[int], placement: str) -> List[Tuple[int, str]]:
    if not candidates:
        return []
    out: List[Tuple[int, str]] = []
    used: set[int] = set()

    def add(target: float, label: str) -> None:
        chosen = nearest_render_id(candidates, target)
        if chosen not in used:
            out.append((chosen, label))
            used.add(chosen)

    if placement in ("midpoint", "both"):
        add(left_id + 0.5 * (right_id - left_id), "midpoint")
    if placement in ("tertile", "both"):
        add(left_id + (right_id - left_id) / 3.0, "tertile_left")
        add(left_id + 2.0 * (right_id - left_id) / 3.0, "tertile_right")
    return out


def surrounding_train_ids(frame_id: int, train_ids: List[int]) -> Tuple[int, int]:
    left = [x for x in train_ids if x < frame_id]
    right = [x for x in train_ids if x > frame_id]
    if not left or not right:
        raise ValueError(f"frame {frame_id} is outside train gaps")
    return max(left), min(right)


def parse_manual_ids(text: Optional[str]) -> List[int]:
    if not text:
        return []
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def build_pseudo_records(source: SourceManifest, run_root: Path, placement: str, manual_ids: List[int], limit: Optional[int]) -> List[PseudoRecord]:
    train_ids = sorted(int(k) for k in source.train_files.keys())
    render_ids = sorted(int(k) for k in source.render_files.keys())
    records: List[PseudoRecord] = []

    if manual_ids:
        chosen_ids: List[Tuple[int, str]] = [(i, "manual") for i in manual_ids]
    else:
        chosen_ids = []
        for left_id, right_id in zip(train_ids[:-1], train_ids[1:]):
            candidates = [i for i in render_ids if left_id < i < right_id]
            chosen_ids.extend(pick_gap_targets(left_id, right_id, candidates, placement))

    seen: set[int] = set()
    for frame_id, label in chosen_ids:
        if frame_id in seen:
            continue
        seen.add(frame_id)
        if frame_id in train_ids:
            raise ValueError(f"selected frame {frame_id} is a train frame")
        if str(frame_id) not in source.render_files:
            raise ValueError(f"selected frame {frame_id} has no render source")
        left_id, right_id = surrounding_train_ids(frame_id, train_ids)
        name = canonical_name(frame_id, source.backend, source.name_width)
        rec = PseudoRecord(
            frame_id=frame_id,
            placement=label if manual_ids else label,
            gap_left_train_id=left_id,
            gap_right_train_id=right_id,
            render_src_path=source.render_files[str(frame_id)],
            left_ref_src_path=source.train_files[str(left_id)],
            right_ref_src_path=source.train_files[str(right_id)],
            render_input_path=str((run_root / "inputs" / "raw_render" / name).resolve()),
            left_ref_input_path=str((run_root / "inputs" / "left_ref" / name).resolve()),
            right_ref_input_path=str((run_root / "inputs" / "right_ref" / name).resolve()),
            left_fixed_path=str((run_root / "difix" / "left_fixed" / name).resolve()),
            right_fixed_path=str((run_root / "difix" / "right_fixed" / name).resolve()),
        )
        records.append(rec)

    records = sorted(records, key=lambda r: r.frame_id)
    if limit is not None:
        records = records[:limit]
    return records


def stage_select(args: argparse.Namespace, run_root: Path) -> List[PseudoRecord]:
    source = build_source_manifest(args, run_root)
    manual_ids = parse_manual_ids(args.manual_ids)
    placement = "manual" if manual_ids else args.placement
    records = build_pseudo_records(source, run_root, placement, manual_ids, args.limit)

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

    write_json(run_root / "manifests" / "pseudo_selection_manifest.json", [asdict(r) for r in records])
    summary = {
        "backend": source.backend,
        "scene_name": source.scene_name,
        "run_root": source.run_root,
        "placement": placement,
        "manual_ids": manual_ids,
        "num_train_frames": len(source.train_files),
        "num_render_frames": len(source.render_files),
        "num_selected_pseudo": len(records),
        "selected_frame_ids": [r.frame_id for r in records],
    }
    write_json(run_root / "manifests" / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return records


def load_pseudo_records(run_root: Path) -> List[PseudoRecord]:
    data = read_json(run_root / "manifests" / "pseudo_selection_manifest.json")
    return [PseudoRecord(**item) for item in data]


def load_difix_model(model_name: Optional[str], model_path: Optional[str], timestep: int):
    # Prefer official HuggingFace pipeline for model-name based inference.
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
        print(f"[{idx}/{len(records)}] frame_id={rec.frame_id} placement={rec.placement}")
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
            "difix_model_path": args.difix_model_path,
            "num_records": len(records),
            "frame_ids": [r.frame_id for r in records],
        },
    )
    return records


def safe_symlink_or_copy(src: Path, dst: Path, copy_file: bool = False) -> None:
    if copy_file:
        copy_force(src, dst)
    else:
        symlink_force(src, dst)


def stage_pack(args: argparse.Namespace, run_root: Path) -> None:
    source = load_source_manifest(run_root)
    records = load_pseudo_records(run_root)

    left_rgb = run_root / "augmented_train_left" / "rgb"
    right_rgb = run_root / "augmented_train_right" / "rgb"
    ensure_dir(left_rgb)
    ensure_dir(right_rgb)

    train_items = sorted((int(k), Path(v)) for k, v in source.train_files.items())
    for frame_id, src in train_items:
        name = canonical_name(frame_id, source.backend, source.name_width)
        if not args.dry_run:
            safe_symlink_or_copy(src, left_rgb / name, copy_file=False)
            safe_symlink_or_copy(src, right_rgb / name, copy_file=False)

    for rec in records:
        name = canonical_name(rec.frame_id, source.backend, source.name_width)
        left_src = Path(rec.left_fixed_path)
        right_src = Path(rec.right_fixed_path)
        if not left_src.is_file() or not right_src.is_file():
            raise FileNotFoundError(f"Difix outputs missing for frame {rec.frame_id}")
        if not args.dry_run:
            copy_force(left_src, left_rgb / name)
            copy_force(right_src, right_rgb / name)

    pack_manifest = {
        "backend": source.backend,
        "scene_name": source.scene_name,
        "run_root": source.run_root,
        "train_ids": sorted(int(k) for k in source.train_files.keys()),
        "pseudo_ids": [r.frame_id for r in records],
        "placements": {str(r.frame_id): r.placement for r in records},
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
