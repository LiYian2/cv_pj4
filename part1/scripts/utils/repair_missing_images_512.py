#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

PROJECT_ROOT = Path.home() / "CV_Project"
DATASET_ROOT = PROJECT_ROOT / "dataset"
TARGET_SIZE = (512, 512)

SCENE_TO_RAW_SUBDIR = {
    "DL3DV-2": "rgb",
    "Re10k-1": "images",
    "405841": None,
}

PART2_LINK_DIRS = [
    "part2_s3po/full/rgb",
    "part2_s3po/sparse/rgb",
    "part2_s3po/test/rgb",
    "part2_s3po/sparse/rgb_512",
]


def is_resolvable_file(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError):
        return False
    return resolved.is_file()


def classify_path(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "missing"
    if path.is_symlink():
        try:
            resolved = path.resolve(strict=True)
            if resolved.is_file():
                return "symlink-ok"
            return "symlink-nonfile"
        except RuntimeError:
            return "symlink-loop"
        except FileNotFoundError:
            return "symlink-broken"
        except OSError as e:
            if e.errno == errno.ELOOP:
                return "symlink-loop"
            return f"symlink-oserror:{e.errno}"
    if path.is_file():
        return "file"
    return "other"


def iter_candidate_sources(scene_root: Path, name: str) -> Iterable[Path]:
    raw_subdir = SCENE_TO_RAW_SUBDIR.get(scene_root.name)
    candidates = []
    if raw_subdir:
        candidates.append(scene_root / raw_subdir / name)
    candidates.extend([
        scene_root / "part1/shared/raw_images" / name,
        scene_root / "part1/planA/colmap_full/images" / name,
        scene_root / "part1/planA/colmap_108/images" / name,
        scene_root / "part1/planB/vggt_108/inputs/scene/images" / name,
        scene_root / "part1/planB/vggt_108/gs_scene/images" / name,
    ])
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        yield p


def materialize_image_512(scene_root: Path, name: str, verbose: bool = True) -> tuple[bool, str]:
    dst = scene_root / "part1/shared/images_512" / name
    src_used = None
    for src in iter_candidate_sources(scene_root, name):
        if is_resolvable_file(src):
            src_used = src.resolve(strict=True)
            break
    if src_used is None:
        return False, "no_valid_source"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()

    with Image.open(src_used) as im:
        resized = im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        resized.save(tmp, format="PNG")
    tmp.replace(dst)
    return True, str(src_used)


def repair_part2_links(scene_root: Path) -> list[tuple[str, str]]:
    image_512_dir = scene_root / "part1/shared/images_512"
    repaired = []
    for rel_dir in PART2_LINK_DIRS:
        link_dir = scene_root / rel_dir
        if not link_dir.is_dir():
            continue
        for entry in sorted(link_dir.iterdir()):
            target = image_512_dir / entry.name
            if not target.is_file():
                continue
            state = classify_path(entry)
            desired = str(target)
            current = os.readlink(entry) if entry.is_symlink() else None
            if state == "symlink-ok":
                try:
                    if entry.resolve(strict=True) == target.resolve(strict=True):
                        continue
                except Exception:
                    pass
            if entry.exists() or entry.is_symlink():
                entry.unlink()
            entry.symlink_to(target)
            repaired.append((rel_dir, entry.name))
    return repaired


def inspect_scene(scene_root: Path) -> dict[str, dict[str, int]]:
    checks = [
        "part1/shared/images_512",
        "part1/shared/raw_images",
        "images",
        "rgb",
        "part2_s3po/full/rgb",
        "part2_s3po/sparse/rgb",
        "part2_s3po/test/rgb",
        "part2_s3po/sparse/rgb_512",
    ]
    out = {}
    for rel in checks:
        d = scene_root / rel
        if not d.is_dir():
            continue
        stat = {"total": 0, "file": 0, "symlink-ok": 0, "symlink-broken": 0, "symlink-loop": 0, "other": 0}
        for p in d.iterdir():
            state = classify_path(p)
            stat["total"] += 1
            if state in stat:
                stat[state] += 1
            else:
                stat["other"] += 1
        out[rel] = stat
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+", help="Scene names, e.g. Re10k-1 DL3DV-2")
    args = parser.parse_args()

    for scene in args.scenes:
        scene_root = DATASET_ROOT / scene
        image_512_dir = scene_root / "part1/shared/images_512"
        if not image_512_dir.is_dir():
            print(f"[skip] {scene}: missing {image_512_dir}")
            continue

        print(f"===== {scene} =====")
        before = inspect_scene(scene_root)
        print("[before]")
        for rel, stat in before.items():
            print(f"  {rel}: {stat}")

        raw_dir = scene_root / "part1/shared/raw_images"
        expected_names = {p.name for p in image_512_dir.iterdir()}
        if raw_dir.is_dir():
            expected_names |= {p.name for p in raw_dir.iterdir()}

        needs_repair = []
        for name in sorted(expected_names):
            state = classify_path(image_512_dir / name)
            if state in {"missing", "symlink-broken", "symlink-loop"}:
                needs_repair.append(name)

        print(f"[images_512] need_repair={len(needs_repair)}")
        repaired_names = []
        for name in needs_repair:
            ok, detail = materialize_image_512(scene_root, name)
            print(f"  {'OK' if ok else 'FAIL'} {name} <- {detail}")
            if ok:
                repaired_names.append(name)

        relinked = repair_part2_links(scene_root)
        print(f"[part2_s3po] relinked={len(relinked)}")
        if relinked:
            preview = ", ".join([f"{rel}:{name}" for rel, name in relinked[:12]])
            print(f"  sample: {preview}")

        after = inspect_scene(scene_root)
        print("[after]")
        for rel, stat in after.items():
            print(f"  {rel}: {stat}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
