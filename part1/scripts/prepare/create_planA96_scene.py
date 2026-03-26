#!/usr/bin/env python3
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, "/home/bzhang512/CV_Project/gaussian-splatting")
from utils.read_write_model import read_model, write_model, Image, Point3D

src_model = Path("/home/bzhang512/CV_Project/part1/part1_data/re10k_1/planA_colmap/gs_scene/sparse/0")
src_images_dir = Path("/home/bzhang512/CV_Project/part1/part1_data/re10k_1/images")
subset_dir = Path("/home/bzhang512/CV_Project/part1/part1_data/re10k_1/planB_vggt/inputs/re10k1_scene_96/images")
out_root = Path("/home/bzhang512/CV_Project/part1/part1_data/re10k_1/planA_colmap_96/gs_scene")
out_sparse = out_root / "sparse" / "0"
out_images = out_root / "images"

subset_names = sorted(p.name for p in subset_dir.iterdir() if p.is_symlink() or p.is_file())
if not subset_names:
    raise SystemExit("No subset images found")
subset_set = set(subset_names)

cams, imgs, pts = read_model(str(src_model), ext=".bin")
keep_imgs = {iid: img for iid, img in imgs.items() if img.name in subset_set}
if len(keep_imgs) != len(subset_set):
    missing = sorted(subset_set - {img.name for img in keep_imgs.values()})
    raise SystemExit(f"Subset names missing from Plan A model: {missing[:10]}")
keep_img_ids = set(keep_imgs.keys())

new_pts = {}
kept_point_ids = set()
for pid, pt in pts.items():
    mask = np.array([iid in keep_img_ids for iid in pt.image_ids], dtype=bool)
    if mask.sum() < 2:
        continue
    new_pt = Point3D(
        id=pt.id,
        xyz=pt.xyz,
        rgb=pt.rgb,
        error=pt.error,
        image_ids=pt.image_ids[mask],
        point2D_idxs=pt.point2D_idxs[mask],
    )
    new_pts[pid] = new_pt
    kept_point_ids.add(pid)

new_imgs = {}
for iid, img in keep_imgs.items():
    point_ids = np.array([pid if pid in kept_point_ids else -1 for pid in img.point3D_ids], dtype=np.int64)
    new_imgs[iid] = Image(
        id=img.id,
        qvec=img.qvec,
        tvec=img.tvec,
        camera_id=img.camera_id,
        name=img.name,
        xys=img.xys,
        point3D_ids=point_ids,
    )

if out_root.exists():
    shutil.rmtree(out_root)
out_sparse.mkdir(parents=True, exist_ok=True)
out_images.mkdir(parents=True, exist_ok=True)

for name in subset_names:
    src = src_images_dir / name
    dst = out_images / name
    dst.symlink_to(src)

write_model(cams, new_imgs, new_pts, str(out_sparse), ext=".bin")
print(f"subset_images={len(new_imgs)}")
print(f"subset_points={len(new_pts)}")
print(f"out_root={out_root}")
