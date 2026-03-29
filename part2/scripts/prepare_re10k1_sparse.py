#!/usr/bin/env python3
from pathlib import Path
import json
src = Path('/home/bzhang512/CV_Project/part1/part1_data/re10k_1/images')
dst = Path('/home/bzhang512/CV_Project/part2/data/re10k_1_sparse')
img_dst = dst / 'images'
img_dst.mkdir(parents=True, exist_ok=True)
files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}])
step = 30
selected = files[::step]
if files and selected[-1].name != files[-1].name:
    selected.append(files[-1])
for p in img_dst.iterdir():
    p.unlink()
for p in selected:
    (img_dst / p.name).symlink_to(p)
meta = {
    'source_root': str(src),
    'dataset_name': 'Re10k-1',
    'selection_rule': 'every 30th frame from sorted image list, plus last frame if not included',
    'sampling_step': step,
    'original_num_images': len(files),
    'selected_num_images': len(selected),
    'selected_filenames': [p.name for p in selected],
}
(dst / 'image_list.txt').write_text('\n'.join(p.name for p in selected) + '\n')
(dst / 'meta.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
