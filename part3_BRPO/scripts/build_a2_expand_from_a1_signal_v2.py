#!/usr/bin/env python3
"""Build A2 expanded signal_v2 from existing A1 signal_v2 + fusion artifacts.

This script takes an existing A1 signal_v2 directory and adds A2 expansion
artifacts by reading fusion artifacts from prepare_root.

Usage:
    python scripts/build_a2_expand_from_a1_signal_v2.py         --a1-signal-root /path/to/a1_signal_v2         --prepare-root /path/to/prepare         --output-root /path/to/a2_signal_v2         [--frame-ids 23 57 92 ...]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pseudo_branch.brpo_v2_signal.support_expand import (
    build_support_expand_from_a1,
    write_support_expand_outputs,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build A2 expanded signal_v2 from A1")
    p.add_argument("--a1-signal-root", required=True, help="A1 signal_v2 directory")
    p.add_argument("--prepare-root", required=True, help="Prepare root with fusion/samples/")
    p.add_argument("--output-root", default=None, help="Output directory (default: same as a1-signal-root)")
    p.add_argument("--frame-ids", type=int, nargs="*", default=None)
    p.add_argument("--seed-threshold", type=float, default=0.7)
    p.add_argument("--depth-diff-threshold", type=float, default=0.05)
    p.add_argument("--fusion-weight-threshold", type=float, default=0.3)
    p.add_argument("--max-expand-iterations", type=int, default=3)
    return p.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    a1_signal_root = Path(args.a1_signal_root)
    prepare_root = Path(args.prepare_root)
    output_root = Path(args.output_root) if args.output_root else a1_signal_root
    
    # Find frame directories in A1 signal_v2
    frame_dirs = sorted([d for d in a1_signal_root.iterdir() if d.is_dir() and d.name.startswith("frame_")])
    if not frame_dirs:
        print(f"No frame directories found in {a1_signal_root}")
        return
    
    # Extract frame IDs
    all_frame_ids = [int(d.name.split("_")[1]) for d in frame_dirs]
    if args.frame_ids:
        wanted = set(args.frame_ids)
        frame_ids = [fid for fid in all_frame_ids if fid in wanted]
    else:
        frame_ids = all_frame_ids
    
    if not frame_ids:
        print("No matching frame IDs")
        return
    
    print(f"Processing {len(frame_ids)} frames: {frame_ids}")
    
    summary = []
    for frame_id in frame_ids:
        frame_dir_a1 = a1_signal_root / f"frame_{frame_id:04d}"
        fusion_dir = prepare_root / "fusion" / "samples" / str(frame_id)
        
        if not frame_dir_a1.exists():
            print(f"  Skipping frame {frame_id}: A1 dir not found")
            continue
        if not fusion_dir.exists():
            print(f"  Skipping frame {frame_id}: fusion dir not found")
            continue
        
        # Load A1 artifacts
        joint_conf = np.load(frame_dir_a1 / "joint_confidence_v2.npy")
        joint_cont = np.load(frame_dir_a1 / "joint_confidence_cont_v2.npy")
        joint_depth = np.load(frame_dir_a1 / "joint_depth_target_v2.npy")
        
        # Load fusion artifacts
        proj_left = np.load(fusion_dir / "projected_depth_left.npy")
        proj_right = np.load(fusion_dir / "projected_depth_right.npy")
        overlap_left = np.load(fusion_dir / "overlap_mask_left.npy")
        overlap_right = np.load(fusion_dir / "overlap_mask_right.npy")
        fusion_left = np.load(fusion_dir / "fusion_weight_left.npy")
        fusion_right = np.load(fusion_dir / "fusion_weight_right.npy")
        
        # Build A2 expansion
        result, meta = build_support_expand_from_a1(
            joint_confidence_v2=joint_conf,
            joint_confidence_cont_v2=joint_cont,
            joint_depth_target_v2=joint_depth,
            projected_depth_left=proj_left,
            projected_depth_right=proj_right,
            overlap_mask_left=overlap_left,
            overlap_mask_right=overlap_right,
            fusion_weight_left=fusion_left,
            fusion_weight_right=fusion_right,
            seed_threshold=args.seed_threshold,
            depth_diff_threshold=args.depth_diff_threshold,
            fusion_weight_threshold=args.fusion_weight_threshold,
            max_expand_iterations=args.max_expand_iterations,
        )
        
        # Write A2 artifacts to output directory
        frame_out = output_root / f"frame_{frame_id:04d}"
        meta["frame_id"] = frame_id
        write_support_expand_outputs(frame_out, result, meta)
        
        print(f"  frame {frame_id}: A1 {meta['final_summary']['a1_nonzero_ratio']*100:.2f}% -> A2 {meta['final_summary']['a2_nonzero_ratio']*100:.2f}% (+{meta['final_summary']['coverage_gain_ratio']*100:.2f}%)")
        
        summary.append({
            "frame_id": frame_id,
            "a1_nonzero_ratio": meta["final_summary"]["a1_nonzero_ratio"],
            "a2_nonzero_ratio": meta["final_summary"]["a2_nonzero_ratio"],
            "coverage_gain_ratio": meta["final_summary"]["coverage_gain_ratio"],
            "source_distribution": meta["final_summary"]["source_distribution"],
        })
    
    # Write summary
    summary_meta = {
        "a1_signal_root": str(a1_signal_root.resolve()),
        "prepare_root": str(prepare_root.resolve()),
        "output_root": str(output_root.resolve()),
        "num_frames": len(summary),
        "parameters": {
            "seed_threshold": args.seed_threshold,
            "depth_diff_threshold": args.depth_diff_threshold,
            "fusion_weight_threshold": args.fusion_weight_threshold,
            "max_expand_iterations": args.max_expand_iterations,
        },
        "mean_a1_coverage": sum(s["a1_nonzero_ratio"] for s in summary) / len(summary) if summary else 0,
        "mean_a2_coverage": sum(s["a2_nonzero_ratio"] for s in summary) / len(summary) if summary else 0,
        "mean_coverage_gain": sum(s["coverage_gain_ratio"] for s in summary) / len(summary) if summary else 0,
    }
    
    write_json(output_root / "summary_a2_expand.json", summary)
    write_json(output_root / "summary_meta_a2_expand.json", summary_meta)
    
    print(f"Done. Mean coverage: A1 {summary_meta['mean_a1_coverage']*100:.2f}% -> A2 {summary_meta['mean_a2_coverage']*100:.2f}%")
    print(f"Output written to: {output_root}")


if __name__ == "__main__":
    main()

