#!/usr/bin/env python3
"""Three-arm comparison: Control vs A1 vs A1+A2 for Re10k StageB.

Arm 1: Control (rgb-first sidecar) - brpo_v2_raw mask + brpo_v2 depth
Arm 2: A1 (joint confidence) - joint_confidence_v2 mask + joint_depth_v2
Arm 3: A1+A2 (expand) - joint_confidence_expand_v1 mask + joint_depth_expand_v1

Usage:
    python scripts/run_a1_a2_stageb_compare.py         --signal-v2-root /path/to/signal_v2_with_a2_expand         --output-root /path/to/output         --num-frames 8
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Three-arm StageB comparison")
    p.add_argument("--signal-v2-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--py", default="/home/bzhang512/miniconda3/envs/s3po-gs/bin/python")
    p.add_argument("--run-script", default="scripts/run_pseudo_refinement_v2.py")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_arm(args, arm_name, rgb_mode, depth_mode, target_depth_mode, output_dir):
    """Run one arm of the comparison."""
    cmd = [
        args.py,
        args.run_script,
        "--mode", "stageB",
        "--signal-v2-root", args.signal_v2_root,
        "--stageB_output_root", output_dir,
        "--stageA_rgb_mask_mode", rgb_mode,
        "--stageA_depth_mask_mode", depth_mode,
        "--stageA_target_depth_mode", target_depth_mode,
        "--stageB_num_frames", str(args.num_frames),
        "--stageB_verbose",
    ]
    
    env = {
        "PYTHONPATH": "/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO",
    }
    
    print(f"Running {arm_name}: {rgb_mode} / {depth_mode} / {target_depth_mode}")
    
    if args.dry_run:
        print(f"  CMD: {chr(32).join(cmd)}")
        return None
    
    result = subprocess.run(
        cmd,
        cwd="/home/bzhang512/CV_Project/part3_BRPO",
        env={**dict(subprocess.os.environ), **env},
        capture_output=True,
        text=True,
        timeout=600,
    )
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return None
    
    # Parse results from output
    output_lines = result.stdout.strip().split("\n")
    metrics = {}
    for line in output_lines:
        if "PSNR:" in line or "SSIM:" in line or "LPIPS:" in line:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = float(parts[1].strip())
                metrics[key] = val
    
    return metrics


def main():
    args = parse_args()
    signal_v2_root = Path(args.signal_v2_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    arms = [
        ("control", "brpo_v2_raw", "brpo_v2_depth", "target_depth_for_refine_v2_brpo"),
        ("a1", "joint_confidence_v2", "joint_confidence_v2", "joint_depth_v2"),
        ("a1_a2", "joint_confidence_expand_v1", "joint_confidence_expand_v1", "joint_depth_expand_v1"),
    ]
    
    results = {}
    for arm_name, rgb_mode, depth_mode, target_depth_mode in arms:
        arm_output = output_root / f"stageB_{arm_name}_{timestamp}"
        metrics = run_arm(args, arm_name, rgb_mode, depth_mode, target_depth_mode, str(arm_output))
        if metrics:
            results[arm_name] = {
                "rgb_mode": rgb_mode,
                "depth_mode": depth_mode,
                "target_depth_mode": target_depth_mode,
                "metrics": metrics,
                "output_dir": str(arm_output),
            }
            print(f"  {arm_name}: PSNR={metrics.get('PSNR', 'N/A'):.6f}, SSIM={metrics.get('SSIM', 'N/A'):.6f}, LPIPS={metrics.get('LPIPS', 'N/A'):.6f}")
    
    # Compute deltas
    if "control" in results and "a1" in results:
        delta_a1_vs_control = {
            "PSNR": results["a1"]["metrics"].get("PSNR", 0) - results["control"]["metrics"].get("PSNR", 0),
            "SSIM": results["a1"]["metrics"].get("SSIM", 0) - results["control"]["metrics"].get("SSIM", 0),
            "LPIPS": results["a1"]["metrics"].get("LPIPS", 0) - results["control"]["metrics"].get("LPIPS", 0),
        }
        results["delta_a1_vs_control"] = delta_a1_vs_control
        print(f"Delta A1 vs Control: PSNR={delta_a1_vs_control['PSNR']:+.6f}, SSIM={delta_a1_vs_control['SSIM']:+.6f}, LPIPS={delta_a1_vs_control['LPIPS']:+.6f}")
    
    if "control" in results and "a1_a2" in results:
        delta_a1_a2_vs_control = {
            "PSNR": results["a1_a2"]["metrics"].get("PSNR", 0) - results["control"]["metrics"].get("PSNR", 0),
            "SSIM": results["a1_a2"]["metrics"].get("SSIM", 0) - results["control"]["metrics"].get("SSIM", 0),
            "LPIPS": results["a1_a2"]["metrics"].get("LPIPS", 0) - results["control"]["metrics"].get("LPIPS", 0),
        }
        results["delta_a1_a2_vs_control"] = delta_a1_a2_vs_control
        print(f"Delta A1+A2 vs Control: PSNR={delta_a1_a2_vs_control['PSNR']:+.6f}, SSIM={delta_a1_a2_vs_control['SSIM']:+.6f}, LPIPS={delta_a1_a2_vs_control['LPIPS']:+.6f}")
    
    if "a1" in results and "a1_a2" in results:
        delta_a2_vs_a1 = {
            "PSNR": results["a1_a2"]["metrics"].get("PSNR", 0) - results["a1"]["metrics"].get("PSNR", 0),
            "SSIM": results["a1_a2"]["metrics"].get("SSIM", 0) - results["a1"]["metrics"].get("SSIM", 0),
            "LPIPS": results["a1_a2"]["metrics"].get("LPIPS", 0) - results["a1"]["metrics"].get("LPIPS", 0),
        }
        results["delta_a2_vs_a1"] = delta_a2_vs_a1
        print(f"Delta A2 vs A1: PSNR={delta_a2_vs_a1['PSNR']:+.6f}, SSIM={delta_a2_vs_a1['SSIM']:+.6f}, LPIPS={delta_a2_vs_a1['LPIPS']:+.6f}")
    
    # Save results
    summary = {
        "timestamp": timestamp,
        "signal_v2_root": str(signal_v2_root),
        "output_root": str(output_root),
        "num_frames": args.num_frames,
        "arms": results,
    }
    write_json(output_root / f"compare_summary_{timestamp}.json", summary)
    print(f"Results saved to: {output_root / f'compare_summary_{timestamp}.json'}")


if __name__ == "__main__":
    main()

