#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from munch import munchify

ROOT = Path(__file__).resolve().parents[1]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)
if f"{S3PO_ROOT}/gaussian_splatting" not in sys.path:
    sys.path.insert(0, f"{S3PO_ROOT}/gaussian_splatting")

from pseudo_branch.common.flow_matcher import FlowMatcher
from pseudo_branch.observation.brpo_reprojection_verify import (  # noqa: E402
    find_neighbor_kfs,
    render_depth_from_state,
    verify_single_branch,
)
from pseudo_branch.mask.brpo_confidence_mask import (  # noqa: E402
    build_brpo_confidence_mask,
    summarize_brpo_mask,
)
from pseudo_branch.target.brpo_depth_target import build_blended_target_depth  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.external_eval_utils import load_gaussians_from_ply  # noqa: E402


SELECTION_VERSION = "signal-aware-selection-v2"
DEFAULT_CANDIDATE_FRACTIONS = (1.0 / 3.0, 0.5, 2.0 / 3.0)


def parse_args():
    p = argparse.ArgumentParser(description="Select signal-aware pseudo frames from internal cache")
    p.add_argument("--internal-cache-root", required=True)
    p.add_argument("--stage-tag", choices=["before_opt", "after_opt"], default="after_opt")
    p.add_argument("--output-root", required=True)
    p.add_argument("--candidate-fractions", default="1/3,1/2,2/3")
    p.add_argument("--pseudo-fused-root", default=None, help="Optional root with per-frame fused pseudo RGBs; falls back to render_rgb when missing")
    p.add_argument("--tau-reproj-px", type=float, default=4.0)
    p.add_argument("--tau-rel-depth", type=float, default=0.15)
    p.add_argument("--cont-tau-reproj", type=float, default=4.0)
    p.add_argument("--cont-tau-depth", type=float, default=0.15)
    p.add_argument("--cont-tau-agree", type=float, default=0.10)
    p.add_argument("--w-both", type=float, default=4.0)
    p.add_argument("--w-verified", type=float, default=2.0)
    p.add_argument("--w-correction", type=float, default=1.0)
    p.add_argument("--w-balance", type=float, default=2.0)
    p.add_argument("--topk-per-gap", type=int, default=1)
    p.add_argument("--allocation-policy", default="score_topk")
    p.add_argument("--limit-gaps", type=int, default=None)
    p.add_argument("--sh-degree", type=int, default=None, help="Override SH degree for loading stage PLY; default auto-infer from PLY header")
    return p.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_fraction_token(token: str) -> float:
    token = token.strip()
    if not token:
        raise ValueError("Empty candidate fraction token")
    if "/" in token:
        num, den = token.split("/", 1)
        value = float(num) / float(den)
    else:
        value = float(token)
    if not (0.0 < value < 1.0):
        raise ValueError(f"Candidate fraction must be in (0,1), got {token}")
    return value


def format_fraction_label(value: float) -> str:
    pct = int(round(value * 1000))
    return f"frac_{pct:03d}"


def canonical_name(frame_id: int) -> str:
    return f"{int(frame_id):05d}.png"


def nearest_frame(target: float, frames: List[int]) -> int:
    return min(frames, key=lambda x: (abs(x - target), x))


def enumerate_gap_candidates(left_kf: int, right_kf: int, rendered_non_kf: List[int], fractions: Tuple[float, ...]):
    gap = sorted({fid for fid in rendered_non_kf if left_kf < fid < right_kf})
    if not gap:
        return []
    out = []
    for frac in fractions:
        target = left_kf + frac * (right_kf - left_kf)
        out.append(
            {
                "candidate_fraction": float(frac),
                "candidate_label": format_fraction_label(frac),
                "candidate_target": float(target),
                "frame_id": int(nearest_frame(target, gap)),
            }
        )
    dedup = {}
    for item in out:
        dedup.setdefault(item["frame_id"], item)
    return [dedup[k] for k in sorted(dedup.keys())]


def resolve_pseudo_rgb_path(frame_id: int, image_name: str, render_rgb_path: Path, pseudo_fused_root: str | None) -> Tuple[Path, str]:
    if pseudo_fused_root:
        root = Path(pseudo_fused_root)
        candidates = [
            root / image_name,
            root / str(int(frame_id)) / "target_rgb_fused.png",
            root / f"frame_{int(frame_id):04d}" / "target_rgb_fused.png",
            root / f"{int(frame_id):05d}.png",
        ]
        for cand in candidates:
            if cand.exists():
                return cand, "fused_root"
    return render_rgb_path, "render_rgb"


def run_branch(
    side: str,
    frame_id: int,
    pseudo_state: Dict,
    ref_state: Dict,
    pseudo_rgb_path: Path,
    pseudo_depth_path: Path,
    gaussians,
    pipe,
    background,
    matcher: FlowMatcher,
    tau_reproj_px: float,
    tau_rel_depth: float,
):
    ref_rgb_path = Path(ref_state["image_path"])
    pseudo_depth = np.load(pseudo_depth_path).astype(np.float32)
    ref_depth = render_depth_from_state(gaussians, ref_state, pipe, background)
    pts_pseudo, pts_ref, _ = matcher.match_pair(str(pseudo_rgb_path), str(ref_rgb_path), size=int(pseudo_state["image_width"]))
    result = verify_single_branch(
        pseudo_state=pseudo_state,
        ref_state=ref_state,
        pseudo_depth=pseudo_depth,
        ref_depth=ref_depth,
        pts_pseudo=pts_pseudo,
        pts_ref=pts_ref,
        tau_reproj_px=tau_reproj_px,
        tau_rel_depth=tau_rel_depth,
    )
    return result


def compute_correction_metrics(render_depth: np.ndarray, target_depth: np.ndarray, verified_mask: np.ndarray, both_mask: np.ndarray) -> Dict[str, float]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    target_depth = np.asarray(target_depth, dtype=np.float32)
    verified = np.asarray(verified_mask, dtype=np.float32) > 0.5
    both = np.asarray(both_mask, dtype=np.float32) > 0.5
    valid_verified = verified & (render_depth > 1e-6) & (target_depth > 1e-6)
    valid_both = both & (render_depth > 1e-6) & (target_depth > 1e-6)

    def _metrics(mask: np.ndarray) -> Tuple[float, float]:
        if not mask.any():
            return 0.0, 0.0
        rel = np.abs(target_depth[mask] - render_depth[mask]) / np.clip(render_depth[mask], 1e-6, None)
        logrel = np.abs(np.log(np.clip(target_depth[mask], 1e-6, None)) - np.log(np.clip(render_depth[mask], 1e-6, None)))
        return float(rel.mean()), float(logrel.mean())

    verified_rel, verified_log = _metrics(valid_verified)
    both_rel, both_log = _metrics(valid_both)
    return {
        "mean_abs_rel_correction_verified": verified_rel,
        "mean_abs_log_correction_verified": verified_log,
        "mean_abs_rel_correction_both": both_rel,
        "mean_abs_log_correction_both": both_log,
    }


def score_candidate(frame_meta: Dict, depth_summary: Dict, correction: Dict, weights: Dict[str, float]) -> Dict[str, float]:
    both_ratio = float(frame_meta.get("support_ratio_both", 0.0))
    verified_ratio = float(depth_summary.get("verified_ratio", 0.0))
    balance_ratio = min(float(frame_meta.get("support_ratio_left", 0.0)), float(frame_meta.get("support_ratio_right", 0.0)))
    correction_mag = float(correction.get("mean_abs_rel_correction_verified", 0.0))
    total = (
        float(weights["w_both"]) * both_ratio
        + float(weights["w_verified"]) * verified_ratio
        + float(weights["w_correction"]) * correction_mag
        + float(weights["w_balance"]) * balance_ratio
    )
    return {
        "score": float(total),
        "score_terms": {
            "w_both_x_support_ratio_both": float(weights["w_both"]) * both_ratio,
            "w_verified_x_verified_ratio": float(weights["w_verified"]) * verified_ratio,
            "w_correction_x_mean_abs_rel_correction_verified": float(weights["w_correction"]) * correction_mag,
            "w_balance_x_balance_ratio": float(weights["w_balance"]) * balance_ratio,
        },
        "balance_ratio": float(balance_ratio),
    }


def aggregate_candidate_metrics(items: List[Dict]) -> Dict[str, float]:
    if not items:
        return {}
    keys = [
        "support_ratio_left",
        "support_ratio_right",
        "support_ratio_both",
        "verified_ratio",
        "both_ratio",
        "continuous_confidence_mean_positive",
        "agreement_mean_positive",
        "mean_abs_rel_correction_verified",
        "mean_abs_log_correction_verified",
        "score",
        "balance_ratio",
    ]
    out = {"num_items": int(len(items))}
    for key in keys:
        vals = [float(x[key]) for x in items if x.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def build_compare(left_items: List[Dict], right_items: List[Dict], left_key: str, right_key: str) -> Dict[str, Dict[str, float]]:
    left_agg = aggregate_candidate_metrics(left_items)
    right_agg = aggregate_candidate_metrics(right_items)
    out = {left_key: left_agg, right_key: right_agg, f"delta_{left_key}_minus_{right_key}": {}}
    for key in left_agg.keys():
        if key == "num_items":
            continue
        left_v = left_agg.get(key)
        right_v = right_agg.get(key)
        out[f"delta_{left_key}_minus_{right_key}"][key] = None if left_v is None or right_v is None else float(left_v - right_v)
    return out


def build_manifest_entry(
    frame_id: int,
    label: str,
    left_ref: int,
    right_ref: int,
    image_name: str,
    score: float,
    rank: int,
    candidate_fraction: float,
    gap_index: int,
    allocation_rank: int,
    allocation_policy: str,
):
    return {
        "frame_id": int(frame_id),
        "placement": "signal_aware_selection",
        "selection_label": label,
        "left_ref_frame_id": int(left_ref),
        "right_ref_frame_id": int(right_ref),
        "image_name": image_name,
        "selection_score": float(score),
        "selection_rank": int(rank),
        "candidate_fraction": float(candidate_fraction),
        "gap_index": int(gap_index),
        "allocation_rank": int(allocation_rank),
        "allocation_policy": allocation_policy,
    }


def main():
    args = parse_args()
    fractions = tuple(parse_fraction_token(tok) for tok in args.candidate_fractions.split(","))
    internal_cache_root = Path(args.internal_cache_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    manifest = load_json(internal_cache_root / "manifest.json")
    camera_states = load_json(internal_cache_root / "camera_states.json")
    stage_meta = load_json(internal_cache_root / args.stage_tag / "stage_meta.json")
    states_by_id = {int(s["frame_id"]): s for s in camera_states}
    kf_indices = [int(x) for x in manifest["kf_indices"]]
    rendered_non_kf = [int(x) for x in stage_meta["rendered_non_kf_frames"]]

    run_root = internal_cache_root.parent
    config = load_config(str(run_root / "config.yml"))
    pipe = munchify(config["pipeline_params"])
    background = torch.tensor(manifest["background"], dtype=torch.float32, device="cuda")
    stage_ply = internal_cache_root / args.stage_tag / "point_cloud" / "point_cloud.ply"
    gaussians = load_gaussians_from_ply(config, str(stage_ply), sh_degree=args.sh_degree)
    matcher = FlowMatcher()

    weights = {
        "w_both": float(args.w_both),
        "w_verified": float(args.w_verified),
        "w_correction": float(args.w_correction),
        "w_balance": float(args.w_balance),
    }

    gap_pairs = list(zip(kf_indices[:-1], kf_indices[1:]))
    if args.limit_gaps is not None:
        gap_pairs = gap_pairs[: int(args.limit_gaps)]

    if args.topk_per_gap < 1:
        raise ValueError("--topk-per-gap must be >= 1")

    per_gap = []
    selected_manifest = []
    selected_primary_rows = []
    selected_topk_rows = []
    midpoint_rows = []

    for gap_idx, (left_kf, right_kf) in enumerate(gap_pairs):
        candidates = enumerate_gap_candidates(left_kf, right_kf, rendered_non_kf, fractions)
        midpoint_frame_id = nearest_frame(left_kf + 0.5 * (right_kf - left_kf), [x["frame_id"] for x in candidates])
        print(f"[selection] gap {gap_idx + 1}/{len(gap_pairs)} left={left_kf} right={right_kf} candidates={[c['frame_id'] for c in candidates]}", flush=True)
        scored_candidates = []

        for cand in candidates:
            frame_id = int(cand["frame_id"])
            pseudo_state = states_by_id[frame_id]
            left_state = states_by_id[left_kf]
            right_state = states_by_id[right_kf]
            image_name = pseudo_state.get("image_name", canonical_name(frame_id))
            render_rgb_path = internal_cache_root / args.stage_tag / "render_rgb" / f"{frame_id}_pred.png"
            pseudo_depth_path = internal_cache_root / args.stage_tag / "render_depth_npy" / f"{frame_id}_pred.npy"
            pseudo_rgb_path, pseudo_rgb_source_mode = resolve_pseudo_rgb_path(frame_id, image_name, render_rgb_path, args.pseudo_fused_root)
            render_depth = np.load(pseudo_depth_path).astype(np.float32)

            left_result = run_branch(
                "left",
                frame_id,
                pseudo_state,
                left_state,
                pseudo_rgb_path,
                pseudo_depth_path,
                gaussians,
                pipe,
                background,
                matcher,
                args.tau_reproj_px,
                args.tau_rel_depth,
            )
            right_result = run_branch(
                "right",
                frame_id,
                pseudo_state,
                right_state,
                pseudo_rgb_path,
                pseudo_depth_path,
                gaussians,
                pipe,
                background,
                matcher,
                args.tau_reproj_px,
                args.tau_rel_depth,
            )
            fused = build_brpo_confidence_mask(
                left_result["support_mask"],
                right_result["support_mask"],
                left_result=left_result,
                right_result=right_result,
                tau_reproj=args.cont_tau_reproj,
                tau_depth=args.cont_tau_depth,
                tau_agree=args.cont_tau_agree,
            )
            frame_meta = summarize_brpo_mask(frame_id=frame_id, left_stats=left_result["stats"], right_stats=right_result["stats"], fused=fused)
            depth_target = build_blended_target_depth(
                render_depth=render_depth,
                projected_depth_left=left_result["projected_depth_map"],
                projected_depth_right=right_result["projected_depth_map"],
                valid_left=left_result["projected_depth_valid_mask"],
                valid_right=right_result["projected_depth_valid_mask"],
                fallback_mode="render_depth",
                both_mode="average",
            )
            depth_summary = depth_target["summary"]
            correction = compute_correction_metrics(
                render_depth=render_depth,
                target_depth=depth_target["target_depth_for_refine"],
                verified_mask=depth_target["verified_depth_mask"],
                both_mask=fused["support_both"],
            )
            scored = score_candidate(frame_meta, depth_summary, correction, weights)
            row = {
                "frame_id": frame_id,
                "image_name": image_name,
                "gap_index": int(gap_idx),
                "left_ref_frame_id": int(left_kf),
                "right_ref_frame_id": int(right_kf),
                "candidate_fraction": float(cand["candidate_fraction"]),
                "candidate_label": cand["candidate_label"],
                "candidate_target": float(cand["candidate_target"]),
                "pseudo_rgb_source_mode": pseudo_rgb_source_mode,
                "pseudo_rgb_path": str(pseudo_rgb_path),
                "render_rgb_path": str(render_rgb_path),
                "render_depth_path": str(pseudo_depth_path),
                "support_ratio_left": float(frame_meta["support_ratio_left"]),
                "support_ratio_right": float(frame_meta["support_ratio_right"]),
                "support_ratio_both": float(frame_meta["support_ratio_both"]),
                "support_ratio_single": float(frame_meta["support_ratio_single"]),
                "verified_ratio": float(depth_summary["verified_ratio"]),
                "both_ratio": float(depth_summary["both_ratio"]),
                "left_only_ratio": float(depth_summary["left_only_ratio"]),
                "right_only_ratio": float(depth_summary["right_only_ratio"]),
                "render_fallback_ratio": float(depth_summary["render_fallback_ratio"]),
                "continuous_confidence_mean_positive": float(frame_meta.get("continuous_confidence_summary", {}).get("mean_positive", 0.0)),
                "continuous_confidence_nonzero_ratio": float(frame_meta.get("continuous_confidence_summary", {}).get("nonzero_ratio", 0.0)),
                "agreement_mean_positive": float(frame_meta.get("agreement_summary", {}).get("mean_positive", 0.0)),
                "agreement_nonzero_ratio": float(frame_meta.get("agreement_summary", {}).get("nonzero_ratio", 0.0)),
                **correction,
                **scored,
            }
            print(
                "[selection]   frame={frame_id} label={label} both={both:.6f} verified={verified:.6f} corr={corr:.6f} score={score:.6f}".format(
                    frame_id=frame_id,
                    label=cand["candidate_label"],
                    both=row["support_ratio_both"],
                    verified=row["verified_ratio"],
                    corr=row["mean_abs_rel_correction_verified"],
                    score=row["score"],
                ),
                flush=True,
            )
            scored_candidates.append(row)

        scored_candidates.sort(key=lambda x: (-x["score"], -x["support_ratio_both"], -x["verified_ratio"], x["frame_id"]))
        for rank, row in enumerate(scored_candidates, start=1):
            row["rank_in_gap"] = int(rank)
        selected = scored_candidates[0]
        midpoint = min(scored_candidates, key=lambda x: (abs(x["candidate_fraction"] - 0.5), x["frame_id"]))
        selected_topk = scored_candidates[: min(args.topk_per_gap, len(scored_candidates))]

        selected_primary_rows.append(selected)
        selected_topk_rows.extend(selected_topk)
        midpoint_rows.append(midpoint)
        for alloc_rank, row in enumerate(selected_topk, start=1):
            selected_manifest.append(
                build_manifest_entry(
                    frame_id=row["frame_id"],
                    label=row["candidate_label"],
                    left_ref=left_kf,
                    right_ref=right_kf,
                    image_name=row["image_name"],
                    score=row["score"],
                    rank=row["rank_in_gap"],
                    candidate_fraction=row["candidate_fraction"],
                    gap_index=gap_idx,
                    allocation_rank=alloc_rank,
                    allocation_policy=args.allocation_policy,
                )
            )

        per_gap.append(
            {
                "gap_index": int(gap_idx),
                "left_kf": int(left_kf),
                "right_kf": int(right_kf),
                "midpoint_frame_id": int(midpoint["frame_id"]),
                "selected_frame_id": int(selected["frame_id"]),
                "selected_candidate_fraction": float(selected["candidate_fraction"]),
                "selected_candidate_label": selected["candidate_label"],
                "selection_changed_vs_midpoint": bool(int(selected["frame_id"]) != int(midpoint["frame_id"])),
                "topk_per_gap": int(args.topk_per_gap),
                "selected_topk_frame_ids": [int(x["frame_id"]) for x in selected_topk],
                "selected_topk_candidate_fractions": [float(x["candidate_fraction"]) for x in selected_topk],
                "selected": selected,
                "selected_topk": selected_topk,
                "midpoint": midpoint,
                "candidates": scored_candidates,
            }
        )

    compare_primary = build_compare(selected_primary_rows, midpoint_rows, "selected", "midpoint")
    compare_topk = build_compare(selected_topk_rows, midpoint_rows, "selected_topk", "midpoint")

    selection_report = {
        "selection_version": SELECTION_VERSION,
        "internal_cache_root": str(internal_cache_root),
        "stage_tag": args.stage_tag,
        "stage_ply": str(stage_ply),
        "candidate_fractions": [float(x) for x in fractions],
        "topk_per_gap": int(args.topk_per_gap),
        "allocation_policy": args.allocation_policy,
        "pseudo_fused_root": args.pseudo_fused_root,
        "score_weights": weights,
        "verification_policy": {
            "tau_reproj_px": float(args.tau_reproj_px),
            "tau_rel_depth": float(args.tau_rel_depth),
            "cont_tau_reproj": float(args.cont_tau_reproj),
            "cont_tau_depth": float(args.cont_tau_depth),
            "cont_tau_agree": float(args.cont_tau_agree),
        },
        "num_gaps": int(len(per_gap)),
        "num_selected": int(len(selected_manifest)),
        "selected_frame_ids": [int(x["frame_id"]) for x in selected_primary_rows],
        "selected_topk_frame_ids": [int(x["frame_id"]) for x in selected_manifest],
        "midpoint_frame_ids": [int(x["frame_id"]) for x in midpoint_rows],
        "num_changed_vs_midpoint": int(sum(1 for x in per_gap if x["selection_changed_vs_midpoint"])),
        "compare_selected_vs_midpoint": compare_primary,
        "compare_selected_topk_vs_midpoint": compare_topk,
        "per_gap": per_gap,
    }

    selection_manifest = {
        "selection_version": SELECTION_VERSION,
        "internal_cache_root": str(internal_cache_root),
        "stage_tag": args.stage_tag,
        "candidate_fractions": [float(x) for x in fractions],
        "topk_per_gap": int(args.topk_per_gap),
        "allocation_policy": args.allocation_policy,
        "num_selected": int(len(selected_manifest)),
        "selected_frame_ids": [int(x["frame_id"]) for x in selected_manifest],
        "records": selected_manifest,
    }

    write_json(output_root / "reports" / "signal_aware_selection_report.json", selection_report)
    write_json(output_root / "manifests" / "signal_aware_selection_manifest.json", selection_manifest)
    write_json(
        output_root / "manifests" / "selection_summary.json",
        {
            "selection_version": SELECTION_VERSION,
            "topk_per_gap": int(args.topk_per_gap),
            "allocation_policy": args.allocation_policy,
            "num_selected": int(len(selected_manifest)),
            "selected_frame_ids": [int(x["frame_id"]) for x in selected_manifest],
            "midpoint_frame_ids": [int(x["frame_id"]) for x in midpoint_rows],
            "num_changed_vs_midpoint": int(sum(1 for x in per_gap if x["selection_changed_vs_midpoint"])),
        },
    )

    print(json.dumps({
        "selection_version": SELECTION_VERSION,
        "topk_per_gap": int(args.topk_per_gap),
        "allocation_policy": args.allocation_policy,
        "selected_frame_ids": [int(x["frame_id"]) for x in selected_manifest],
        "midpoint_frame_ids": [int(x["frame_id"]) for x in midpoint_rows],
        "num_changed_vs_midpoint": int(sum(1 for x in per_gap if x["selection_changed_vs_midpoint"])),
        "compare_selected_vs_midpoint": compare_primary,
        "compare_selected_topk_vs_midpoint": compare_topk,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
