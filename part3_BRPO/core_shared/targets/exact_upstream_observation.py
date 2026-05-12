from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from core_shared.targets.depth_supervision_v2 import build_exact_upstream_depth_target


def _save_mask_png(mask: np.ndarray, path: Path):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _save_float_png(arr: np.ndarray, path: Path, vmax: float | None = None):
    arr = np.asarray(arr, dtype=np.float32)
    if vmax is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-8)
    img = np.clip(arr / vmax, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def _save_source_map_png(source_map: np.ndarray, path: Path):
    arr = np.asarray(source_map, dtype=np.uint8)
    vmax = max(int(arr.max()), 1)
    img = (arr.astype(np.float32) / float(vmax) * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def build_exact_brpo_upstream_target_observation(
    support_left_exact: np.ndarray,
    support_right_exact: np.ndarray,
    projected_depth_left_exact: np.ndarray,
    projected_depth_right_exact: np.ndarray,
    confidence_left_exact: np.ndarray,
    confidence_right_exact: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    provenance_left: np.ndarray | None = None,
    provenance_right: np.ndarray | None = None,
    use_confidence_weighted_composition: bool = True,
    target_depth_left_override: np.ndarray | None = None,
    target_depth_right_override: np.ndarray | None = None,
    target_field_semantics: str = "exact_upstream_v1",
    depth_target_rule: str | None = None,
    depth_input_semantics: str = "projected_depth_exact",
    confidence_cm_override: np.ndarray | None = None,
) -> Dict[str, np.ndarray | Dict]:
    """Exact BRPO upstream target observation builder.
    
    This is the final Phase T2 target: exact M~ + exact upstream T~.
    
    Key differences from build_exact_brpo_full_target_observation:
    1. Uses exact backend bundle (not proxy projected_depth)
    2. No render_depth fallback
    3. Continuous confidence from exact backend
    4. Provenance tracked throughout
    """    
    
    # Build exact upstream depth target
    depth_result = build_exact_upstream_depth_target(
        support_left_exact=support_left_exact,
        support_right_exact=support_right_exact,
        projected_depth_left_exact=projected_depth_left_exact,
        projected_depth_right_exact=projected_depth_right_exact,
        confidence_left_exact=confidence_left_exact,
        confidence_right_exact=confidence_right_exact,
        fusion_weight_left=fusion_weight_left,
        fusion_weight_right=fusion_weight_right,
        provenance_left=provenance_left,
        provenance_right=provenance_right,
        use_confidence_weighted_composition=use_confidence_weighted_composition,
        target_depth_left_override=target_depth_left_override,
        target_depth_right_override=target_depth_right_override,
        target_field_semantics=target_field_semantics,
        depth_input_semantics=depth_input_semantics,
    )
    
    # Build exact C_m from support sets (same as exact_brpo_full_target)
    support_left = np.asarray(support_left_exact, dtype=np.float32) > 0.5
    support_right = np.asarray(support_right_exact, dtype=np.float32) > 0.5
    
    verify_both = support_left & support_right
    verify_left_only = support_left & (~support_right)
    verify_right_only = support_right & (~support_left)
    verify_xor = verify_left_only | verify_right_only
    verify_union = support_left | support_right
    
    # Strict BRPO C_m: both->1.0, xor->0.5, none->0.0
    # OR use override if provided (for cm_expansion)
    if confidence_cm_override is not None:
        confidence_cm = np.asarray(confidence_cm_override, dtype=np.float32)
        cm_source = "cm_expansion_override"
    else:
        confidence_cm = np.zeros_like(support_left, dtype=np.float32)
        confidence_cm[verify_both] = 1.0
        confidence_cm[verify_xor] = 0.5
        cm_source = "exact_backend_support"
    
    # Raw discrete C_m from reciprocal support, kept for diagnostics even when
    # a soft override is consumed by the loss.
    raw_confidence_cm = np.zeros_like(support_left, dtype=np.float32)
    raw_confidence_cm[verify_both] = 1.0
    raw_confidence_cm[verify_xor] = 0.5
    cm_positive = confidence_cm > 0
    raw_cm_positive = raw_confidence_cm > 0
    num_pixels = float(confidence_cm.size)
    raw_effective_weight = float(raw_confidence_cm.sum() / num_pixels)
    cm_effective_weight = float(confidence_cm.sum() / num_pixels)
    cm_weight_gain_vs_raw = float(cm_effective_weight / max(raw_effective_weight, 1e-8))
    cm_added_nonzero = cm_positive & (~raw_cm_positive)
    cm_changed_existing = cm_positive & raw_cm_positive & (np.abs(confidence_cm - raw_confidence_cm) > 1e-6)
    def _value_ratio(value: float) -> float:
        return float((np.abs(confidence_cm - float(value)) <= 1e-6).mean())

    # Combine with depth target result
    depth_target = depth_result["pseudo_depth_target_exact_upstream_v1"]
    source_map = depth_result["pseudo_source_map_exact_upstream_v1"]
    valid_mask = depth_result["pseudo_valid_mask_exact_upstream_v1"]
    target_confidence = depth_result["pseudo_confidence_exact_upstream_v1"]
    
    summary = {
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": str(target_field_semantics),
        "target_loss_contract": "exact_shared_cm_v1",
        "valid_ratio": float(valid_mask.mean()),
        "cm_nonzero_ratio": float(cm_positive.mean()),
        "cm_source": cm_source,
        "confidence_cm_override_applied": bool(confidence_cm_override is not None),
        "cm_mean_positive": float(confidence_cm[cm_positive].mean()) if cm_positive.any() else 0.0,
        # Raw reciprocal-support stats.  These are intentionally raw support, not
        # expanded-soft provenance bins.
        "cm_support_stats_semantics": "raw reciprocal support for *_raw fields; consumed soft C_m for cm_*effective/weight fields",
        "cm_raw_nonzero_ratio": float(raw_cm_positive.mean()),
        "cm_raw_both_ratio": float(verify_both.mean()),
        "cm_raw_single_ratio": float(verify_xor.mean()),
        "cm_raw_effective_weight": raw_effective_weight,
        # Backward-compatible names, but now explicitly equivalent to raw support.
        "cm_both_ratio": float(verify_both.mean()),
        "cm_single_ratio": float(verify_xor.mean()),
        # Consumed C_m stats (soft override if present, otherwise raw discrete C_m).
        "cm_effective_weight": cm_effective_weight,
        "cm_weight_gain_vs_raw": cm_weight_gain_vs_raw,
        "cm_added_nonzero_ratio": float(cm_added_nonzero.mean()),
        "cm_changed_existing_ratio": float(cm_changed_existing.mean()),
        "cm_weight_1_00_ratio": _value_ratio(1.0),
        "cm_weight_0_60_ratio": _value_ratio(0.6),
        "cm_weight_0_50_ratio": _value_ratio(0.5),
        "cm_weight_0_25_ratio": _value_ratio(0.25),
        "depth_target_filled_ratio": float((depth_target > 1e-6).mean()),
        "depth_target_filled_within_raw_cm_ratio": float((depth_target[verify_union] > 1e-6).mean()) if verify_union.any() else 0.0,
        "depth_target_filled_within_cm_ratio": float((depth_target[cm_positive] > 1e-6).mean()) if cm_positive.any() else 0.0,
        "avg_target_confidence": float(target_confidence[valid_mask > 0].mean()) if (valid_mask > 0).any() else 0.0,
        "source_counts": depth_result["summary"]["source_counts"],
        "no_render_fallback": True,
        "depth_input_semantics": str(depth_result["summary"].get("depth_input_semantics", depth_input_semantics)),
        "target_depth_override_applied": bool(depth_result["summary"].get("target_depth_override_applied", False)),
        "policy": {
            "version": "exact_brpo_upstream_target_v1",
            "confidence_rule": "soft C_m override" if confidence_cm_override is not None else "strict BRPO-style C_m from exact backend support sets",
            "depth_target_rule": str(depth_target_rule or "exact upstream projected-depth composition with continuous confidence, no render fallback"),
            "strict_brpo_scope": "cm_and_target_and_upstream_backend",
            "upstream_backend": "exact_branch_native_v1",
            "target_confidence_same_source": True,
            "recommended_stageA_depth_loss_mode": "exact_shared_cm_v1",  # NOT legacy
        },
    }
    
    return {
        "pseudo_depth_target_exact_brpo_upstream_target_v1": depth_target.astype(np.float32),
        "pseudo_confidence_exact_brpo_upstream_target_v1": confidence_cm.astype(np.float32),
        "pseudo_source_map_exact_brpo_upstream_target_v1": source_map.astype(np.int16),
        "pseudo_valid_mask_exact_brpo_upstream_target_v1": valid_mask.astype(np.float32),
        "pseudo_target_confidence_exact_brpo_upstream_target_v1": target_confidence.astype(np.float32),
        "pseudo_verify_left_exact_brpo_upstream_target_v1": support_left.astype(np.float32),
        "pseudo_verify_right_exact_brpo_upstream_target_v1": support_right.astype(np.float32),
        "pseudo_verify_both_exact_brpo_upstream_target_v1": verify_both.astype(np.float32),
        "pseudo_verify_xor_exact_brpo_upstream_target_v1": verify_xor.astype(np.float32),
        "pseudo_verify_union_exact_brpo_upstream_target_v1": verify_union.astype(np.float32),
        "summary": summary,
    }


def write_exact_brpo_upstream_target_observation_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    """Write exact BRPO upstream target observation outputs."""    
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(frame_out / "pseudo_depth_target_exact_brpo_upstream_target_v1.npy", result["pseudo_depth_target_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_confidence_exact_brpo_upstream_target_v1.npy", result["pseudo_confidence_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_source_map_exact_brpo_upstream_target_v1.npy", result["pseudo_source_map_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_valid_mask_exact_brpo_upstream_target_v1.npy", result["pseudo_valid_mask_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_target_confidence_exact_brpo_upstream_target_v1.npy", result["pseudo_target_confidence_exact_brpo_upstream_target_v1"])
    
    _save_float_png(result["pseudo_depth_target_exact_brpo_upstream_target_v1"], frame_out / "pseudo_depth_target_exact_brpo_upstream_target_v1.png")
    _save_mask_png(result["pseudo_confidence_exact_brpo_upstream_target_v1"], frame_out / "pseudo_confidence_exact_brpo_upstream_target_v1.png")
    _save_source_map_png(result["pseudo_source_map_exact_brpo_upstream_target_v1"], frame_out / "pseudo_source_map_exact_brpo_upstream_target_v1.png")
    _save_mask_png(result["pseudo_valid_mask_exact_brpo_upstream_target_v1"], frame_out / "pseudo_valid_mask_exact_brpo_upstream_target_v1.png")
    _save_float_png(result["pseudo_target_confidence_exact_brpo_upstream_target_v1"], diag_dir / "pseudo_target_confidence_exact_brpo_upstream_target_v1.png", vmax=1.0)
    
    meta["observation_builder"] = "build_exact_brpo_upstream_target_observation"
    meta["summary"] = result["summary"]
    
    with open(frame_out / "exact_brpo_upstream_target_observation_meta_v1.json", "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)
