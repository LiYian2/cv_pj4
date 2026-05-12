from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC rgb array, got shape={arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def tensor_chw_to_hwc_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return arr.astype(np.float32)


def save_float_png(path: Path, value: np.ndarray, scale: float | None = None) -> None:
    ensure_dir(path.parent)
    arr = np.asarray(value, dtype=np.float32)
    if scale is None:
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        scale = vmax if vmax > 1e-8 else 1.0
    vis = np.clip(arr / float(scale), 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(path)


def write_runtime_exact_backend_frame(
    *,
    exact_frame_out: Path,
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    exact_meta: dict[str, Any],
    left_ref_depth: np.ndarray,
    right_ref_depth: np.ndarray,
) -> None:
    ensure_dir(exact_frame_out)
    diag_dir = ensure_dir(exact_frame_out / "diag")
    np.save(exact_frame_out / "support_left_exact.npy", left_result["support_mask"])
    np.save(exact_frame_out / "support_right_exact.npy", right_result["support_mask"])
    np.save(exact_frame_out / "projected_depth_left_exact.npy", left_result["projected_depth_map"])
    np.save(exact_frame_out / "projected_depth_right_exact.npy", right_result["projected_depth_map"])
    np.save(exact_frame_out / "projected_valid_left_exact.npy", left_result["projected_depth_valid_mask"])
    np.save(exact_frame_out / "projected_valid_right_exact.npy", right_result["projected_depth_valid_mask"])
    np.save(exact_frame_out / "provenance_left.npy", left_result["provenance_map"])
    np.save(exact_frame_out / "provenance_right.npy", right_result["provenance_map"])
    np.save(exact_frame_out / "hit_count_left.npy", left_result["hit_count"])
    np.save(exact_frame_out / "hit_count_right.npy", right_result["hit_count"])
    np.save(exact_frame_out / "occlusion_reason_left.npy", left_result["occlusion_reason_map"])
    np.save(exact_frame_out / "occlusion_reason_right.npy", right_result["occlusion_reason_map"])
    np.save(exact_frame_out / "confidence_left_exact.npy", left_result["confidence_map"])
    np.save(exact_frame_out / "confidence_right_exact.npy", right_result["confidence_map"])
    np.save(exact_frame_out / "depth_variance_left.npy", left_result["depth_variance_map"])
    np.save(exact_frame_out / "depth_variance_right.npy", right_result["depth_variance_map"])
    np.save(exact_frame_out / "ref_depth_left_render.npy", left_ref_depth.astype(np.float32))
    np.save(exact_frame_out / "ref_depth_right_render.npy", right_ref_depth.astype(np.float32))
    write_json(exact_frame_out / "exact_backend_meta.json", exact_meta)

    save_float_png(diag_dir / "support_left_exact.png", left_result["support_mask"], scale=1.0)
    save_float_png(diag_dir / "support_right_exact.png", right_result["support_mask"], scale=1.0)
    save_float_png(diag_dir / "projected_depth_left_exact.png", left_result["projected_depth_map"])
    save_float_png(diag_dir / "projected_depth_right_exact.png", right_result["projected_depth_map"])
    save_float_png(diag_dir / "confidence_left_exact.png", left_result["confidence_map"], scale=1.0)
    save_float_png(diag_dir / "confidence_right_exact.png", right_result["confidence_map"], scale=1.0)


def write_runtime_pseudo_record_frame(
    *,
    record_frame_out: Path,
    target_rgb: np.ndarray,
    target_depth: np.ndarray,
    confidence_mask: np.ndarray,
    source_map: np.ndarray,
    valid_mask: np.ndarray,
    target_confidence: np.ndarray,
    support_both_mask: np.ndarray,
    camera_json: dict[str, Any],
    record_meta: dict[str, Any],
) -> None:
    ensure_dir(record_frame_out)
    save_rgb_png(record_frame_out / "target_rgb_runtime.png", target_rgb)
    np.save(record_frame_out / "target_depth_runtime.npy", target_depth.astype(np.float32))
    np.save(record_frame_out / "confidence_mask_runtime.npy", confidence_mask.astype(np.float32))
    np.save(record_frame_out / "source_map_runtime.npy", source_map.astype(np.int16))
    np.save(record_frame_out / "valid_mask_runtime.npy", valid_mask.astype(np.float32))
    np.save(record_frame_out / "target_confidence_runtime.npy", target_confidence.astype(np.float32))
    np.save(record_frame_out / "support_both_runtime.npy", support_both_mask.astype(np.float32))
    write_json(record_frame_out / "camera.json", camera_json)
    write_json(record_frame_out / "record_meta.json", record_meta)
