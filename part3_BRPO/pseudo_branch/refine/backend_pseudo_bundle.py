from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


@dataclass
class LoadedPseudoBundleSample:
    sample_id: int
    frame_id: int
    target_rgb: np.ndarray
    target_depth: np.ndarray
    confidence_mask: np.ndarray
    source_map: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    target_confidence: Optional[np.ndarray]
    support_both_mask: Optional[np.ndarray]


@dataclass
class PseudoBundleSample:
    sample_id: int
    frame_id: int
    target_rgb_path: Path
    target_depth_path: Path
    confidence_path: Path
    source_map_path: Optional[Path] = None
    valid_mask_path: Optional[Path] = None
    target_confidence_path: Optional[Path] = None
    support_both_path: Optional[Path] = None
    observation_meta_path: Optional[Path] = None
    stageA_scene_scale: Optional[float] = None
    view_state: Optional[dict[str, Any]] = None
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def _load_rgb(self) -> np.ndarray:
        arr = np.asarray(Image.open(self.target_rgb_path), dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0
        return arr.astype(np.float32)

    def _load_optional_npy(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        if not path.exists():
            return None
        return np.load(path).astype(np.float32)

    def load(self) -> LoadedPseudoBundleSample:
        return LoadedPseudoBundleSample(
            sample_id=int(self.sample_id),
            frame_id=int(self.frame_id),
            target_rgb=self._load_rgb(),
            target_depth=np.load(self.target_depth_path).astype(np.float32),
            confidence_mask=np.load(self.confidence_path).astype(np.float32),
            source_map=self._load_optional_npy(self.source_map_path),
            valid_mask=self._load_optional_npy(self.valid_mask_path),
            target_confidence=self._load_optional_npy(self.target_confidence_path),
            support_both_mask=self._load_optional_npy(self.support_both_path),
        )


@dataclass
class PseudoBundleBatch:
    source_history_json: Path
    stage_mode: str
    pseudo_observation_mode: Optional[str]
    samples: list[PseudoBundleSample]
    args: dict[str, Any] = field(default_factory=dict)
    effective_source_summary: dict[str, Any] = field(default_factory=dict)

    def sample_ids(self) -> list[int]:
        return [int(sample.sample_id) for sample in self.samples]



def _optional_path(value: Any) -> Optional[Path]:
    if value in (None, "", False):
        return None
    return Path(value)



def _require_path(path: Optional[Path], *, field_name: str, sample_id: int) -> Path:
    if path is None:
        raise ValueError(f"sample_id={sample_id} missing required field {field_name}")
    if not path.exists():
        raise FileNotFoundError(f"sample_id={sample_id} {field_name} not found: {path}")
    return path



def _state_by_sample_id(items: Any) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        sid = row.get("sample_id")
        if sid is None:
            continue
        out[int(sid)] = row
    return out



def load_pseudo_bundle_from_stageA_history(
    stageA_history_json: str | Path,
    *,
    require_exact_upstream: bool = False,
) -> PseudoBundleBatch:
    history_path = Path(stageA_history_json)
    data = json.loads(history_path.read_text())
    sample_rows = data.get("pseudo_sample_meta", [])
    if not isinstance(sample_rows, list) or not sample_rows:
        raise ValueError(f"No pseudo_sample_meta rows found in {history_path}")

    final_state_by_id = _state_by_sample_id(data.get("final_states", []))
    stage_mode = str(data.get("stage_mode", ""))
    source_summary = data.get("effective_source_summary", {}) or {}
    pseudo_observation_mode = source_summary.get("pseudo_observation_mode")

    samples: list[PseudoBundleSample] = []
    for row in sample_rows:
        sample_id = int(row["sample_id"])
        frame_id = int(row.get("frame_id", sample_id))
        effective_mode = row.get("pseudo_observation_mode_effective")
        if require_exact_upstream and effective_mode != "exact_brpo_upstream_target_v1":
            raise ValueError(
                f"sample_id={sample_id} expected exact_brpo_upstream_target_v1, got {effective_mode}"
            )

        confidence_path = _require_path(
            _optional_path(row.get("confidence_path")),
            field_name="confidence_path",
            sample_id=sample_id,
        )
        target_rgb_path = _require_path(
            _optional_path(row.get("target_rgb_path")),
            field_name="target_rgb_path",
            sample_id=sample_id,
        )
        target_depth_path = _require_path(
            _optional_path(row.get("target_depth_for_refine_path")),
            field_name="target_depth_for_refine_path",
            sample_id=sample_id,
        )
        source_map_path = _optional_path(row.get("target_depth_for_refine_source_map_path"))
        observation_meta_path = _optional_path(row.get("pseudo_observation_meta_path"))

        valid_mask_path = None
        target_confidence_path = None
        support_both_path = None
        if observation_meta_path is not None and observation_meta_path.exists():
            frame_dir = observation_meta_path.parent
            if effective_mode:
                valid_mask_candidate = frame_dir / f"pseudo_valid_mask_{effective_mode}.npy"
                target_conf_candidate = frame_dir / f"pseudo_target_confidence_{effective_mode}.npy"
                support_both_candidate = frame_dir / 'diag' / f"pseudo_verify_both_{effective_mode}.npy"
                if valid_mask_candidate.exists():
                    valid_mask_path = valid_mask_candidate
                if target_conf_candidate.exists():
                    target_confidence_path = target_conf_candidate
                if support_both_candidate.exists():
                    support_both_path = support_both_candidate

        samples.append(
            PseudoBundleSample(
                sample_id=sample_id,
                frame_id=frame_id,
                target_rgb_path=target_rgb_path,
                target_depth_path=target_depth_path,
                confidence_path=confidence_path,
                source_map_path=source_map_path if source_map_path is not None and source_map_path.exists() else None,
                valid_mask_path=valid_mask_path,
                target_confidence_path=target_confidence_path,
                support_both_path=support_both_path,
                observation_meta_path=observation_meta_path if observation_meta_path is not None and observation_meta_path.exists() else None,
                stageA_scene_scale=row.get("stageA_scene_scale"),
                view_state=final_state_by_id.get(sample_id),
                extra_meta=dict(row),
            )
        )

    return PseudoBundleBatch(
        source_history_json=history_path,
        stage_mode=stage_mode,
        pseudo_observation_mode=(str(pseudo_observation_mode) if pseudo_observation_mode is not None else None),
        samples=samples,
        args=data.get("args", {}) or {},
        effective_source_summary=source_summary,
    )
