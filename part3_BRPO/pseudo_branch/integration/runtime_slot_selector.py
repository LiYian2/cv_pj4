from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RuntimeSlotSelectorConfig:
    placement_mode: str = "midpoint_only"
    max_pseudo_per_gap: int = 1


@dataclass(frozen=True)
class RuntimePseudoSlot:
    frame_id: int
    left_ref_frame_id: int
    right_ref_frame_id: int
    placement: str
    gap_index: int
    gap_key: str
    trigger_keyframe: int
    selection_source: str = "newly_closed_gap"

    def as_dict(self) -> dict:
        return {
            "frame_id": int(self.frame_id),
            "left_ref_frame_id": int(self.left_ref_frame_id),
            "right_ref_frame_id": int(self.right_ref_frame_id),
            "placement": self.placement,
            "gap_index": int(self.gap_index),
            "gap_key": self.gap_key,
            "trigger_keyframe": int(self.trigger_keyframe),
            "selection_source": self.selection_source,
        }


def _normalize_available_ids(available_frame_ids: Iterable[int] | None) -> list[int] | None:
    if available_frame_ids is None:
        return None
    uniq = sorted({int(x) for x in available_frame_ids})
    return uniq if uniq else None


def _pick_midpoint_candidate(left_kf: int, right_kf: int, available_frame_ids: list[int] | None) -> int | None:
    if right_kf - left_kf <= 1:
        return None
    if available_frame_ids is None:
        cand = int(round((int(left_kf) + int(right_kf)) / 2.0))
        if cand <= int(left_kf) or cand >= int(right_kf):
            return None
        return cand
    gap = [fid for fid in available_frame_ids if int(left_kf) < int(fid) < int(right_kf)]
    if not gap:
        return None
    target = (float(left_kf) + float(right_kf)) / 2.0
    return min(gap, key=lambda fid: abs(float(fid) - target))


def select_runtime_pseudo_slots(
    *,
    current_window: Sequence[int],
    trigger_keyframe: int,
    seen_gap_keys: set[str] | None = None,
    placement_mode: str = "midpoint_only",
    max_pseudo_per_gap: int = 1,
    available_frame_ids: Iterable[int] | None = None,
) -> list[RuntimePseudoSlot]:
    if placement_mode != "midpoint_only":
        raise ValueError(f"Unsupported placement_mode={placement_mode}")
    if int(max_pseudo_per_gap) != 1:
        raise ValueError(f"Phase-2 selector only supports max_pseudo_per_gap=1, got {max_pseudo_per_gap}")

    seen_gap_keys = seen_gap_keys or set()
    ordered_kfs = sorted({int(x) for x in current_window})
    available_ids = _normalize_available_ids(available_frame_ids)
    slots: list[RuntimePseudoSlot] = []
    for gap_index, (left_kf, right_kf) in enumerate(zip(ordered_kfs[:-1], ordered_kfs[1:])):
        gap_key = f"{int(left_kf)}-{int(right_kf)}"
        if gap_key in seen_gap_keys:
            continue
        frame_id = _pick_midpoint_candidate(left_kf, right_kf, available_ids)
        if frame_id is None:
            continue
        slots.append(
            RuntimePseudoSlot(
                frame_id=int(frame_id),
                left_ref_frame_id=int(left_kf),
                right_ref_frame_id=int(right_kf),
                placement="midpoint",
                gap_index=int(gap_index),
                gap_key=gap_key,
                trigger_keyframe=int(trigger_keyframe),
            )
        )
    return slots
