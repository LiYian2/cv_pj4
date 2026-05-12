from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List


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


def _normalize_available_ids(available_frame_ids: Iterable[int] | None) -> List[int] | None:
    if available_frame_ids is None:
        return None
    uniq = sorted({int(x) for x in available_frame_ids})
    return uniq if uniq else None


def _pick_candidate_at_ratio(left_kf: int, right_kf: int, ratio: float, available_frame_ids: List[int] | None) -> int | None:
    """Pick a candidate frame at a given ratio between left and right keyframes.
    
    ratio=0.5 -> midpoint, ratio=0.25 -> quartile at 1/4, etc.
    """
    if right_kf - left_kf <= 1:
        return None
    target = float(left_kf) + ratio * float(right_kf - left_kf)
    if available_frame_ids is None:
        cand = int(round(target))
        if cand <= int(left_kf) or cand >= int(right_kf):
            return None
        return cand
    gap = [fid for fid in available_frame_ids if int(left_kf) < int(fid) < int(right_kf)]
    if not gap:
        return None
    return min(gap, key=lambda fid: abs(float(fid) - target))


def _get_placement_ratios(placement_mode: str, max_pseudo_per_gap: int) -> List[float]:
    """Get the ratio positions for each pseudo based on placement mode.
    
    Returns list of ratios (0.0-1.0) for each pseudo position.
    """
    if placement_mode == "midpoint_only":
        return [0.5]
    elif placement_mode == "quartile":
        # 3 positions: 1/4, 1/2, 3/4
        return [0.25, 0.5, 0.75]
    elif placement_mode == "quintile":
        # 5 positions: 1/5, 2/5, 3/5, 4/5
        return [0.2, 0.4, 0.6, 0.8]
    elif placement_mode == "uniform":
        # Uniform distribution based on max_pseudo_per_gap
        if max_pseudo_per_gap <= 0:
            return []
        step = 1.0 / (max_pseudo_per_gap + 1)
        return [step * i for i in range(1, max_pseudo_per_gap + 1)]
    else:
        raise ValueError(f"Unknown placement_mode={placement_mode}")


def select_runtime_pseudo_slots(
    *,
    current_window: Sequence[int],
    trigger_keyframe: int,
    seen_gap_keys: set[str] | None = None,
    placement_mode: str = "midpoint_only",
    max_pseudo_per_gap: int = 1,
    available_frame_ids: Iterable[int] | None = None,
) -> List[RuntimePseudoSlot]:
    """Select pseudo slots based on placement mode.
    
    Supported placement modes:
    - midpoint_only: 1 pseudo at midpoint (legacy)
    - quartile: 3 pseudos at 1/4, 1/2, 3/4 positions
    - quintile: 4 pseudos at 1/5, 2/5, 3/5, 4/5 positions (no midpoint)
    - uniform: evenly distributed based on max_pseudo_per_gap
    
    For quartile/quintile modes, max_pseudo_per_gap is ignored (mode defines count).
    """
    seen_gap_keys = seen_gap_keys or set()
    ordered_kfs = sorted({int(x) for x in current_window})
    available_ids = _normalize_available_ids(available_frame_ids)
    
    ratios = _get_placement_ratios(placement_mode, max_pseudo_per_gap)
    if not ratios:
        return []
    
    slots: List[RuntimePseudoSlot] = []
    for gap_index, (left_kf, right_kf) in enumerate(zip(ordered_kfs[:-1], ordered_kfs[1:])):
        gap_key = f"{int(left_kf)}-{int(right_kf)}"
        if gap_key in seen_gap_keys:
            continue
        
        for i, ratio in enumerate(ratios):
            frame_id = _pick_candidate_at_ratio(left_kf, right_kf, ratio, available_ids)
            if frame_id is None:
                continue
            
            # Determine placement label
            if placement_mode == "midpoint_only":
                placement = "midpoint"
            elif placement_mode == "quartile":
                placement = f"quartile_{i+1}"  # quartile_1, quartile_2, quartile_3
            elif placement_mode == "quintile":
                placement = f"quintile_{i+1}"  # quintile_1, quintile_2, quintile_3, quintile_4
            else:
                placement = f"uniform_{i+1}"
            
            slots.append(
                RuntimePseudoSlot(
                    frame_id=int(frame_id),
                    left_ref_frame_id=int(left_kf),
                    right_ref_frame_id=int(right_kf),
                    placement=placement,
                    gap_index=int(gap_index),
                    gap_key=gap_key,
                    trigger_keyframe=int(trigger_keyframe),
                )
            )
    
    return slots
