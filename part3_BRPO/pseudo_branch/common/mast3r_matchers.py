# -*- coding: utf-8 -*-
"""Reusable MASt3R matcher variants for future live-pipeline integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import sys

import numpy as np

S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from .flow_matcher import FlowMatcher
from .mast3r_pair_forward import DEFAULT_MODEL_NAME, MASt3RPairForward, get_shared_mast3r_pair_forward


@dataclass
class MatcherDiagnostics:
    matcher_mode: str
    candidate_ratio_query: float | None = None
    candidate_ratio_ref: float | None = None
    num_candidate_query: int | None = None
    num_candidate_ref: int | None = None
    num_reciprocal_matches: int | None = None
    conf_quantile_query: float | None = None
    conf_quantile_ref: float | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "matcher_mode": self.matcher_mode,
            "candidate_ratio_query": self.candidate_ratio_query,
            "candidate_ratio_ref": self.candidate_ratio_ref,
            "num_candidate_query": self.num_candidate_query,
            "num_candidate_ref": self.num_candidate_ref,
            "num_reciprocal_matches": self.num_reciprocal_matches,
            "conf_quantile_query": self.conf_quantile_query,
            "conf_quantile_ref": self.conf_quantile_ref,
        }


class BasePairMatcher:
    def __init__(self):
        self._last_match_meta: Dict[str, object] = {}

    def match_pair(self, img1_path: str, img2_path: str, size: int = 512):
        raise NotImplementedError

    def get_last_match_meta(self) -> Dict[str, object]:
        return dict(self._last_match_meta)


class Dense3DMatcher(BasePairMatcher):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        conf_mode: str = "quantile",
        conf_quantile: float = 0.90,
        use_shared_forwarder: bool = True,
        pair_forwarder: MASt3RPairForward | None = None,
    ):
        super().__init__()
        if conf_mode != "quantile":
            raise ValueError(f"Unsupported dense3d conf_mode={conf_mode}; only 'quantile' is implemented in step-1/2 landing")
        if not (0.0 < float(conf_quantile) < 1.0):
            raise ValueError(f"conf_quantile must be in (0,1), got {conf_quantile}")
        self.model_name = model_name
        self.device = device
        self.conf_mode = conf_mode
        self.conf_quantile = float(conf_quantile)
        self.forwarder = pair_forwarder or (
            get_shared_mast3r_pair_forward(model_name=model_name, device=device)
            if use_shared_forwarder
            else MASt3RPairForward(model_name=model_name, device=device)
        )

    @staticmethod
    def _valid_point_mask(conf_map: np.ndarray | None, pts3d: np.ndarray | None) -> np.ndarray:
        if conf_map is None or pts3d is None:
            raise ValueError("Dense3DMatcher requires MASt3R confidence maps and 3D pointmaps")
        return np.isfinite(conf_map) & np.isfinite(pts3d).all(axis=-1)

    def _build_candidate_mask(self, conf_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
        if not valid_mask.any():
            return np.zeros_like(valid_mask, dtype=bool), float("nan")
        thr = float(np.quantile(conf_map[valid_mask], self.conf_quantile))
        cand = valid_mask & (conf_map >= thr)
        return cand, thr

    def match_pair(self, img1_path: str, img2_path: str, size: int = 512):
        bundle = self.forwarder.run_pair(img1_path=img1_path, img2_path=img2_path, size=int(size))
        conf1 = np.asarray(bundle.conf1, dtype=np.float32)
        conf2 = np.asarray(bundle.conf2, dtype=np.float32)
        pts3d_1 = np.asarray(bundle.pts3d_1, dtype=np.float32)
        pts3d_2 = np.asarray(bundle.pts3d_2_in_1, dtype=np.float32)
        h, w = bundle.image_size

        valid1 = self._valid_point_mask(conf1, pts3d_1)
        valid2 = self._valid_point_mask(conf2, pts3d_2)
        cand1, thr1 = self._build_candidate_mask(conf1, valid1)
        cand2, thr2 = self._build_candidate_mask(conf2, valid2)

        if not cand1.any() or not cand2.any():
            self._last_match_meta = MatcherDiagnostics(
                matcher_mode="dense_pts3d_3d",
                candidate_ratio_query=float(cand1.mean()),
                candidate_ratio_ref=float(cand2.mean()),
                num_candidate_query=int(cand1.sum()),
                num_candidate_ref=int(cand2.sum()),
                num_reciprocal_matches=0,
                conf_quantile_query=thr1,
                conf_quantile_ref=thr2,
            ).as_dict()
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        xy = xy_grid(int(w), int(h)).reshape(-1, 2)
        xy1_sel = xy[cand1.reshape(-1)]
        xy2_sel = xy[cand2.reshape(-1)]
        pts3d_1_sel = pts3d_1.reshape(-1, 3)[cand1.reshape(-1)]
        pts3d_2_sel = pts3d_2.reshape(-1, 3)[cand2.reshape(-1)]
        conf1_sel = conf1.reshape(-1)[cand1.reshape(-1)]
        conf2_sel = conf2.reshape(-1)[cand2.reshape(-1)]

        reciprocal_in_p2, nn2_in_p1, num_matches = find_reciprocal_matches(pts3d_1_sel, pts3d_2_sel)
        if int(num_matches) <= 0:
            self._last_match_meta = MatcherDiagnostics(
                matcher_mode="dense_pts3d_3d",
                candidate_ratio_query=float(cand1.mean()),
                candidate_ratio_ref=float(cand2.mean()),
                num_candidate_query=int(cand1.sum()),
                num_candidate_ref=int(cand2.sum()),
                num_reciprocal_matches=0,
                conf_quantile_query=thr1,
                conf_quantile_ref=thr2,
            ).as_dict()
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        pts_query_xy = xy1_sel[nn2_in_p1][reciprocal_in_p2].astype(np.float32)
        pts_ref_xy = xy2_sel[reciprocal_in_p2].astype(np.float32)
        match_conf = np.sqrt(
            np.clip(conf1_sel[nn2_in_p1][reciprocal_in_p2] * conf2_sel[reciprocal_in_p2], 0.0, None)
        ).astype(np.float32)

        self._last_match_meta = MatcherDiagnostics(
            matcher_mode="dense_pts3d_3d",
            candidate_ratio_query=float(cand1.mean()),
            candidate_ratio_ref=float(cand2.mean()),
            num_candidate_query=int(cand1.sum()),
            num_candidate_ref=int(cand2.sum()),
            num_reciprocal_matches=int(len(match_conf)),
            conf_quantile_query=thr1,
            conf_quantile_ref=thr2,
        ).as_dict()
        self._last_match_meta.update(bundle.meta)
        return pts_query_xy, pts_ref_xy, match_conf


def build_pair_matcher(
    matcher_mode: str = "sparse_desc_2d",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    dense3d_conf_quantile: float = 0.90,
):
    mode = str(matcher_mode or "sparse_desc_2d").strip().lower()
    if mode == "sparse_desc_2d":
        return FlowMatcher(model_name=model_name, device=device)
    if mode == "dense_pts3d_3d":
        return Dense3DMatcher(
            model_name=model_name,
            device=device,
            conf_quantile=float(dense3d_conf_quantile),
        )
    raise ValueError(f"Unsupported matcher_mode={matcher_mode}")
