# -*- coding: utf-8 -*-
"""Shared MASt3R pair forward helper for reusable matcher backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import sys

import numpy as np
import torch

S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
if S3PO_ROOT not in sys.path:
    sys.path.insert(0, S3PO_ROOT)

import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images


DEFAULT_MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


@dataclass
class MASt3RPairBundle:
    desc1: np.ndarray | None
    desc2: np.ndarray | None
    desc_conf1: np.ndarray | None
    desc_conf2: np.ndarray | None
    pts3d_1: np.ndarray | None
    pts3d_2_in_1: np.ndarray | None
    conf1: np.ndarray | None
    conf2: np.ndarray | None
    image_size: Tuple[int, int]
    meta: Dict[str, object]


class MASt3RPairForward:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        use_pair_cache: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_pair_cache = bool(use_pair_cache)
        self._model = None
        self._pair_cache: Dict[Tuple[str, str, int], MASt3RPairBundle] = {}

    def _ensure_model(self):
        if self._model is None:
            print(f"Loading MASt3R model for shared pair forward: {self.model_name}")
            self._model = AsymmetricMASt3R.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            print("Shared MASt3R pair forward model loaded.")
        return self._model

    @staticmethod
    def _to_numpy_map(pred: dict, key: str):
        if key not in pred:
            return None
        value = pred[key].squeeze(0).detach().cpu().numpy().astype(np.float32)
        return value

    def run_pair(self, img1_path: str, img2_path: str, size: int = 512) -> MASt3RPairBundle:
        cache_key = (str(img1_path), str(img2_path), int(size))
        if self.use_pair_cache and cache_key in self._pair_cache:
            return self._pair_cache[cache_key]

        model = self._ensure_model()
        images = load_images([img1_path, img2_path], size=int(size), square_ok=True, verbose=False)
        with torch.no_grad():
            output = inference([tuple(images)], model, self.device, batch_size=1, verbose=False)

        pred1, pred2 = output["pred1"], output["pred2"]
        desc_conf1 = self._to_numpy_map(pred1, "desc_conf")
        h, w = desc_conf1.shape if desc_conf1 is not None else self._to_numpy_map(pred1, "conf").shape
        bundle = MASt3RPairBundle(
            desc1=self._to_numpy_map(pred1, "desc"),
            desc2=self._to_numpy_map(pred2, "desc"),
            desc_conf1=desc_conf1,
            desc_conf2=self._to_numpy_map(pred2, "desc_conf"),
            pts3d_1=self._to_numpy_map(pred1, "pts3d"),
            pts3d_2_in_1=self._to_numpy_map(pred2, "pts3d_in_other_view"),
            conf1=self._to_numpy_map(pred1, "conf"),
            conf2=self._to_numpy_map(pred2, "conf"),
            image_size=(int(h), int(w)),
            meta={
                "img1_path": str(img1_path),
                "img2_path": str(img2_path),
                "size": int(size),
                "model_name": self.model_name,
                "device": self.device,
                "cache_hit": False,
            },
        )
        if self.use_pair_cache:
            self._pair_cache[cache_key] = bundle
        return bundle


_SHARED_FORWARDERS: Dict[Tuple[str, str], MASt3RPairForward] = {}


def get_shared_mast3r_pair_forward(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    use_pair_cache: bool = True,
) -> MASt3RPairForward:
    key = (str(model_name), str(device))
    if key not in _SHARED_FORWARDERS:
        _SHARED_FORWARDERS[key] = MASt3RPairForward(
            model_name=model_name,
            device=device,
            use_pair_cache=use_pair_cache,
        )
    return _SHARED_FORWARDERS[key]
