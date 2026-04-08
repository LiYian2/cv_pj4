# -*- coding: utf-8 -*-
"""Flow/correspondence matching via MASt3R + DUSt3R."""
import os
import sys
import numpy as np
import torch

# Path setup
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
sys.path.insert(0, S3PO_ROOT)

import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images


class FlowMatcher:
    def __init__(
        self,
        model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        device: str = "cuda",
    ):
        self.device = device
        print("Loading MASt3R model...")
        self.model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        self.model.eval()
        print("MASt3R model loaded.")

    def match_pair(self, img1_path: str, img2_path: str, size: int = 512):
        """Return reciprocal matches (pts1, pts2, conf)."""
        images = load_images([img1_path, img2_path], size=size, square_ok=True, verbose=False)

        with torch.no_grad():
            output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        pred1, pred2 = output["pred1"], output["pred2"]
        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=8,
            device=self.device,
            dist="dot",
            block_size=2**13,
        )

        if matches_im0 is None or len(matches_im0) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        matches_im0 = np.asarray(matches_im0, dtype=np.float32)
        matches_im1 = np.asarray(matches_im1, dtype=np.float32)

        # confidence from descriptor confidence map at matched pixels
        conf_map = pred1["desc_conf"].squeeze(0).detach().cpu().numpy()  # (H, W)
        H, W = conf_map.shape
        xi = np.clip(np.round(matches_im0[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(matches_im0[:, 1]).astype(int), 0, H - 1)
        conf = conf_map[yi, xi].astype(np.float32)

        return matches_im0, matches_im1, conf
