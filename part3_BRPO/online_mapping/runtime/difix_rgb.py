from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def run_single_difix_pil(model_bundle, image, ref_image, prompt, height, width):
    """Difix restoration for single branch (PIL input)."""
    if model_bundle is None:
        return image
    if model_bundle["kind"] == "hf_pipeline":
        pipe = model_bundle["obj"]
        # Keep Difix execution on the dynamically selected backend device.
        # If launched with CUDA_VISIBLE_DEVICES=1, model_bundle["device"] == "cuda:0"
        # still maps to physical GPU 1, not physical GPU 0.
        import torch
        target_device = torch.device(model_bundle.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        if target_device.type == "cuda":
            torch.cuda.set_device(0 if target_device.index is None else target_device.index)
        pipe = pipe.to(target_device)
        out = pipe(
            prompt,
            image=image,
            ref_image=ref_image,
            height=height,
            width=width,
            num_inference_steps=1,
            timesteps=[model_bundle["timestep"]],
            guidance_scale=0.0,
        ).images[0]
    else:
        model = model_bundle["obj"]
        out = model.sample(image=image, ref_image=ref_image, prompt=prompt, height=height, width=width)
    if out.size != image.size:
        out = out.resize(image.size, Image.LANCZOS)
    return out


def run_difix_restoration(
    model_bundle,
    pseudo_rgb: np.ndarray,
    left_ref_rgb: np.ndarray,
    right_ref_rgb: np.ndarray,
    cfg: Any,
) -> tuple:
    """Execute bidirectional Difix restoration.
    
    Args:
        model_bundle: Difix model bundle (None if disabled)
        pseudo_rgb: Coarse render RGB (H, W, 3) uint8
        left_ref_rgb: Left reference RGB (H, W, 3) uint8
        right_ref_rgb: Right reference RGB (H, W, 3) uint8
        cfg: Config with prompt, height, width
    
    Returns:
        left_fixed_rgb: Left-branch restored RGB (H, W, 3) uint8
        right_fixed_rgb: Right-branch restored RGB (H, W, 3) uint8
    """
    if model_bundle is None:
        return pseudo_rgb, pseudo_rgb
    
    pseudo_img = Image.fromarray(pseudo_rgb.astype(np.uint8))
    left_ref_img = Image.fromarray(left_ref_rgb.astype(np.uint8))
    right_ref_img = Image.fromarray(right_ref_rgb.astype(np.uint8))
    
    left_fixed = run_single_difix_pil(
        model_bundle=model_bundle,
        image=pseudo_img,
        ref_image=left_ref_img,
        prompt=str(cfg.difix_prompt or ""),
        height=int(cfg.difix_height or 512),
        width=int(cfg.difix_width or 512),
    )
    right_fixed = run_single_difix_pil(
        model_bundle=model_bundle,
        image=pseudo_img,
        ref_image=right_ref_img,
        prompt=str(cfg.difix_prompt or ""),
        height=int(cfg.difix_height or 512),
        width=int(cfg.difix_width or 512),
    )
    
    return np.array(left_fixed), np.array(right_fixed)
