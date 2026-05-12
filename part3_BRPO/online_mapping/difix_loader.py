from __future__ import annotations

"""Lightweight Difix loader used by the online runtime bridge.

This module exists so the runtime no longer needs the historical
`scripts.legacy_prepare.prepare_stage1_difix_dataset_s3po` file at import time.
"""

from pathlib import Path
from typing import Optional
import sys


def load_difix_model(model_name: Optional[str], model_path: Optional[str], timestep: int, target_device=None):
    import torch

    if target_device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(target_device)

    if target_device.type == "cuda":
        torch.cuda.set_device(0 if target_device.index is None else target_device.index)
        _ = torch.empty(1, device=target_device)

    if model_name and not model_path:
        from diffusers import DiffusionPipeline

        custom_pipeline = "/home/bzhang512/CV_Project/third_party/Difix3D/src/pipeline_difix.py"
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            custom_pipeline=custom_pipeline,
            trust_remote_code=True,
        )
        pipe = pipe.to(target_device)

        vae = getattr(pipe, "vae", None)
        encoder = getattr(vae, "encoder", None)
        decoder = getattr(vae, "decoder", None)
        if encoder is not None and hasattr(encoder, "forward") and not hasattr(encoder, "my_vae_encoder_fwd"):
            if getattr(getattr(encoder, "forward", None), "__name__", "") == "my_vae_encoder_fwd":
                encoder.my_vae_encoder_fwd = encoder.forward
        if decoder is not None and hasattr(decoder, "forward") and not hasattr(decoder, "my_vae_decoder_fwd"):
            if getattr(getattr(decoder, "forward", None), "__name__", "") == "my_vae_decoder_fwd":
                decoder.my_vae_decoder_fwd = decoder.forward
        return {"kind": "hf_pipeline", "obj": pipe, "timestep": timestep, "device": str(target_device)}

    difix_src = Path("/home/bzhang512/CV_Project/third_party/Difix3D/src")
    if str(difix_src) not in sys.path:
        sys.path.insert(0, str(difix_src))
    from model import Difix

    model = Difix(
        pretrained_name=model_name,
        pretrained_path=model_path,
        timestep=timestep,
        mv_unet=True,
    )
    model.set_eval()
    if target_device.type == "cuda" and hasattr(model, "to"):
        model = model.to(target_device)
    return {"kind": "local_model", "obj": model, "timestep": timestep, "device": str(target_device)}


__all__ = ["load_difix_model"]
