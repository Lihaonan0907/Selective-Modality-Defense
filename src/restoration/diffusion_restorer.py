"""Stable Diffusion inpainting based modality-specific restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from src.utils.common import require_path


class ModalityRestorer:
    """Restore corrupted regions for one modality using inpainting."""

    def __init__(self, cfg: dict[str, Any], modality: str):
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.cfg = cfg
        self.modality = modality
        self.device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.steps = int(cfg.get("steps", 40))
        self.guidance_scale = float(cfg.get("guidance_scale", 7.5))
        self.strength = float(cfg.get("strength", 0.85))

        base_model = require_path(cfg.get("stable_diffusion_inpaint"), "stable diffusion inpaint model")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(base_model),
            torch_dtype=self.dtype,
            safety_checker=None,
            local_files_only=bool(cfg.get("local_files_only", True)),
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        ckpt = cfg.get("checkpoint")
        if ckpt:
            self._load_unet_checkpoint(Path(ckpt))

    def _load_unet_checkpoint(self, ckpt_path: Path) -> None:
        """Load a modality-specific UNet checkpoint if provided."""
        ckpt_path = require_path(ckpt_path, f"{self.modality} restoration checkpoint")
        state = torch.load(ckpt_path, map_location=self.device)
        unet_state = state.get("unet_state_dict", state)
        current = self.pipe.unet.state_dict()
        filtered = {k: v for k, v in unet_state.items() if k in current}
        self.pipe.unet.load_state_dict(filtered, strict=False)
        self.pipe.unet.to(dtype=self.dtype)

    def prompts(self) -> tuple[str, str]:
        """Return positive and negative prompts for this modality."""
        if self.modality == "ir":
            return (
                self.cfg.get("infrared_prompt", "smooth thermal infrared imaging, grayscale, pedestrians"),
                self.cfg.get("infrared_negative_prompt", "colorful, glowing, deformed, noisy, patches"),
            )
        return (
            self.cfg.get("visible_prompt", "a night street scene, asphalt road, pedestrians, realistic"),
            self.cfg.get("visible_negative_prompt", "colorful blocks, glowing artifacts, deformed, noisy"),
        )

    def restore(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Restore corrupted region. If mask is empty, return the original image."""
        if mask is None or cv2.countNonZero(mask) == 0:
            return image.copy()

        prompt, negative_prompt = self.prompts()
        inpaint_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        base_prefilled = cv2.inpaint(image, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        mask_for_sd = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)

        with torch.inference_mode():
            result_img = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.fromarray(base_prefilled),
                mask_image=Image.fromarray(mask_for_sd),
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                strength=self.strength,
            ).images[0]

        restored = np.array(result_img.resize((image.shape[1], image.shape[0])))
        mask_blend = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
        mask_float = cv2.GaussianBlur(mask_blend.astype(np.float32) / 255.0, (21, 21), 0)[..., None]
        mask_float[inpaint_mask > 0] = 1.0
        return (base_prefilled * (1 - mask_float) + restored * mask_float).astype(np.uint8)

