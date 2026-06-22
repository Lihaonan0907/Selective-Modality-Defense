"""Self-contained modality-specific restoration fine-tuning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from src.restoration.discriminator import MaskedPatchGAN
from src.restoration.restoration_losses import EdgeAwareTVLoss, FrequencyLoss
from src.utils.common import ensure_dir, require_path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _find_image(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


class RestorationFineTuneDataset(Dataset):
    """Dataset for mask-conditioned restoration fine-tuning."""

    def __init__(self, data_root: str | Path, modality: str, split: str = "train", resolution: int = 512):
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.data_root = Path(data_root)
        self.modality = modality
        self.split = split
        self.resolution = resolution
        self.image_tf = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        self.samples = self._discover_samples()

    def _discover_samples(self) -> list[dict[str, str]]:
        branch = "vis" if self.modality == "vis" else "ir"
        full_branch = "visible" if self.modality == "vis" else "infrared"
        categories = ["vis_only", "cross"] if self.modality == "vis" else ["ir_only", "cross"]
        samples: list[dict[str, str]] = []

        for category in categories:
            attack_dir = self.data_root / "data" / self.split / category / branch
            mask_dir = self.data_root / "data" / self.split / category / f"masks_{branch}"
            clean_dir = self.data_root / full_branch / "images" / f"clean_{self.split}"
            if not attack_dir.exists() or not mask_dir.exists() or not clean_dir.exists():
                continue
            for attack_path in sorted(attack_dir.iterdir()):
                if attack_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                clean_path = _find_image(clean_dir, attack_path.stem)
                mask_path = _find_image(mask_dir, attack_path.stem)
                if clean_path is not None and mask_path is not None:
                    samples.append({"attack": str(attack_path), "clean": str(clean_path), "mask": str(mask_path)})

        simple_root = self.data_root / full_branch
        attack_dir = simple_root / "attacked" / self.split
        clean_dir = simple_root / "clean" / self.split
        mask_dir = simple_root / "masks" / self.split
        if attack_dir.exists() and clean_dir.exists() and mask_dir.exists():
            for attack_path in sorted(attack_dir.iterdir()):
                if attack_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                clean_path = _find_image(clean_dir, attack_path.stem)
                mask_path = _find_image(mask_dir, attack_path.stem)
                if clean_path is not None and mask_path is not None:
                    samples.append({"attack": str(attack_path), "clean": str(clean_path), "mask": str(mask_path)})
        return samples

    def _robust_mask(self, mask: Image.Image) -> torch.Tensor:
        mask_np = np.array(mask.resize((self.resolution, self.resolution), Image.NEAREST))
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        choice = np.random.random()
        if choice < 0.30:
            kernel = np.ones((15, 15), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        elif choice < 0.40:
            kernel = np.ones((5, 5), np.uint8)
            mask_np = cv2.erode(mask_np, kernel, iterations=1)
        elif self.modality == "ir" and choice > 0.85:
            num_spots = np.random.randint(10, 40)
            for _ in range(num_spots):
                width = int(np.random.randint(4, 10))
                x = int(np.random.randint(0, max(1, self.resolution - width)))
                y = int(np.random.randint(0, max(1, self.resolution - width)))
                mask_np[y : y + width, x : x + width] = 255
        return torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        attack = Image.open(sample["attack"]).convert("RGB")
        clean = Image.open(sample["clean"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        mask_tensor = self._robust_mask(mask) if self.split == "train" else self.mask_tf(mask)
        return {"attack": self.image_tf(attack), "clean": self.image_tf(clean), "mask": mask_tensor}


class RestorationFineTuner:
    """Fine-tune Stable Diffusion inpainting UNet for one modality."""

    def __init__(self, cfg: dict[str, Any], modality: str):
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.cfg = cfg
        self.modality = modality
        device_name = cfg.get("device", "cuda:0")
        self.device = torch.device(device_name if torch.cuda.is_available() or str(device_name).startswith("cpu") else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_path = require_path(cfg.get("stable_diffusion_inpaint"), "stable diffusion inpainting base model")
        self.vae = AutoencoderKL.from_pretrained(str(model_path), subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(str(model_path), subfolder="unet").to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(str(model_path), subfolder="scheduler")
        for param in self.vae.parameters():
            param.requires_grad = False
        if bool(cfg.get("gradient_checkpointing", True)):
            self.unet.enable_gradient_checkpointing()

        self.discriminator = MaskedPatchGAN().to(self.device)
        self.tv_loss = EdgeAwareTVLoss()
        self.freq_loss = FrequencyLoss()
        lr = float(cfg.get("lr", 1e-5))
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 1e-4)))
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 1e-4)))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def train_step(self, batch: dict[str, torch.Tensor], epoch: int) -> dict[str, float]:
        attack = batch["attack"].to(self.device)
        clean = batch["clean"].to(self.device)
        mask = batch["mask"].to(self.device)
        masked_attack = attack * (1.0 - mask)
        batch_size = attack.shape[0]

        self.unet.train()
        self.discriminator.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            with torch.no_grad():
                clean_latent = self.vae.encode(clean).latent_dist.sample() * self.vae.config.scaling_factor
                masked_latent = self.vae.encode(masked_attack).latent_dist.sample() * self.vae.config.scaling_factor
                mask_latent = F.interpolate(mask, size=clean_latent.shape[-2:], mode="nearest")
            noise = torch.randn_like(clean_latent)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
            noisy_latent = self.noise_scheduler.add_noise(clean_latent, noise, timesteps)
            latent_input = torch.cat([noisy_latent, mask_latent, masked_latent], dim=1)
            encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=self.device, dtype=latent_input.dtype)
            noise_pred = self.unet(latent_input, timesteps, encoder_hidden_states).sample
            loss_diffusion = F.mse_loss(noise_pred.float(), noise.float())

            alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].to(self.device).view(-1, 1, 1, 1)
            x0_latent = (noisy_latent - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            pred = self.vae.decode(x0_latent / self.vae.config.scaling_factor).sample
            loss_l1_global = F.l1_loss(pred, clean)
            loss_l1_local = F.l1_loss(pred * mask, clean * mask)
            loss_frequency = self.freq_loss(pred * mask, clean * mask)
            loss_tv = self.tv_loss(pred * mask) * float(self.cfg.get("lambda_tv", 0.005))

            heat_loss = pred.new_tensor(0.0)
            if self.modality == "ir":
                erasure = torch.clamp(clean - pred, min=0.0) * mask
                heat_loss = erasure.mean() * float(self.cfg.get("lambda_ir_heat", 20.0))

            gan_loss = pred.new_tensor(0.0)
            discriminator_loss_value = 0.0
            if epoch > int(self.cfg.get("gan_start_epoch", 10)):
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = self.discriminator(clean.detach(), mask)
                fake_logits = self.discriminator(pred.detach(), mask)
                discriminator_loss = (
                    F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                    + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                ) * 0.5
                self.scaler.scale(discriminator_loss).backward()
                self.scaler.step(self.discriminator_optimizer)
                discriminator_loss_value = float(discriminator_loss.detach().cpu())
                fake_logits_for_g = self.discriminator(pred, mask)
                gan_loss = F.binary_cross_entropy_with_logits(fake_logits_for_g, torch.ones_like(fake_logits_for_g))

            total = (
                loss_diffusion
                + float(self.cfg.get("lambda_l1_global", 1.0)) * loss_l1_global
                + float(self.cfg.get("lambda_l1_local", 10.0)) * loss_l1_local
                + float(self.cfg.get("lambda_frequency", 0.01)) * loss_frequency
                + float(self.cfg.get("lambda_gan", 0.1)) * gan_loss
                + loss_tv
                + heat_loss
            )

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), float(self.cfg.get("grad_clip", 1.0)))
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss_total": float(total.detach().cpu()),
            "loss_diffusion": float(loss_diffusion.detach().cpu()),
            "loss_l1_global": float(loss_l1_global.detach().cpu()),
            "loss_l1_local": float(loss_l1_local.detach().cpu()),
            "loss_frequency": float(loss_frequency.detach().cpu()),
            "loss_heat": float(heat_loss.detach().cpu()),
            "loss_gan": float(gan_loss.detach().cpu()),
            "loss_discriminator": discriminator_loss_value,
        }

    def fit(self, loader: Any, epochs: int, output_dir: str | Path) -> Path:
        output_dir = ensure_dir(output_dir)
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            started = time.time()
            running: dict[str, float] = {}
            for batch in loader:
                metrics = self.train_step(batch, epoch)
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value
            count = max(1, len(loader))
            avg = {key: value / count for key, value in running.items()}
            avg["epoch"] = float(epoch)
            avg["seconds"] = time.time() - started
            print(avg)
            latest = output_dir / f"latest_{self.modality}.pt"
            torch.save({"unet_state_dict": self.unet.state_dict(), "modality": self.modality, "metrics": avg}, latest)
            if avg.get("loss_total", float("inf")) < best_loss:
                best_loss = avg["loss_total"]
                torch.save({"unet_state_dict": self.unet.state_dict(), "modality": self.modality, "metrics": avg}, output_dir / f"best_{self.modality}.pt")
        return output_dir
