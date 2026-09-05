from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PIL import Image
from torchvision import transforms


class _SizePreservingRandomResizedCrop:
    """Small random crop that keeps the original (W, H) after crop+resize."""

    def __init__(
        self,
        scale_min: float = 0.9,
        scale_max: float = 1.0,
        ratio_min: float = 0.95,
        ratio_max: float = 1.05,
    ) -> None:
        self.scale = (float(scale_min), float(scale_max))
        self.ratio = (float(ratio_min), float(ratio_max))

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        crop = transforms.RandomResizedCrop((h, w), scale=self.scale, ratio=self.ratio)
        return crop(img)


def build_training_image_augment(cfg: Optional[Dict[str, Any]]) -> Optional[Callable[[Image.Image], Image.Image]]:
    """Build PIL-level augment for training. Returns None when disabled (default).

    Config under ``framework.dino.image_augment``::

        image_augment:
          enabled: false
          brightness: 0.3
          contrast: 0.3
          saturation: 0.2
          hue: 0.05
          crop_scale_min: 0.9
          crop_scale_max: 1.0
    """
    if not cfg or not bool(cfg.get("enabled", False)):
        return None

    crop_scale_min = float(cfg.get("crop_scale_min", 0.9))
    crop_scale_max = float(cfg.get("crop_scale_max", 1.0))
    if crop_scale_min > crop_scale_max:
        crop_scale_min, crop_scale_max = crop_scale_max, crop_scale_min

    return transforms.Compose(
        [
            _SizePreservingRandomResizedCrop(
                scale_min=crop_scale_min,
                scale_max=crop_scale_max,
            ),
            transforms.ColorJitter(
                brightness=float(cfg.get("brightness", 0.3)),
                contrast=float(cfg.get("contrast", 0.3)),
                saturation=float(cfg.get("saturation", 0.2)),
                hue=float(cfg.get("hue", 0.05)),
            ),
        ]
    )


def apply_image_augment(
    batch_images: List[List[Any]],
    augment: Callable[[Image.Image], Image.Image],
) -> List[List[Any]]:
    """Apply augment to each view; supports PIL or uint8 ndarray (from dataset)."""
    from .vision_encoder import _ensure_pil_image

    out: List[List[Any]] = []
    for views in batch_images:
        out.append([augment(_ensure_pil_image(img)) for img in views])
    return out
