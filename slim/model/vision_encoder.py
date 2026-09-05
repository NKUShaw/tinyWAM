from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


@dataclass
class VisionEncoderOutput:
    cls_tokens: torch.Tensor
    patch_tokens: torch.Tensor


def _ensure_pil_image(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type for vision encoder: {type(img)}")


def _flatten_and_preprocess(
    images: List[List[Image.Image | np.ndarray]],
    preprocess: transforms.Compose,
    ref_param: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    flattened = [img for views in images for img in views]
    num_views = len(images[0]) if images and images[0] else 1
    x = torch.stack([preprocess(_ensure_pil_image(img)) for img in flattened], dim=0)
    x = x.to(device=ref_param.device, dtype=ref_param.dtype)
    return x, len(images), num_views


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder with a stable output interface."""

    _HIDDEN_SIZE: Dict[str, int] = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1408,
    }

    def __init__(
        self,
        backbone_name: str,
        model_path: str | None = None,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.model_path = model_path
        self.image_size_override = image_size
        if backbone_name not in self._HIDDEN_SIZE:
            raise NotImplementedError(f"Unsupported DINOv2 backbone: {backbone_name}")

        self._hidden_size = self._HIDDEN_SIZE[backbone_name]
        self.model = self._load_model(backbone_name=backbone_name, model_path=model_path)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(self._resolve_input_size()),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, backbone_name: str, model_path: str | None) -> nn.Module:
        # Prefer a FULLY-LOCAL load (source="local") below; only fall back to the remote
        # github source if a local load is unavailable. Resolve the hub dir via
        # torch.hub.get_dir() so it honours TORCH_HOME / XDG_CACHE_HOME. This avoids any
        # network dependency on the DINOv2 architecture code when it is cached locally.
        hub_error = None
        local_code_path = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        weight_candidates: List[Path] = []
        if model_path:
            mp = Path(model_path)
            if mp.is_file():
                weight_candidates.append(mp)
            elif mp.is_dir():
                weight_candidates.extend(
                    [
                        mp / f"{backbone_name}_pretrain.pth",
                        mp / f"{backbone_name}.pth",
                    ]
                )
                weight_candidates.extend(sorted(mp.glob("*.pth")))

        existing_weights = []
        seen_weights = set()
        for weight_path in weight_candidates:
            if weight_path.exists() and weight_path not in seen_weights:
                existing_weights.append(weight_path)
                seen_weights.add(weight_path)

        load_errors = []
        for weight_path in existing_weights:
            try:
                if not local_code_path.exists():
                    raise FileNotFoundError(
                        f"Local DINOv2 hub code not found at {local_code_path}; "
                        "cannot instantiate an architecture for local weights."
                    )
                model = torch.hub.load(
                    str(local_code_path), backbone_name, source="local", pretrained=False
                )
                state = torch.load(str(weight_path), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if not isinstance(state, dict):
                    raise TypeError(f"Expected a state_dict-like object, got {type(state)}")
                model_keys = set(model.state_dict().keys())
                state_keys = set(state.keys())
                if not model_keys.intersection(state_keys):
                    raise RuntimeError(
                        "Local checkpoint has no keys matching the DINOv2 architecture. "
                        "Do not use HuggingFace dinov2-* checkpoints here; their key "
                        "layout is incompatible with this torch.hub model."
                    )
                model.load_state_dict(state, strict=False)
                return model
            except Exception as exc:
                load_errors.append(f"{weight_path}: {exc}")

        # Local load unavailable/failed -> last-resort remote (github) source.
        try:
            return torch.hub.load("facebookresearch/dinov2", backbone_name)
        except Exception as hub_exc:
            hub_error = hub_exc

        details = [
            f"torch.hub pretrained load failed for {backbone_name}: {hub_error}",
            f"local hub code path: {local_code_path}",
        ]
        if model_path:
            details.append(f"model_path: {model_path}")
            if existing_weights:
                details.append("local weight load errors: " + " | ".join(load_errors))
            else:
                details.append("no local .pth weights found under model_path")
        else:
            details.append("model_path was not provided")
        raise RuntimeError(
            "Unable to load a pretrained DINOv2 model. Refusing to continue with a "
            "randomly initialized vision encoder.\n" + "\n".join(details)
        ) from hub_error

    def _resolve_input_size(self) -> tuple[int, int]:
        if self.image_size_override is not None:
            raw_override = self.image_size_override
            if isinstance(raw_override, int):
                return (raw_override, raw_override)
            if isinstance(raw_override, (tuple, list)) and len(raw_override) == 2:
                return (int(raw_override[0]), int(raw_override[1]))
            raise ValueError(f"Invalid image_size override: {raw_override!r}")
        raw = getattr(self.model, "img_size", None)
        if raw is None:
            raw = getattr(getattr(self.model, "patch_embed", None), "img_size", None)
        if isinstance(raw, int):
            return (raw, raw)
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return (int(raw[0]), int(raw[1]))
        return (224, 224)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, images: List[List[Image.Image | np.ndarray]]) -> VisionEncoderOutput:
        x, batch_size, num_views = _flatten_and_preprocess(images, self.preprocess, next(self.parameters()))
        feats = self.model.forward_features(x)
        if isinstance(feats, dict):
            cls = feats["x_norm_clstoken"]
            patch = feats["x_norm_patchtokens"]
        else:
            cls = feats[:, 0, :]
            patch = feats[:, 1:, :]

        cls = cls.view(batch_size, num_views, -1)
        patch = patch.view(batch_size, num_views, patch.shape[1], patch.shape[2])
        return VisionEncoderOutput(cls_tokens=cls, patch_tokens=patch)


class DINOv3Encoder(nn.Module):
    """Hugging Face DINOv3 ViT encoder with patch-only dense outputs."""

    _HIDDEN_SIZE: Dict[str, int] = {
        "dinov3_vits16_lvd1689m": 384,
        "dinov3_vitb16_lvd1689m": 768,
    }

    def __init__(
        self,
        backbone_name: str,
        model_path: str | None = None,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if backbone_name not in self._HIDDEN_SIZE:
            raise NotImplementedError(f"Unsupported DINOv3 backbone: {backbone_name}")
        if not model_path:
            raise ValueError(
                "DINOv3 requires a local Hugging Face model_path; refusing a remote or "
                "randomly initialized load."
            )

        from transformers import AutoModel

        self.backbone_name = backbone_name
        self.model_path = str(model_path)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self._hidden_size = int(getattr(self.model.config, "hidden_size"))
        expected_hidden = self._HIDDEN_SIZE[backbone_name]
        if self._hidden_size != expected_hidden:
            raise ValueError(
                f"{backbone_name} expected hidden_size={expected_hidden}, "
                f"got {self._hidden_size} from {self.model_path}"
            )
        self.patch_size = int(getattr(self.model.config, "patch_size"))
        self.image_size_override = image_size
        input_height, input_width = self._resolve_input_size()
        if input_height % self.patch_size or input_width % self.patch_size:
            raise ValueError(
                f"Input {(input_height, input_width)} must be divisible by "
                f"patch_size={self.patch_size}"
            )
        self.num_patches = (input_height // self.patch_size) * (
            input_width // self.patch_size
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(self._resolve_input_size()),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _resolve_input_size(self) -> tuple[int, int]:
        raw = self.image_size_override
        if raw is None:
            raw = getattr(self.model.config, "image_size", 224)
        if isinstance(raw, int):
            return (raw, raw)
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return (int(raw[0]), int(raw[1]))
        raise ValueError(f"Invalid image_size: {raw!r}")

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, images: List[List[Image.Image | np.ndarray]]) -> VisionEncoderOutput:
        return self.forward_at_size(images, self._resolve_input_size())

    def forward_at_size(
        self,
        images: List[List[Image.Image | np.ndarray]],
        image_size: int | tuple[int, int],
    ) -> VisionEncoderOutput:
        """Encode an alternate resolution with the same DINOv3 weights."""
        if isinstance(image_size, int):
            size = (image_size, image_size)
        else:
            size = (int(image_size[0]), int(image_size[1]))
        if size[0] % self.patch_size or size[1] % self.patch_size:
            raise ValueError(f"Input {size} must be divisible by patch_size={self.patch_size}")
        preprocess = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        x, batch_size, num_views = _flatten_and_preprocess(
            images, preprocess, next(self.parameters())
        )
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input {(height, width)} must be divisible by patch_size={self.patch_size}"
            )
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        hidden = self.model(pixel_values=x).last_hidden_state
        if hidden.shape[1] < num_patches + 1:
            raise ValueError(
                f"DINOv3 returned {hidden.shape[1]} tokens for {num_patches} patches"
            )

        # DINOv3 prepends CLS and optional register tokens. Patch tokens are the
        # final spatially ordered tokens; registers must never enter FDM slots.
        cls = hidden[:, 0]
        patch = hidden[:, -num_patches:]
        cls = cls.view(batch_size, num_views, self._hidden_size)
        patch = patch.view(batch_size, num_views, num_patches, self._hidden_size)
        return VisionEncoderOutput(cls_tokens=cls, patch_tokens=patch)


def build_vision_encoder(
    backbone_name: str,
    model_path: str | None = None,
    image_size: int | tuple[int, int] | None = None,
) -> nn.Module:
    if backbone_name.startswith("dinov2_"):
        return DINOv2Encoder(
            backbone_name=backbone_name,
            model_path=model_path,
            image_size=image_size,
        )
    if backbone_name.startswith("dinov3_"):
        return DINOv3Encoder(
            backbone_name=backbone_name,
            model_path=model_path,
            image_size=image_size,
        )
    raise NotImplementedError(f"Unsupported vision backbone: {backbone_name}")


VisionEncoder = build_vision_encoder
