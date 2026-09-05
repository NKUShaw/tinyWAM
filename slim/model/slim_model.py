"""SLIM policy model with DINOv2 observations and a two-stream transformer."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel

from slim.model.base import PolicyModel
from slim.model.config import merge_runtime_defaults


from .defaults import SLIMDefaultConfig
from .image_augment import apply_image_augment, build_training_image_augment
from .vision_encoder import build_vision_encoder
from .wrist_detail import HighResolutionWristAdapter
from .control_latent import FactorizedControlEncoder

from .slim_transformer import SLIMTransformer


class SLIMModel(PolicyModel):
    """Language-conditioned latent interaction policy."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_runtime_defaults(SLIMDefaultConfig, config)

        dino_cfg = self.config.framework.dino
        self.vision_encoder = build_vision_encoder(
            backbone_name=dino_cfg.backbone_name,
            model_path=dino_cfg.get("model_path", None),
            image_size=dino_cfg.get("image_size", None),
        )
        if dino_cfg.get("freeze_backbone", True):
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.num_image_views = int(dino_cfg.get("num_image_views", 2))
        self.image_augment = build_training_image_augment(dino_cfg.get("image_augment"))
        if self.image_augment is not None:
            print("[SLIM] training image_augment enabled (crop + color jitter)", flush=True)
        self.vision_condition_mode = str(
            dino_cfg.get("vision_condition_mode", "dense_patch")
        ).lower()
        vision_hidden = self.vision_encoder.hidden_size
        wrist_detail_cfg = dino_cfg.get("wrist_detail", {})
        self.wrist_detail_enabled = bool(wrist_detail_cfg.get("enabled", False))
        self.wrist_detail_high_resolution = int(wrist_detail_cfg.get("high_resolution", 256))
        self.wrist_detail_adapter = None
        if self.wrist_detail_enabled:
            if self.vision_condition_mode != "dense_patch" or self.num_image_views != 2:
                raise ValueError("wrist_detail requires dense_patch mode with exactly two views")
            if not hasattr(self.vision_encoder, "forward_at_size"):
                raise ValueError("wrist_detail requires a vision encoder with forward_at_size")
            self.wrist_detail_adapter = HighResolutionWristAdapter(
                vision_hidden, int(wrist_detail_cfg.get("bottleneck_dim", 96))
            )

        control_cfg = self.config.framework.get("control_latent", {})
        self.control_latent_enabled = bool(control_cfg.get("enabled", False))
        self.control_encoder = None
        if self.control_latent_enabled:
            if self.vision_condition_mode != "dense_patch":
                raise ValueError("control_latent requires dense_patch inputs")
            if not dino_cfg.get("freeze_backbone", True):
                raise ValueError("The initial control_latent recipe requires a frozen vision backbone")
            if self.wrist_detail_enabled:
                raise ValueError("Run control_latent and wrist_detail as separate experiments")
            patches = getattr(self.vision_encoder, "num_patches", None)
            if patches is None:
                raise ValueError("control_latent requires an encoder with explicit num_patches")
            self.control_encoder = FactorizedControlEncoder(
                vision_hidden, self.num_image_views, int(patches),
                int(self.config.framework.action_model.get("language_embedding_dim", 512)),
                control_cfg,
            )

        if self.vision_condition_mode == "cls":
            cond_dim = vision_hidden * self.num_image_views
        elif self.vision_condition_mode == "dense_patch":
            cond_dim = vision_hidden
            patches_per_view = getattr(self.vision_encoder, "num_patches", None)
            if patches_per_view is not None:
                expected_tokens = self.num_image_views * int(patches_per_view)
                if self.control_encoder is not None:
                    expected_tokens = self.control_encoder.output_tokens
                configured_tokens = int(
                    self.config.framework.action_model.mot.get(
                        "num_future_tokens", expected_tokens
                    )
                )
                if configured_tokens != expected_tokens:
                    raise ValueError(
                        "Dense visual token mismatch: "
                        f"representation has {expected_tokens} tokens, "
                        f"but num_future_tokens={configured_tokens}"
                    )
        else:
            raise ValueError(
                "SLIM supports vision_condition_mode='dense_patch' or 'cls', "
                f"got {self.vision_condition_mode!r}"
            )

        # -- EMA vision target encoder ----------------------------------------
        ema_cfg = self.config.framework.get("ema", {})
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema_momentum = float(ema_cfg.get("momentum", 0.999))
        self.ema_control_encoder = None
        if self.ema_enabled:
            self.ema_vision_encoder = copy.deepcopy(self.vision_encoder)
            for p in self.ema_vision_encoder.parameters():
                p.requires_grad = False
            if self.control_encoder is not None:
                self.ema_control_encoder = copy.deepcopy(self.control_encoder)
                self.ema_control_encoder.requires_grad_(False)
            self._init_ema_fp32_shadow()
        else:
            self.ema_vision_encoder = None
            self._ema_fp32_buffer_names = []

        # Keep module attribute names stable for checkpoint compatibility.
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = cond_dim
        self.action_model = SLIMTransformer(full_config=self.config)
        dynamics_cfg = self.config.framework.get("policy_dynamics", {})
        self.policy_dynamics_enabled = bool(dynamics_cfg.get("enabled", False))
        self.policy_dynamics_weight = float(dynamics_cfg.get("loss_weight", 0.1))
        if self.policy_dynamics_enabled and not self.ema_enabled:
            raise ValueError("policy_dynamics requires EMA targets")
        if self.policy_dynamics_enabled and self.policy_dynamics_weight <= 0:
            raise ValueError("policy_dynamics.loss_weight must be positive")
        if self.wrist_detail_enabled and bool(wrist_detail_cfg.get("freeze_except_adapter", False)):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.wrist_detail_adapter.parameters():
                parameter.requires_grad = True
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        state_aug_cfg = self.config.framework.action_model.get("state_augment", {})
        self.state_augment_enabled = bool(state_aug_cfg.get("enabled", False))
        self.state_augment_pos_std = float(state_aug_cfg.get("pos_std", state_aug_cfg.get("std", 0.0)))
        self.state_augment_rot_std = float(state_aug_cfg.get("rot_std", state_aug_cfg.get("std", 0.0)))
        self.state_augment_gripper_std = float(
            state_aug_cfg.get("gripper_std", state_aug_cfg.get("std", 0.0))
        )
        if self.state_augment_enabled:
            print(
                "[SLIM] training state_augment enabled "
                f"(pos_std={self.state_augment_pos_std}, "
                f"rot_std={self.state_augment_rot_std}, "
                f"gripper_std={self.state_augment_gripper_std})",
                flush=True,
            )

        self.use_future_image_condition = bool(
            self.config.framework.action_model.get("use_future_image_condition", False)
        )
        self.mot_task_mode = str(
            self.config.framework.action_model.get("mot", {}).get("task_mode", "idm_fdm")
        )
        self.use_language_condition = bool(
            self.config.framework.action_model.get("use_language_condition", False)
        )
        language_encoder_path = (
            self.config.framework.action_model.get("language_encoder_path")
            or os.environ.get("T5_MODEL_DIR")
        )
        self.language_encoder_path = (
            str(language_encoder_path) if language_encoder_path else None
        )
        self.lang_index_by_dataset: dict = {}
        self.lang_embeddings_by_dataset: dict = {}
        self.lang_tokenizer = None
        self.lang_encoder = None
        self._load_offline_language_embeddings()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.control_latent_enabled:
            self.vision_encoder.eval()
            if self.ema_vision_encoder is not None:
                self.ema_vision_encoder.eval()
            if self.ema_control_encoder is not None:
                self.ema_control_encoder.eval()
        return self

    # -- Language helpers (identical to DINO_MoT) ----------------------------

    def _load_offline_language_embeddings(self):
        offline_root = self.config.framework.action_model.get("offline_lang_emb_root", None)
        if not self.use_language_condition or not offline_root:
            return
        root = Path(str(offline_root))
        for npy_path in sorted(root.glob("*_t5small_lang_emb.npy")):
            dataset_name = npy_path.name.removesuffix("_t5small_lang_emb.npy")
            index_path = Path(str(npy_path).removesuffix(".npy") + ".index.json")
            if not index_path.exists():
                alt_index_path = npy_path.with_suffix(".npy.index.json")
                if alt_index_path.exists():
                    index_path = alt_index_path
                else:
                    print(
                        f"[SLIM][LANG_CACHE] skip dataset={dataset_name}: missing index json for {npy_path}",
                        flush=True,
                    )
                    continue
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.lang_index_by_dataset[dataset_name] = payload.get("index", payload)
            self.lang_embeddings_by_dataset[dataset_name] = np.load(str(npy_path), mmap_mode="r")

    def _lazy_load_online_language_encoder(self, device: torch.device):
        if self.lang_tokenizer is not None and self.lang_encoder is not None:
            return
        if not self.language_encoder_path:
            raise RuntimeError(
                "Online language encoding requires model.language_encoder_path "
                "or the T5_MODEL_DIR environment variable."
            )
        self.lang_tokenizer = AutoTokenizer.from_pretrained(
            self.language_encoder_path, local_files_only=True
        )
        self.lang_encoder = T5EncoderModel.from_pretrained(
            self.language_encoder_path, local_files_only=True
        ).to(device)
        self.lang_encoder.eval()

    def _encode_language_online(
        self, examples: List[dict], device: torch.device, dtype: torch.dtype
    ):
        texts = [str(x.get("lang", "")) for x in examples]
        self._lazy_load_online_language_encoder(device=device)
        max_lang_tokens = int(self.config.framework.action_model.get("max_lang_tokens", 32))
        toks = self.lang_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_lang_tokens,
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            hidden = self.lang_encoder(**toks).last_hidden_state
        return hidden.to(dtype=dtype), toks["attention_mask"].to(dtype=dtype)

    def _encode_language(
        self, examples: List[dict], device: torch.device, dtype: torch.dtype
    ):
        if not self.use_language_condition:
            return None, None
        if "lang_embs" in examples[0]:
            lang_embs = np.array([x["lang_embs"] for x in examples], dtype=np.float32)
            lengths = [int(x.get("lang_length", lang_embs.shape[1])) for x in examples]
            attn_mask = np.zeros((lang_embs.shape[0], lang_embs.shape[1]), dtype=np.float32)
            for idx, length in enumerate(lengths):
                attn_mask[idx, : min(length, lang_embs.shape[1])] = 1.0
            return (
                torch.from_numpy(lang_embs).to(device=device, dtype=dtype),
                torch.from_numpy(attn_mask).to(device=device, dtype=dtype),
            )
        if not self.lang_embeddings_by_dataset or not self.lang_index_by_dataset:
            return self._encode_language_online(examples=examples, device=device, dtype=dtype)

        embs, lengths = [], []
        for example in examples:
            text = str(example.get("lang", ""))
            dataset_name = str(example.get("dataset_name", ""))
            dataset_index = self.lang_index_by_dataset.get(dataset_name)
            dataset_embs = self.lang_embeddings_by_dataset.get(dataset_name)
            if dataset_index is None or dataset_embs is None:
                return self._encode_language_online(examples=examples, device=device, dtype=dtype)
            meta = dataset_index.get(text)
            if meta is None:
                return self._encode_language_online(examples=examples, device=device, dtype=dtype)
            offset, length = int(meta["offset"]), int(meta["length"])
            embs.append(np.asarray(dataset_embs[offset : offset + length], dtype=np.float32))
            lengths.append(length)
        max_len = max(x.shape[0] for x in embs)
        dim = embs[0].shape[1]
        padded = np.zeros((len(embs), max_len, dim), dtype=np.float32)
        attn_mask = np.zeros((len(embs), max_len), dtype=np.float32)
        for idx, arr in enumerate(embs):
            padded[idx, : arr.shape[0]] = arr
            attn_mask[idx, : lengths[idx]] = 1.0
        return (
            torch.from_numpy(padded).to(device=device, dtype=dtype),
            torch.from_numpy(attn_mask).to(device=device, dtype=dtype),
        )

    # -- Vision helpers -------------------------------------------------------

    def _maybe_augment_images(self, batch_images):
        """Apply PIL augment on online encoder inputs during training only."""
        if not self.training or self.image_augment is None:
            return batch_images
        return apply_image_augment(batch_images, self.image_augment)

    def _encode_vision_with(
        self,
        batch_images,
        encoder: nn.Module,
    ) -> torch.Tensor:
        vision_out = encoder(batch_images)
        cls = vision_out.cls_tokens
        if cls.shape[1] != self.num_image_views:
            raise ValueError(
                f"Expected {self.num_image_views} image views, got {cls.shape[1]}"
            )
        if self.vision_condition_mode == "cls":
            return cls.reshape(cls.shape[0], 1, cls.shape[1] * cls.shape[2])
        if self.vision_condition_mode == "dense_patch":
            patch = vision_out.patch_tokens
            if patch.dim() != 4:
                raise ValueError(f"Expected dense patch tokens [B,V,P,D], got {tuple(patch.shape)}")
            if self.wrist_detail_enabled and encoder is self.vision_encoder:
                patches_per_view = patch.shape[2]
                base_side = int(patches_per_view**0.5)
                if base_side * base_side != patches_per_view:
                    raise ValueError(f"wrist_detail requires a square patch grid, got {patches_per_view}")
                wrist_images = [[views[1]] for views in batch_images]
                hr_out = encoder.forward_at_size(
                    wrist_images, self.wrist_detail_high_resolution
                )
                hr_patch = hr_out.patch_tokens[:, 0]
                hr_tokens = hr_patch.shape[1]
                hr_side = int(hr_tokens**0.5)
                if hr_side * hr_side != hr_tokens:
                    raise ValueError(f"wrist_detail requires a square HR grid, got {hr_tokens}")
                fused_wrist = self.wrist_detail_adapter(
                    patch[:, 1], hr_patch, (base_side, base_side), (hr_side, hr_side)
                )
                patch = torch.stack((patch[:, 0], fused_wrist), dim=1)
            return patch.reshape(patch.shape[0], patch.shape[1] * patch.shape[2], patch.shape[3])
        raise RuntimeError(f"Unexpected vision mode: {self.vision_condition_mode}")

    def _encode_vision(
        self,
        batch_images,
        lang_embs: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ):
        """Online encoder used for current-frame embeddings (gradients flow through)."""
        batch_images = self._maybe_augment_images(batch_images)
        features = self._encode_vision_with(batch_images, self.vision_encoder)
        auxiliary = {}
        if self.control_encoder is not None:
            features, auxiliary = self.control_encoder(
                features, lang_embs, lang_mask, compute_aux=return_aux
            )
        return (features, auxiliary) if return_aux else features

    @torch.no_grad()
    def _encode_vision_ema(
        self,
        batch_images,
        lang_embs: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """EMA encoder used as target for future latents (no gradient)."""
        if self.ema_enabled:
            features = self._encode_vision_with(batch_images, self.ema_vision_encoder)
            if self.ema_control_encoder is not None:
                features, _ = self.ema_control_encoder(features, lang_embs, lang_mask)
            return features
        return self._encode_vision(batch_images, lang_embs=lang_embs, lang_mask=lang_mask).detach()

    def _ema_parameter_pairs(self):
        pairs = list(zip(self.vision_encoder.parameters(), self.ema_vision_encoder.parameters()))
        if self.ema_control_encoder is not None:
            pairs.extend(zip(self.control_encoder.parameters(), self.ema_control_encoder.parameters()))
        return pairs

    def sync_ema_from_online(self):
        if not self.ema_enabled:
            return
        self.ema_vision_encoder.load_state_dict(self.vision_encoder.state_dict())
        if self.ema_control_encoder is not None:
            self.ema_control_encoder.load_state_dict(self.control_encoder.state_dict())
        self.refresh_ema_fp32_shadow()

    def _init_ema_fp32_shadow(self) -> None:
        """Keep EMA accumulation in fp32 even when the model runs in bf16."""
        self._ema_fp32_buffer_names = []
        params = [target for _, target in self._ema_parameter_pairs()]
        for idx, param in enumerate(params):
            name = f"_ema_fp32_{idx}"
            self.register_buffer(name, param.detach().float().clone(), persistent=True)
            self._ema_fp32_buffer_names.append(name)

    def refresh_ema_fp32_shadow(self) -> None:
        """Synchronize fp32 EMA buffers with the current EMA module weights."""
        if not self.ema_enabled:
            return
        params = [target for _, target in self._ema_parameter_pairs()]
        if len(params) != len(self._ema_fp32_buffer_names):
            raise RuntimeError(
                f"EMA shadow/parameter count mismatch: "
                f"buffers={len(self._ema_fp32_buffer_names)} params={len(params)}"
            )
        with torch.no_grad():
            for shadow_name, param in zip(self._ema_fp32_buffer_names, params):
                shadow = getattr(self, shadow_name)
                shadow.data.copy_(param.detach().float())

    def _ema_update_param(
        self,
        online_p: torch.nn.Parameter,
        ema_p: torch.nn.Parameter,
        shadow_name: str,
        momentum: float,
    ) -> None:
        """Update one EMA parameter, including DeepSpeed ZeRO-3 partitioned params."""
        try:
            import deepspeed

            is_zero_param = hasattr(online_p, "ds_id") or hasattr(ema_p, "ds_id")
        except Exception:
            deepspeed = None
            is_zero_param = False

        shadow = getattr(self, shadow_name)
        if shadow.dtype != torch.float32:
            shadow.data = shadow.data.float()

        if deepspeed is not None and is_zero_param:
            rank = dist.get_rank() if dist.is_initialized() else 0
            with deepspeed.zero.GatheredParameters([online_p, ema_p], modifier_rank=0):
                if rank == 0:
                    shadow.data.mul_(momentum).add_(online_p.data.float(), alpha=1.0 - momentum)
                    ema_p.data.copy_(shadow.data.to(dtype=ema_p.dtype))
            return

        shadow.data.mul_(momentum).add_(online_p.data.float(), alpha=1.0 - momentum)
        ema_p.data.copy_(shadow.data.to(device=ema_p.device, dtype=ema_p.dtype))

    def update_ema(self):
        """EMA momentum update for vision modules.  Call after each optimizer step."""
        if not self.ema_enabled:
            return
        m = self.ema_momentum
        shadow_names = iter(self._ema_fp32_buffer_names)
        with torch.no_grad():
            for online_p, ema_p in self._ema_parameter_pairs():
                self._ema_update_param(online_p, ema_p, next(shadow_names), m)

    # -- State helper ---------------------------------------------------------

    def _extract_state(
        self, examples: List[dict], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        if not self.action_model.use_state_condition:
            return None
        states = [example.get("state") for example in examples]
        if all(s is None for s in states):
            return None
        state_dim = int(self.config.framework.action_model.get("state_dim", 7))
        arr = np.array(
            [
                np.squeeze(s, axis=0) if s is not None else np.zeros(state_dim, dtype=np.float32)
                for s in states
            ],
            dtype=np.float32,
        )
        state = torch.from_numpy(arr).to(device=device, dtype=dtype)
        if self.training and self.state_augment_enabled:
            noise = torch.zeros_like(state)
            if self.state_augment_pos_std > 0.0:
                noise[:, :3] = torch.randn_like(state[:, :3]) * self.state_augment_pos_std
            if state.shape[1] >= 6 and self.state_augment_rot_std > 0.0:
                noise[:, 3:6] = torch.randn_like(state[:, 3:6]) * self.state_augment_rot_std
            if state.shape[1] >= 7 and self.state_augment_gripper_std > 0.0:
                noise[:, 6:7] = torch.randn_like(state[:, 6:7]) * self.state_augment_gripper_std
            state = state + noise
        return state

    # -- Forward / predict ----------------------------------------------------

    def forward(self, examples: List[dict] = None, **kwargs):
        batch_images = [example["image"] for example in examples]
        actions_np = np.array([example["action"] for example in examples], dtype=np.float32)

        vision_param = next(self.vision_encoder.parameters())
        lang_embs, lang_mask = self._encode_language(
            examples=examples, device=vision_param.device, dtype=vision_param.dtype
        )
        vl_embs, representation_losses = self._encode_vision(
            batch_images, lang_embs=lang_embs, lang_mask=lang_mask, return_aux=True
        )
        task_mode = str(kwargs.get("mot_task_mode", self.mot_task_mode)).strip().lower()
        run_policy_dynamics = (
            task_mode == "policy" and self.policy_dynamics_enabled
            and bool(kwargs.get("run_policy_dynamics", True))
        )
        future_vl_embs_cond = None
        future_vl_embs_target = None
        needs_idm_cond = task_mode in {"idm", "idm_fdm"}
        needs_future_target = task_mode in {"fdm", "idm_fdm"} or run_policy_dynamics
        current_vl_embs_target = None
        if needs_future_target and self.action_model.future_delta_loss_weight > 0:
            current_vl_embs_target = self._encode_vision_ema(
                batch_images, lang_embs=lang_embs, lang_mask=lang_mask
            )
        if needs_idm_cond or needs_future_target:
            if "future_image" not in examples[0]:
                raise ValueError(
                    "Dynamics objectives require `future_image` in each example."
                )
            future_images = [example["future_image"] for example in examples]
            # IDM conditions on the actual future observation; keep this online so gradients
            # can flow through the conditional future branch.
            if needs_idm_cond:
                future_vl_embs_cond = self._encode_vision(
                    future_images, lang_embs=lang_embs, lang_mask=lang_mask
                )
            if needs_future_target:
                future_vl_embs_target = self._encode_vision_ema(
                    future_images, lang_embs=lang_embs, lang_mask=lang_mask
                )

            # Individual task branches only consume one side, but mixed modes expect both.
            if task_mode == "idm_fdm":
                if future_vl_embs_cond is None:
                    future_vl_embs_cond = future_vl_embs_target
                if future_vl_embs_target is None:
                    future_vl_embs_target = future_vl_embs_cond.detach()

        device, dtype = vl_embs.device, vl_embs.dtype
        state_tensor = self._extract_state(examples, device=device, dtype=dtype)
        actions = torch.from_numpy(actions_np).to(device=device, dtype=dtype)[
            :, -self.action_horizon :, :
        ]
        repeated_steps = int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 1)
        )

        if task_mode == "policy":
            action_kwargs = {
                "lang_embs": lang_embs,
                "language_encoder_attention_mask": lang_mask,
                "mot_task_mode": task_mode,
                "state": state_tensor,
                "repeated_steps": repeated_steps,
            }
            if future_vl_embs_cond is not None:
                action_kwargs["future_vl_embs_cond"] = future_vl_embs_cond
            if future_vl_embs_target is not None:
                action_kwargs["future_vl_embs_target"] = future_vl_embs_target
            loss_output = self.action_model(vl_embs, actions, **action_kwargs)
            if run_policy_dynamics:
                dynamics = self.action_model.forward_fdm(
                    vl_embs, actions, future_vl_embs_target,
                    language=lang_embs, language_mask=lang_mask, state=state_tensor,
                    current_target=current_vl_embs_target, return_details=True,
                )
                loss_output.update(dynamics)
                loss_output["action_loss"] = (
                    loss_output["action_loss"] + self.policy_dynamics_weight * dynamics["fdm_loss"]
                )
        else:
            action_kwargs = {
                "lang_embs": lang_embs.repeat(repeated_steps, 1, 1) if lang_embs is not None else None,
                "language_encoder_attention_mask": (
                    lang_mask.repeat(repeated_steps, 1) if lang_mask is not None else None
                ),
                "mot_task_mode": task_mode,
                "state": state_tensor.repeat(repeated_steps, 1) if state_tensor is not None else None,
            }
            if future_vl_embs_cond is not None:
                action_kwargs["future_vl_embs_cond"] = future_vl_embs_cond.repeat(repeated_steps, 1, 1)
            if future_vl_embs_target is not None:
                action_kwargs["future_vl_embs_target"] = future_vl_embs_target.repeat(repeated_steps, 1, 1)
            if current_vl_embs_target is not None:
                action_kwargs["current_vl_embs_target"] = current_vl_embs_target.repeat(repeated_steps, 1, 1)
            loss_output = self.action_model(
                vl_embs.repeat(repeated_steps, 1, 1),
                actions.repeat(repeated_steps, 1, 1),
                **action_kwargs,
            )

        loss_output.update(representation_losses)
        if "representation_loss" in representation_losses:
            loss_output["action_loss"] = (
                loss_output["action_loss"] + representation_losses["representation_loss"]
            )
        loss_output["monitor/past_vl_embs"] = vl_embs.detach()
        monitor_future = (
            future_vl_embs_target if future_vl_embs_target is not None else future_vl_embs_cond
        )
        if monitor_future is not None:
            loss_output["monitor/future_vl_embs"] = monitor_future.detach()
        return loss_output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        batch_images = [example["image"] for example in examples]
        vision_param = next(self.vision_encoder.parameters())
        lang_embs, lang_mask = self._encode_language(
            examples=examples, device=vision_param.device, dtype=vision_param.dtype
        )
        vl_embs = self._encode_vision(batch_images, lang_embs=lang_embs, lang_mask=lang_mask)
        state_tensor = self._extract_state(
            examples, device=vl_embs.device, dtype=vl_embs.dtype
        )
        pred_actions = self.action_model.predict_action(
            vl_embs,
            state=state_tensor,
            lang_embs=lang_embs,
            language_encoder_attention_mask=lang_mask,
        )
        return {
            "normalized_actions": pred_actions.detach().to(dtype=torch.float32).cpu().numpy()
        }

    @torch.inference_mode()
    def eval_future_latent(self, examples: List[dict]) -> float:
        batch_images = [example["image"] for example in examples]
        future_images = [example["future_image"] for example in examples]
        vision_param = next(self.vision_encoder.parameters())
        lang_embs, lang_mask = self._encode_language(
            examples=examples, device=vision_param.device, dtype=vision_param.dtype
        )
        vl_embs = self._encode_vision(batch_images, lang_embs=lang_embs, lang_mask=lang_mask)
        gt_future = self._encode_vision_ema(future_images, lang_embs=lang_embs, lang_mask=lang_mask)
        state_tensor = self._extract_state(
            examples, device=vl_embs.device, dtype=vl_embs.dtype
        )
        loss = self.action_model.eval_future_latent(
            vl_embs,
            gt_future,
            actions=torch.as_tensor(
                np.asarray([example["action"] for example in examples], dtype=np.float32),
                device=vl_embs.device, dtype=vl_embs.dtype,
            )[:, -self.action_horizon :],
            lang_embs=lang_embs,
            language_encoder_attention_mask=lang_mask,
            state=state_tensor,
        )
        return float(loss.item())

    @torch.inference_mode()
    def evaluate_dynamics(self, examples: List[dict]) -> dict[str, float]:
        """Offline diagnostics for an existing checkpoint, without optimizer updates."""
        if self.training:
            raise ValueError("Call model.eval() before evaluating dynamics")
        images = [example["image"] for example in examples]
        future_images = [example["future_image"] for example in examples]
        param = next(self.vision_encoder.parameters())
        language, mask = self._encode_language(examples, param.device, param.dtype)
        current = self._encode_vision(images, language, mask)
        current_target = self._encode_vision_ema(images, language, mask)
        future = self._encode_vision_ema(future_images, language, mask)
        state = self._extract_state(examples, current.device, current.dtype)
        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples], dtype=np.float32),
            device=current.device, dtype=current.dtype,
        )[:, -self.action_horizon :]
        prediction, delta = self.action_model.predict_future(
            current, actions, language, mask, state, return_delta=True
        )
        real_loss = self.action_model._future_loss(prediction, future)
        identity_loss = self.action_model._future_loss(current_target, future)
        metrics = {"fdm_future_loss": float(real_loss), "identity_loss": float(identity_loss)}
        if len(examples) > 1:
            shuffled = self.action_model.predict_future(
                current, actions.roll(1, 0), language, mask, state
            )
            shuffled_loss = self.action_model._future_loss(shuffled, future)
            metrics["shuffled_action_loss"] = float(shuffled_loss)
            metrics["action_sensitivity_gap"] = float(shuffled_loss - real_loss)
        if delta is not None:
            import torch.nn.functional as F
            target_delta = (
                F.layer_norm(future.float(), (future.shape[-1],))
                - F.layer_norm(current_target.float(), (current_target.shape[-1],))
            )
            metrics["delta_loss"] = float(F.smooth_l1_loss(delta.float(), target_delta))
            metrics["zero_delta_loss"] = float(F.smooth_l1_loss(torch.zeros_like(delta).float(), target_delta))
        return metrics
