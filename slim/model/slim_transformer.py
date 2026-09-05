"""SLIM Transformer.

Observation and action tokens use separate projections and feed-forward
networks while sharing joint self-attention. Each stream independently
cross-attends to language tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from .components import ACTION_MODEL_PRESETS, MLP
from .action_encoder import ActionEncoder



# -- Small helpers -------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SLIMSinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim - 1, 1))
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb.to(dtype=timesteps.dtype if timesteps.is_floating_point() else torch.float32)


# -- Cross-attention helper ----------------------------------------------------

class _CrossAttn(nn.Module):
    """Minimal cross-attention used per-block for language injection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        b, s, _, _ = x.shape
        return x.view(b, s, self.dim)

    def compute_kv(self, ctx: torch.Tensor):
        """Pre-compute K and V from context (language) for caching."""
        return self.k(ctx), self.v(ctx)

    def forward_with_kv(
        self,
        q_input: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward using pre-computed (possibly cached) K, V."""
        q = self._split(self.q(q_input))
        k = self._split(k)
        v = self._split(v)
        if ctx_mask is not None:
            # ctx_mask: [B, S] float (1=valid, 0=pad) -> attn_bias [B, 1, Nq, Nk]
            pad = ctx_mask < 0.5  # [B, S]
            attn_bias = torch.zeros(
                q.shape[0], 1, q.shape[2], k.shape[2],
                device=q.device, dtype=q.dtype,
            )
            attn_bias.masked_fill_(pad.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_out = self._merge(F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias))
        else:
            attn_out = self._merge(F.scaled_dot_product_attention(q, k, v))
        return self.out(attn_out)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k, v = self.compute_kv(ctx)
        return self.forward_with_kv(x, k, v, ctx_mask=ctx_mask)


# -- SLIM Block ----------------------------------------------------------------

class SLIMBlock(nn.Module):
    """Two-stream (state + action) joint self-attention with per-stream cross-attn to language.

    Language tokens are NOT in the joint SA stream.  Instead, each stream has an
    independent cross-attention sublayer where it attends to the language context.

    Conditioning (AdaLN) mirrors the original MoT:
      - ctx_cond_proj: task-only, applied to both state SA and state FFN (4 chunks).
      - act_cond_proj: time+task,  applied to both action SA and action FFN (4 chunks).
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        hidden_dim = int(dim * ffn_ratio)

        # -- state stream --
        self.norm_state_sa = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_qkv_state = nn.Linear(dim, dim * 3)
        self.to_out_state = nn.Linear(dim, dim)
        self.norm_state_ca = nn.LayerNorm(dim)   # standard LN (no AdaLN for CA)
        self.cross_attn_state = _CrossAttn(dim, num_heads)
        self.norm_state_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_state = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim)
        )

        # -- action stream --
        self.norm_action_sa = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_qkv_action = nn.Linear(dim, dim * 3)
        self.to_out_action = nn.Linear(dim, dim)
        self.norm_action_ca = nn.LayerNorm(dim)
        self.cross_attn_action = _CrossAttn(dim, num_heads)
        self.norm_action_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_action = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim)
        )

        # conditioning: ctx stream (state) -> 4 chunks; act stream (action) -> 4 chunks
        self.ctx_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))
        self.act_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        b, s, _, _ = x.shape
        return x.view(b, s, self.dim)

    def build_ctx_kv(
        self,
        state_tokens: torch.Tensor,
        cond_ctx: torch.Tensor,
        lang_context: torch.Tensor | None,
        lang_mask: torch.Tensor | None,
    ) -> dict:
        """Pre-compute all time-invariant K/V tensors for caching.

        Computes:
          k_s, v_s: state K/V for joint SA (used by action in forward_action_only).
          k_l_state, v_l_state: lang K/V for state cross-attn (only needed when
              advancing state in build_ctx_cache; not used in forward_action_only).
          k_l_action, v_l_action: lang K/V for action cross-attn (used every step).
        """
        s_sa_shift, s_sa_scale, _s_ffn_shift, _s_ffn_scale = (
            self.ctx_cond_proj(cond_ctx).chunk(4, dim=-1)
        )
        state_norm = _modulate(self.norm_state_sa(state_tokens), s_sa_shift, s_sa_scale)
        _, k_s, v_s = self.to_qkv_state(state_norm).chunk(3, dim=-1)
        cache: dict = {"k_s": k_s, "v_s": v_s}
        if lang_context is not None:
            k_l_state, v_l_state = self.cross_attn_state.compute_kv(lang_context)
            k_l_action, v_l_action = self.cross_attn_action.compute_kv(lang_context)
            cache.update({
                "k_l_state": k_l_state, "v_l_state": v_l_state,
                "k_l_action": k_l_action, "v_l_action": v_l_action,
                "lang_mask": lang_mask,
            })
        return cache

    def forward_action_only(
        self,
        action_tokens: torch.Tensor,
        cond_act: torch.Tensor,
        ctx_kv: dict,
    ) -> torch.Tensor:
        """Process only the action stream using pre-computed state/lang K/V.

        Flow per block:
          1. Joint SA: action attends to [cached state K/V | action K/V].
          2. Cross-attn: action attends to cached lang K/V (if present).
          3. FFN.
        """
        a_sa_shift, a_sa_scale, a_ffn_shift, a_ffn_scale = (
            self.act_cond_proj(cond_act).chunk(4, dim=-1)
        )
        # 1. Joint SA
        action_norm = _modulate(self.norm_action_sa(action_tokens), a_sa_shift, a_sa_scale)
        q_a, k_a, v_a = self.to_qkv_action(action_norm).chunk(3, dim=-1)
        k = torch.cat([ctx_kv["k_s"], k_a], dim=1)
        v = torch.cat([ctx_kv["v_s"], v_a], dim=1)
        # No causal mask in action-only mode: action already cannot be attended by state.
        action_tokens = action_tokens + self.to_out_action(
            self._merge(F.scaled_dot_product_attention(self._split(q_a), self._split(k), self._split(v)))
        )
        # 2. Cross-attn to language (if cached)
        if "k_l_action" in ctx_kv:
            action_tokens = action_tokens + self.cross_attn_action.forward_with_kv(
                self.norm_action_ca(action_tokens),
                ctx_kv["k_l_action"], ctx_kv["v_l_action"],
                ctx_mask=ctx_kv.get("lang_mask"),
            )
        # 3. FFN
        action_tokens = action_tokens + self.ffn_action(
            _modulate(self.norm_action_ffn(action_tokens), a_ffn_shift, a_ffn_scale)
        )
        return action_tokens

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor | None,
        cond_ctx: torch.Tensor,
        cond_act: torch.Tensor,
        lang_context: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
        n_state_ctx_rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Full two-stream forward.

        Args:
            state_tokens: [B, Ns, D]
            action_tokens: [B, Na, D] or None (state-only path for caching)
            cond_ctx: [B, D] - task-only AdaLN for state
            cond_act: [B, D] - time+task AdaLN for action
            lang_context: [B, Nl, D] - language context for cross-attn
            lang_mask: [B, Nl] float (1=valid, 0=pad)
            n_state_ctx_rows: how many leading state rows are blocked from attending to action
        """
        s_sa_shift, s_sa_scale, s_ffn_shift, s_ffn_scale = (
            self.ctx_cond_proj(cond_ctx).chunk(4, dim=-1)
        )
        state_norm = _modulate(self.norm_state_sa(state_tokens), s_sa_shift, s_sa_scale)
        q_s, k_s, v_s = self.to_qkv_state(state_norm).chunk(3, dim=-1)

        if action_tokens is None:
            # State-only path (for build_ctx_cache).
            out_s = self._merge(
                F.scaled_dot_product_attention(self._split(q_s), self._split(k_s), self._split(v_s))
            )
            state_tokens = state_tokens + self.to_out_state(out_s)
            if lang_context is not None:
                state_tokens = state_tokens + self.cross_attn_state(
                    self.norm_state_ca(state_tokens), lang_context, ctx_mask=lang_mask
                )
            state_tokens = state_tokens + self.ffn_state(
                _modulate(self.norm_state_ffn(state_tokens), s_ffn_shift, s_ffn_scale)
            )
            return state_tokens, None

        # Full two-stream joint SA.
        a_sa_shift, a_sa_scale, a_ffn_shift, a_ffn_scale = (
            self.act_cond_proj(cond_act).chunk(4, dim=-1)
        )
        action_norm = _modulate(self.norm_action_sa(action_tokens), a_sa_shift, a_sa_scale)
        q_a, k_a, v_a = self.to_qkv_action(action_norm).chunk(3, dim=-1)

        n_state = state_tokens.shape[1]
        n_action = action_tokens.shape[1]
        n_blocked = n_state if n_state_ctx_rows is None else n_state_ctx_rows

        q = torch.cat([q_s, q_a], dim=1)
        k = torch.cat([k_s, k_a], dim=1)
        v = torch.cat([v_s, v_a], dim=1)

        # P1-C: first n_blocked state rows cannot attend to action columns.
        n_total = n_state + n_action
        attn_bias = torch.zeros(
            state_tokens.shape[0], 1, n_total, n_total,
            device=state_tokens.device, dtype=state_tokens.dtype,
        )
        attn_bias[:, :, :n_blocked, n_state:] = float("-inf")

        attn_out = self._merge(
            F.scaled_dot_product_attention(
                self._split(q), self._split(k), self._split(v), attn_mask=attn_bias
            )
        )
        state_tokens = state_tokens + self.to_out_state(attn_out[:, :n_state])
        action_tokens = action_tokens + self.to_out_action(attn_out[:, n_state:])

        # Per-stream cross-attention to language.
        if lang_context is not None:
            state_tokens = state_tokens + self.cross_attn_state(
                self.norm_state_ca(state_tokens), lang_context, ctx_mask=lang_mask
            )
            action_tokens = action_tokens + self.cross_attn_action(
                self.norm_action_ca(action_tokens), lang_context, ctx_mask=lang_mask
            )

        # FFN
        state_tokens = state_tokens + self.ffn_state(
            _modulate(self.norm_state_ffn(state_tokens), s_ffn_shift, s_ffn_scale)
        )
        action_tokens = action_tokens + self.ffn_action(
            _modulate(self.norm_action_ffn(action_tokens), a_ffn_shift, a_ffn_scale)
        )
        return state_tokens, action_tokens


# -- SLIM Trunk ---------------------------------------------------------------

class SLIMTrunk(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            SLIMBlock(dim=dim, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_state_norm = nn.LayerNorm(dim)
        self.final_action_norm = nn.LayerNorm(dim)

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor | None,
        cond_ctx: torch.Tensor,
        cond_act: torch.Tensor,
        lang_context: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
        n_state_ctx_rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for block in self.blocks:
            state_tokens, action_tokens = block(
                state_tokens=state_tokens,
                action_tokens=action_tokens,
                cond_ctx=cond_ctx,
                cond_act=cond_act,
                lang_context=lang_context,
                lang_mask=lang_mask,
                n_state_ctx_rows=n_state_ctx_rows,
            )
        state_tokens = self.final_state_norm(state_tokens)
        if action_tokens is not None:
            action_tokens = self.final_action_norm(action_tokens)
        return state_tokens, action_tokens

    def build_ctx_cache(
        self,
        state_tokens: torch.Tensor,
        cond_ctx: torch.Tensor,
        lang_context: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
    ) -> tuple[list[dict], torch.Tensor]:
        """Pre-compute per-block time-invariant K/V cache.

        Returns:
            cache: list of per-block dicts with state SA K/V and optional lang CA K/V.
            final_state_tokens: normalized state output [B, Ns, D] after all blocks.
        """
        cache: list[dict] = []
        dummy_cond_act = cond_ctx.new_zeros(cond_ctx.shape)
        for block in self.blocks:
            cache.append(
                block.build_ctx_kv(state_tokens, cond_ctx, lang_context, lang_mask)
            )
            # Advance state-only through this block.
            state_tokens, _ = block(
                state_tokens, None, cond_ctx, dummy_cond_act,
                lang_context=lang_context, lang_mask=lang_mask,
            )
        return cache, self.final_state_norm(state_tokens)

    @staticmethod
    def expand_ctx_cache(cache: list[dict], n: int) -> list[dict]:
        """Expand batch-B cache to B*n via repeat_interleave (for repeated_diffusion_steps)."""
        expanded = []
        for layer_kv in cache:
            new_kv: dict = {}
            for key, val in layer_kv.items():
                if isinstance(val, torch.Tensor):
                    new_kv[key] = val.repeat_interleave(n, dim=0)
                else:
                    new_kv[key] = val
            expanded.append(new_kv)
        return expanded

    def forward_action_only(
        self,
        action_tokens: torch.Tensor,
        cond_act: torch.Tensor,
        ctx_cache: list[dict],
    ) -> torch.Tensor:
        """Action-stream-only forward using pre-computed cache (inference / repeated steps)."""
        for block, kv in zip(self.blocks, ctx_cache):
            action_tokens = block.forward_action_only(action_tokens, cond_act, kv)
        return self.final_action_norm(action_tokens)


# -- Future loss (cosine similarity) ------------------------------------------

def cosine_future_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity loss: mean of (1 - cos_sim) over all token positions.

    pred:   [B, T, D] - model prediction (gradients flow)
    target: [B, T, D] - regression target (should be .detach()-ed by caller)
    """
    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(target.float(), dim=-1)
    return (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()


def normalized_l2_future_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalized predictions and targets."""
    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(target.float(), dim=-1)
    return F.mse_loss(pred_n, tgt_n)


def mse_future_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between raw prediction and target vectors."""
    return F.mse_loss(pred.float(), target.float())


def norm_l1_future_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between feature-dim LayerNorm-normalized prediction and target vectors."""
    pred_f = pred.float()
    target_f = target.float()
    pred_n = F.layer_norm(pred_f, (pred_f.shape[-1],))
    tgt_n = F.layer_norm(target_f, (target_f.shape[-1],))
    return (pred_n - tgt_n).abs().mean()


def compute_future_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    kind = str(loss_type).strip().lower()
    if kind in {"cosine", "cos"}:
        return cosine_future_loss(pred, target)
    if kind in {"normalized_l2", "norm_l2", "l2_norm", "l2"}:
        return normalized_l2_future_loss(pred, target)
    if kind in {"mse", "raw_mse", "l2_raw"}:
        return mse_future_loss(pred, target)
    if kind in {"norm_l1", "normalized_l1", "layernorm_l1", "ln_l1"}:
        return norm_l1_future_loss(pred, target)
    raise ValueError(
        f"Unsupported future_loss_type={loss_type!r}. "
        "Choose 'cosine', 'normalized_l2', 'mse', or 'norm_l1'."
    )


class SLIMTransformer(nn.Module):
    """Two-stream transformer used by both SLIM training stages."""

    TASK_TO_ID = {"policy": 0, "idm": 1, "fdm": 2}
    TASK_MODES = frozenset({"policy", "idm", "fdm", "idm_fdm"})

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        transformer = config.get("mot", {})
        preset = ACTION_MODEL_PRESETS[str(config.get("action_model_type", "DiT-B"))]

        self.full_config = full_config
        self.input_embedding_dim = int(
            transformer.get("hidden_dim", preset["input_embedding_dim"])
        )
        self.condition_dim = int(config.diffusion_model_cfg["cross_attention_dim"])
        self.hidden_size = int(
            config.get("hidden_size", config.diffusion_model_cfg.get("output_dim", 1024))
        )
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.num_inference_timesteps = int(config.get("num_inference_timesteps", 4))
        self.num_timestep_buckets = int(config.get("num_timestep_buckets", 1000))
        self.noise_s = float(config.get("noise_s", 0.999))
        self.beta_dist = Beta(
            float(config.get("noise_beta_alpha", 1.5)),
            float(config.get("noise_beta_beta", 1.0)),
        )

        self.num_future_tokens = int(transformer.get("num_future_tokens", 512))
        self.use_language_condition = bool(config.get("use_language_condition", False))
        self.language_embedding_dim = int(config.get("language_embedding_dim", 512))
        self.max_lang_tokens = int(config.get("max_lang_tokens", 32))
        self.idm_loss_weight = float(transformer.get("idm_loss_weight", 1.0))
        self.fdm_loss_weight = float(transformer.get("fdm_loss_weight", 1.0))
        self.future_loss_type = str(
            transformer.get("future_loss_type", "cosine")
        ).strip().lower()
        self.num_action_register_tokens = int(
            transformer.get("num_action_register_tokens", 4)
        )

        self.use_state_condition = bool(transformer.get("use_state_condition", False))
        self.proprio_dim = int(config.get("state_dim", 7))
        if self.use_state_condition:
            self.proprio_proj = nn.Linear(self.proprio_dim, self.input_embedding_dim)
            nn.init.normal_(self.proprio_proj.weight, std=0.02)
            nn.init.zeros_(self.proprio_proj.bias)
        else:
            self.proprio_proj = None

        dim = self.input_embedding_dim
        self.state_input_proj = nn.Linear(self.condition_dim, dim)
        self.action_encoder = ActionEncoder(self.action_dim, dim)
        self.action_decoder = MLP(dim, self.hidden_size, self.action_dim)
        self.state_decoder = MLP(dim, self.hidden_size, self.condition_dim)

        self.future_mask_tokens = nn.Embedding(self.num_future_tokens, dim)
        nn.init.normal_(self.future_mask_tokens.weight, mean=0.0, std=0.02)

        self.state_pos_embed = nn.Embedding(
            int(transformer.get("max_state_tokens", 64)), dim
        )
        self.action_pos_embed = nn.Embedding(
            int(transformer.get("max_action_tokens", 64)), dim
        )
        nn.init.normal_(self.state_pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.action_pos_embed.weight, mean=0.0, std=0.02)

        self.action_register_tokens = nn.Embedding(
            self.num_action_register_tokens, dim
        )
        nn.init.normal_(self.action_register_tokens.weight, mean=0.0, std=0.02)

        self.lang_condition_proj = nn.Linear(self.language_embedding_dim, dim)
        self.lang_pos_embed = nn.Embedding(self.max_lang_tokens, dim)
        nn.init.normal_(self.lang_pos_embed.weight, mean=0.0, std=0.02)

        self.time_embed = SLIMSinusoidalTimeEmbedding(dim)
        self.time_condition_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.task_embedding = nn.Embedding(len(self.TASK_TO_ID), dim)
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)
        self.trunk = SLIMTrunk(
            dim=dim,
            num_layers=int(transformer.get("num_layers", 16)),
            num_heads=int(
                transformer.get("num_heads", preset["num_attention_heads"])
            ),
            ffn_ratio=float(transformer.get("ffn_ratio", 4.0)),
            dropout=float(transformer.get("dropout", 0.0)),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def sample_time(self, batch_size, device, dtype):
        sample = (
            self.beta_dist.sample([batch_size])
            .to(device=device, dtype=dtype)
            .clamp(max=self.noise_s)
        )
        return (self.noise_s - sample) / self.noise_s

    def _language_context(self, embeddings, mask):
        if not self.use_language_condition or embeddings is None:
            return None, None
        embeddings = embeddings[:, : self.max_lang_tokens]
        mask = mask[:, : self.max_lang_tokens] if mask is not None else None
        if embeddings.shape[-1] != self.language_embedding_dim:
            raise ValueError(
                f"Expected language dim {self.language_embedding_dim}, "
                f"got {embeddings.shape[-1]}"
            )
        tokens = self.lang_condition_proj(embeddings)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.lang_pos_embed(positions).unsqueeze(0).to(tokens.dtype)
        return tokens, mask

    def _proprio_token(self, state):
        if self.proprio_proj is None or state is None:
            return None
        state = state.squeeze(1) if state.ndim == 3 else state
        return self.proprio_proj(state.to(self.proprio_proj.weight.dtype)).unsqueeze(1)

    def _state_tokens(self, current, future=None, state=None):
        current = self.state_input_proj(current)
        if future is None:
            future = self.future_mask_tokens.weight.unsqueeze(0).expand(
                current.shape[0], -1, -1
            )
        else:
            future = self.state_input_proj(future)
        tokens = torch.cat([current, future.to(current.dtype)], dim=1)
        proprio = self._proprio_token(state)
        if proprio is not None:
            tokens = torch.cat([proprio.to(tokens.dtype), tokens], dim=1)
        if tokens.shape[1] > self.state_pos_embed.num_embeddings:
            raise ValueError("State sequence exceeds configured position embeddings")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return tokens + self.state_pos_embed(positions).unsqueeze(0).to(tokens.dtype)

    def _prepend_registers(self, tokens):
        if self.num_action_register_tokens <= 0:
            return tokens
        registers = self.action_register_tokens.weight.unsqueeze(0).expand(
            tokens.shape[0], -1, -1
        )
        return torch.cat([registers.to(tokens.dtype), tokens], dim=1)

    def _action_tokens(self, actions, timesteps):
        tokens = self._prepend_registers(self.action_encoder(actions, timesteps))
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return tokens + self.action_pos_embed(positions).unsqueeze(0).to(tokens.dtype)

    def _flow_targets(self, actions):
        noise = torch.randn_like(actions)
        time = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        velocity = actions - noise
        timesteps = (time * self.num_timestep_buckets).long()
        return noisy, velocity, timesteps

    def _clean_timestep(self, batch_size, device):
        return torch.full(
            (batch_size,),
            self.num_timestep_buckets - 1,
            device=device,
            dtype=torch.long,
        )

    def _context_condition(self, batch_size, task, dtype, device):
        ids = torch.full(
            (batch_size,), self.TASK_TO_ID[task], device=device, dtype=torch.long
        )
        return self.task_embedding(ids).to(dtype)

    def _action_condition(self, timesteps, task, dtype):
        task_ids = torch.full(
            (timesteps.shape[0],),
            self.TASK_TO_ID[task],
            device=timesteps.device,
            dtype=torch.long,
        )
        time = self.time_condition_proj(self.time_embed(timesteps).to(dtype))
        return time + self.task_embedding(task_ids).to(dtype)

    def _future_loss(self, prediction, target):
        return compute_future_loss(
            prediction, target.detach(), loss_type=self.future_loss_type
        )

    def forward_policy(
        self,
        observations,
        actions,
        language=None,
        language_mask=None,
        state=None,
        repeated_steps=1,
    ):
        lang_context, lang_mask = self._language_context(language, language_mask)
        context_condition = self._context_condition(
            actions.shape[0], "policy", actions.dtype, actions.device
        )
        state_tokens = self._state_tokens(observations, state=state)

        if repeated_steps > 1:
            cache, _ = self.trunk.build_ctx_cache(
                state_tokens,
                context_condition,
                lang_context=lang_context,
                lang_mask=lang_mask,
            )
            cache = SLIMTrunk.expand_ctx_cache(cache, repeated_steps)
            actions = actions.repeat_interleave(repeated_steps, dim=0)
            noisy, velocity, timesteps = self._flow_targets(actions)
            action_tokens = self._action_tokens(noisy, timesteps)
            action_condition = self._action_condition(
                timesteps, "policy", actions.dtype
            )
            output = self.trunk.forward_action_only(
                action_tokens, action_condition, cache
            )
        else:
            noisy, velocity, timesteps = self._flow_targets(actions)
            action_tokens = self._action_tokens(noisy, timesteps)
            action_condition = self._action_condition(
                timesteps, "policy", actions.dtype
            )
            _, output = self.trunk(
                state_tokens,
                action_tokens,
                context_condition,
                action_condition,
                lang_context=lang_context,
                lang_mask=lang_mask,
            )
        prediction = self.action_decoder(
            output[:, self.num_action_register_tokens :]
        )
        return F.mse_loss(prediction, velocity)

    def forward_idm(
        self, observations, actions, future, language=None, language_mask=None, state=None
    ):
        lang_context, lang_mask = self._language_context(language, language_mask)
        noisy, velocity, timesteps = self._flow_targets(actions)
        state_tokens = self._state_tokens(observations, future, state)
        action_tokens = self._action_tokens(noisy, timesteps)
        context_condition = self._context_condition(
            actions.shape[0], "idm", actions.dtype, actions.device
        )
        action_condition = self._action_condition(timesteps, "idm", actions.dtype)
        _, output = self.trunk(
            state_tokens,
            action_tokens,
            context_condition,
            action_condition,
            lang_context=lang_context,
            lang_mask=lang_mask,
        )
        prediction = self.action_decoder(
            output[:, self.num_action_register_tokens :]
        )
        return F.mse_loss(prediction, velocity)

    def forward_fdm(
        self, observations, actions, future, language=None, language_mask=None, state=None
    ):
        if observations.shape[1] != self.num_future_tokens:
            raise ValueError("Observation token count does not match future slots")
        lang_context, lang_mask = self._language_context(language, language_mask)
        state_tokens = self._state_tokens(observations, state=state)
        timesteps = self._clean_timestep(actions.shape[0], actions.device)
        action_tokens = self._action_tokens(actions, timesteps)
        context_condition = self._context_condition(
            actions.shape[0], "fdm", actions.dtype, actions.device
        )
        action_condition = self._action_condition(timesteps, "fdm", actions.dtype)
        current_rows = observations.shape[1] + int(self.use_state_condition)
        state_output, _ = self.trunk(
            state_tokens,
            action_tokens,
            context_condition,
            action_condition,
            lang_context=lang_context,
            lang_mask=lang_mask,
            n_state_ctx_rows=current_rows,
        )
        prediction = self.state_decoder(
            state_output[:, -self.num_future_tokens :]
        )
        return self._future_loss(prediction, future)

    def forward(
        self,
        vl_embs,
        actions,
        state=None,
        lang_embs=None,
        language_encoder_attention_mask=None,
        future_vl_embs_cond=None,
        future_vl_embs_target=None,
        mot_task_mode=None,
        repeated_steps=1,
        **unused,
    ):
        del unused
        mode = mot_task_mode or "idm_fdm"
        if mode not in self.TASK_MODES:
            raise ValueError(f"Unsupported objective {mode!r}")
        if mode == "policy":
            loss = self.forward_policy(
                vl_embs,
                actions,
                lang_embs,
                language_encoder_attention_mask,
                state,
                repeated_steps,
            )
            return {
                "policy_loss": loss,
                "action_loss": loss,
                "task_mode_id": torch.tensor(0, device=vl_embs.device),
            }

        condition = (
            future_vl_embs_cond
            if future_vl_embs_cond is not None
            else future_vl_embs_target
        )
        target = (
            future_vl_embs_target
            if future_vl_embs_target is not None
            else condition
        )
        if condition is None:
            raise ValueError("Stage 1 objectives require future observations")

        if mode == "idm":
            inverse = self.forward_idm(
                vl_embs,
                actions,
                condition,
                lang_embs,
                language_encoder_attention_mask,
                state,
            )
            return {
                "idm_loss": inverse,
                "action_loss": inverse,
                "task_mode_id": torch.tensor(1, device=vl_embs.device),
            }
        if mode == "fdm":
            future = self.forward_fdm(
                vl_embs,
                actions,
                target,
                lang_embs,
                language_encoder_attention_mask,
                state,
            )
            return {
                "fdm_loss": future,
                "action_loss": future,
                "task_mode_id": torch.tensor(2, device=vl_embs.device),
            }

        inverse = self.forward_idm(
            vl_embs,
            actions,
            condition,
            lang_embs,
            language_encoder_attention_mask,
            state,
        )
        future = self.forward_fdm(
            vl_embs,
            actions,
            target,
            lang_embs,
            language_encoder_attention_mask,
            state,
        )
        total = self.idm_loss_weight * inverse + self.fdm_loss_weight * future
        return {
            "idm_loss": inverse,
            "fdm_loss": future,
            "action_loss": total,
            "task_mode_id": torch.tensor(-2, device=vl_embs.device),
        }

    @torch.no_grad()
    def eval_future_latent(
        self,
        vl_embs,
        future,
        lang_embs=None,
        language_encoder_attention_mask=None,
        state=None,
    ):
        language, mask = self._language_context(
            lang_embs, language_encoder_attention_mask
        )
        state_tokens = self._state_tokens(vl_embs, state=state)
        condition = self._context_condition(
            vl_embs.shape[0], "fdm", vl_embs.dtype, vl_embs.device
        )
        action_condition = torch.zeros_like(condition)
        state_output, _ = self.trunk(
            state_tokens,
            None,
            condition,
            action_condition,
            lang_context=language,
            lang_mask=mask,
        )
        prediction = self.state_decoder(
            state_output[:, -self.num_future_tokens :]
        )
        return self._future_loss(prediction, future)

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs,
        state=None,
        lang_embs=None,
        language_encoder_attention_mask=None,
    ):
        language, mask = self._language_context(
            lang_embs, language_encoder_attention_mask
        )
        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=vl_embs.device,
            dtype=vl_embs.dtype,
        )
        context_condition = self._context_condition(
            batch_size, "policy", vl_embs.dtype, vl_embs.device
        )
        state_tokens = self._state_tokens(vl_embs, state=state)
        cache, _ = self.trunk.build_ctx_cache(
            state_tokens,
            context_condition,
            lang_context=language,
            lang_mask=mask,
        )
        step_size = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            timesteps = torch.full(
                (batch_size,),
                int(step * self.num_timestep_buckets / self.num_inference_timesteps),
                device=vl_embs.device,
                dtype=torch.long,
            )
            action_tokens = self._action_tokens(actions, timesteps)
            action_condition = self._action_condition(
                timesteps, "policy", vl_embs.dtype
            )
            output = self.trunk.forward_action_only(
                action_tokens, action_condition, cache
            )
            velocity = self.action_decoder(
                output[:, self.num_action_register_tokens :]
            )
            actions = actions + step_size * velocity
        return actions
