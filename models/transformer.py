"""
APEXAI — Transformer Architecture
Upgrades over baseline GPT:
  - Rotary Positional Embeddings (RoPE) with NTK-aware scaling
  - SwiGLU activation in FFN
  - Pre-RMSNorm instead of LayerNorm
  - Dropout on embeddings, attention, residuals
  - Weight tying (token embed <-> lm_head)
  - Flash Attention (with fallback)
  - KV Cache for inference
  - Gradient checkpointing
  - Mixture of Experts (MoE) support
  - Long context support (up to 32k tokens)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.config import ModelConfig, compute_rope_scaling


# ─────────────────────────────────────────────
# Utility: Check Flash Attention availability
# ─────────────────────────────────────────────
_FLASH_AVAIL = None

def flash_attn_available() -> bool:
    global _FLASH_AVAIL
    if _FLASH_AVAIL is not None:
        return _FLASH_AVAIL
    try:
        import flash_attn
        _FLASH_AVAIL = True
    except ImportError:
        _FLASH_AVAIL = False
    return _FLASH_AVAIL


# ─────────────────────────────────────────────
# RMSNorm  (faster than LayerNorm, used in LLaMA)
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale


# ─────────────────────────────────────────────
# Rotary Positional Embedding (RoPE) with Scaling
# Supports original RoPE, linear scaling, and NTK-aware scaling
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0,
                 scaling: str = "none", scale_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling = scaling
        self.scale_factor = scale_factor

        # Compute base inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cache = None
        self._sin_cache = None
        self._cache_seq_len = 0

    def _build_cache(self, seq_len: int, device: torch.device):
        """Build cos/sin cache for given sequence length with scaling."""
        t = torch.arange(seq_len, device=device).float()

        if self.scaling == "linear" and self.scale_factor > 1.0:
            t = t / self.scale_factor
        elif self.scaling == "dynamic":
            # Dynamic NTK: adjust frequencies, not positions
            # Already handled via base frequency adjustment in inv_freq
            pass
        elif self.scaling == "ntk":
            # NTK-aware: partial position interpolation
            t = t / (self.scale_factor ** 0.5)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self._cos_cache = emb.cos()[None, None, :, :].to(device)
        self._sin_cache = emb.sin()[None, None, :, :].to(device)
        self._cache_seq_len = seq_len

    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self._cache_seq_len:
            self._build_cache(seq_len, device)
        return (self._cos_cache[:, :, :seq_len, :],
                self._sin_cache[:, :, :seq_len, :])


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), \
           (k * cos) + (rotate_half(k) * sin)


# ─────────────────────────────────────────────
# KV Cache for inference
# ─────────────────────────────────────────────
class KVCache:
    """Stores and manages key-value cache for autoregressive generation."""
    def __init__(self, max_batch_size: int = 1, max_seq_len: int = 32768,
                 dtype=torch.float16, device="cuda"):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self._cache = {}  # layer_idx -> (k, v)
        self.seen_tokens = 0

    def allocate(self, num_layers: int, n_head: int, head_dim: int):
        """Pre-allocate cache buffers."""
        shape = (self.max_batch_size, n_head, self.max_seq_len, head_dim)
        for i in range(num_layers):
            k = torch.zeros(shape, dtype=self.dtype, device=self.device)
            v = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self._cache[i] = (k, v)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append new keys/values to cache."""
        k_cache, v_cache = self._cache[layer_idx]
        B, H, T, D = k.shape
        n_new = T
        start = self.seen_tokens
        k_cache[:, :, start:start+n_new, :] = k
        v_cache[:, :, start:start+n_new, :] = v
        return k_cache[:, :, :start+n_new, :], v_cache[:, :, :start+n_new, :]

    def reset(self):
        """Clear the cache."""
        self._cache = {}
        self.seen_tokens = 0

    @property
    def is_allocated(self) -> bool:
        return len(self._cache) > 0


# ─────────────────────────────────────────────
# Causal Self-Attention with Flash & KV Cache
# ─────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_head    = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.n_embd    = cfg.n_embd
        self.scale     = math.sqrt(self.head_size)
        self.use_flash = cfg.use_flash_attn and flash_attn_available()
        self.flash_fallback = cfg.flash_attn_fallback

        if self.use_flash:
            try:
                from flash_attn import flash_attn_func
                self._flash_fn = flash_attn_func
            except ImportError:
                self.use_flash = False

        self.c_attn    = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj    = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=False)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.res_drop  = nn.Dropout(cfg.resid_dropout)

        # RoPE with scaling
        rope_base, rope_scale = compute_rope_scaling(cfg)
        self.rope = RotaryEmbedding(
            self.head_size,
            base=rope_base,
            scaling=cfg.rope_scaling,
            scale_factor=rope_scale,
        )

        # Causal mask for vanilla attention
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
              .view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

        # KV cache reference (set externally during generation)
        self.kv_cache = None
        self.layer_idx = 0

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)

        # KV Cache update during generation
        if self.kv_cache is not None and self.kv_cache.is_allocated:
            k, v = self.kv_cache.update(self.layer_idx, k, v)

        if past_kv is not None:
            # For generate() with external cache
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
            T_full = k.shape[2]
        else:
            T_full = T

        if self.use_flash and T_full > 1:
            # Flash Attention (requires contiguous tensors in specific layout)
            q_f = q.transpose(1, 2).contiguous()  # (B, T, H, D)
            k_f = k.transpose(1, 2).contiguous()
            v_f = v.transpose(1, 2).contiguous()
            out = self._flash_fn(q_f, k_f, v_f, dropout_p=0.0, causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        else:
            # Vanilla attention
            att = (q @ k.transpose(-2, -1)) / self.scale
            if T_full <= self.mask.shape[-1]:
                att = att.masked_fill(
                    self.mask[:, :, :T_full, :T_full] == 0,
                    float("-inf")
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.res_drop(self.c_proj(out))


# ─────────────────────────────────────────────
# SwiGLU Feed-Forward
# ─────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.n_embd
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden,     cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.resid_dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ─────────────────────────────────────────────
# Mixture of Experts (MoE) Feed-Forward
# ─────────────────────────────────────────────
class MoEFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.moe_num_experts
        self.top_k       = cfg.moe_top_k
        self.loss_coef   = cfg.moe_loss_coef

        hidden = cfg.ffn_mult * cfg.n_embd
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.n_embd, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, cfg.n_embd, bias=False),
            ) for _ in range(self.num_experts)
        ])

        self.router = nn.Linear(cfg.n_embd, self.num_experts, bias=False)
        self.drop   = nn.Dropout(cfg.resid_dropout)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        router_probs  = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)

        expert_usage = router_probs.mean(dim=0)
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = (expert_usage - target_usage).pow(2).sum() * self.loss_coef

        final_output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_mask = (top_k_indices == expert_idx)
            prob_mask = top_k_probs * expert_mask.float()
            routing_weights = prob_mask.sum(dim=-1)
            expert_output = self.experts[expert_idx](x_flat[mask])
            final_output[mask] += expert_output * routing_weights[mask].unsqueeze(-1)

        output = final_output.view(B, T, C)
        return self.drop(output), load_balancing_loss


# ─────────────────────────────────────────────
# Transformer Block (supports MoE + gradient checkpointing)
# ─────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn  = CausalSelfAttention(cfg)
        self.attn.layer_idx = layer_idx
        self.norm2 = RMSNorm(cfg.n_embd)

        use_moe = cfg.use_moe
        if use_moe and cfg.moe_layer_interval > 0:
            use_moe = (layer_idx % cfg.moe_layer_interval == 0)

        self.ffn = MoEFFN(cfg) if use_moe else SwiGLUFFN(cfg)
        self.is_moe = use_moe
        self.use_grad_checkpoint = cfg.use_grad_checkpoint

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.is_moe:
            ffn_out, moe_loss = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x, moe_loss
        else:
            x = x + self.ffn(self.norm2(x))
            return x

    def forward(self, x):
        if self.use_grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)


# ─────────────────────────────────────────────
# APEXAI GPT — Full Model
# ─────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop    = nn.Dropout(cfg.embd_dropout),
            blocks  = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layer)]),
            norm_f  = RMSNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying
        self.transformer.tok_emb.weight = self.lm_head.weight

        # KV cache for generation
        self.kv_cache = None

        self._init_weights()
        print(f"[APEXAI] GPT ready — {self.num_params()/1e6:.1f}M params  "
              f"vocab={cfg.vocab_size}  ctx={cfg.block_size}  "
              f"max_seq={cfg.max_seq_len}  "
              f"layers={cfg.n_layer}  heads={cfg.n_head}  embd={cfg.n_embd}  "
              f"moe={cfg.use_moe}  rope_scaling={cfg.rope_scaling}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def enable_kv_cache(self, batch_size: int = 1, device: str = "cuda"):
        """Enable KV cache for inference."""
        self.kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=min(self.cfg.max_seq_len, 32768),
            device=device,
        )
        self.kv_cache.allocate(
            self.cfg.n_layer,
            self.cfg.n_head,
            self.cfg.n_embd // self.cfg.n_head,
        )
        # Attach cache to all attention layers
        for block in self.transformer.blocks:
            block.attn.kv_cache = self.kv_cache

    def disable_kv_cache(self):
        """Disable KV cache."""
        self.kv_cache = None
        for block in self.transformer.blocks:
            block.attn.kv_cache = None

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= max(self.cfg.block_size, self.cfg.max_seq_len), \
            f"Input length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.transformer.drop(self.transformer.tok_emb(idx))

        total_moe_loss = 0.0
        moe_n_layers = 0

        for block in self.transformer.blocks:
            if block.is_moe:
                x, moe_loss = block(x)
                total_moe_loss += moe_loss
                moe_n_layers += 1
            else:
                x = block(x)

        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            loss = ce_loss
            if moe_n_layers > 0:
                avg_moe_loss = total_moe_loss / moe_n_layers
                loss = ce_loss + self.cfg.moe_loss_coef * avg_moe_loss
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens,
                 temperature=0.8, top_k=40, top_p=0.95,
                 use_cache=True):
        """Generates tokens autoregressively with optional KV cache."""
        self.eval()

        if use_cache:
            self.enable_kv_cache(batch_size=idx.shape[0], device=idx.device)

        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -min(self.cfg.block_size, self.cfg.max_seq_len):]

                if use_cache and self.kv_cache is not None and self.kv_cache.seen_tokens > 0:
                    # During cached generation, only pass the single new token
                    idx_cond = idx[:, -1:]

                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)

                # top-k
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # top-p (nucleus)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter_(
                        1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_t], dim=1)

        finally:
            if use_cache:
                self.disable_kv_cache()

        return idx