"""
APEXAI — Model Configuration
Single source of truth for all hyperparameters.
Presets: small (~15M), medium (~85M), large (~307M), XL (~1B), XXL (~3B)

FIX LOG:
  - small_config: batch_size 32→8, block_size 512→256, grad_accum_steps=4
  - medium_config: batch_size 16→4, block_size 512→256, grad_accum_steps=4
  - large_config: block_size 1024→512, grad_accum_steps=8
  - Added grad_accum_steps field to ModelConfig
  - Added Mixture of Experts (MoE) configuration support
  - Added extended context length options (up to 32k tokens)
  - Added RoPE scaling/interpolation for long context (4k/8k/16k/32k)
  - Added Flash Attention toggle with hardware detection
  - Added gradient checkpointing toggle
  - Added DeepSpeed/FSDP distributed training options
  - Added multilingual tokenizer config
  - Added SEO token specializations

VRAM estimates on T4 (15GB):
  small  — batch=8,  block=256 → ~3–4 GB  ✓ safe
  medium — batch=4,  block=256 → ~6–8 GB  ✓ safe
  large  — batch=4,  block=512 → ~14 GB   ⚠ tight, use Colab Pro
  XL     — batch=2,  block=1024 → ~24 GB   ✗ requires A100/H100
  XXL    — batch=1,  block=2048 → ~48 GB   ✗ requires multi-GPU

Long context VRAM estimates (A100 80GB):
  8k  — large:  batch=2 → ~28 GB  ✓ safe
  16k — large:  batch=1 → ~32 GB  ✓ safe
  32k — XL:     batch=1 → ~64 GB  ⚠ tight, use A100 80GB

To further reduce fragmentation if you still see OOM:
  import os; os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

For multi-GPU training, use torch.distributed with appropriate backend.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    # ── Vocabulary & Sequence ────────────────
    vocab_size:  int = 32000    # matches The Stack BPE tokenizer
    block_size:  int = 256      # base context length

    # ── Architecture ─────────────────────────
    n_embd:  int = 512          # embedding dimension
    n_head:  int = 8            # attention heads
    n_layer: int = 8            # transformer blocks
    ffn_mult: int = 4           # FFN hidden = ffn_mult * n_embd

    # ── Long Context / RoPE Scaling ──────────
    rope_base:        float = 10000.0   # base frequency for RoPE
    rope_scaling:     Literal["linear", "dynamic", "ntk", "none"] = "none"
    rope_scale_factor: float = 1.0      # scaling factor for long context (e.g. 2.0 for 2x)
    max_seq_len:      int = 4096        # maximum sequence length supported

    # ── Flash Attention ─────────────────────
    use_flash_attn: bool = False   # Use Flash Attention if available
    flash_attn_fallback: bool = True  # Fall back to vanilla attention if Flash unavailable

    # ── Mixture of Experts ──────────────────
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_loss_coef: float = 0.01
    moe_layer_interval: int = 2

    # ── Gradient Checkpointing ──────────────
    use_grad_checkpoint: bool = False
    grad_checkpoint_segments: int = 4

    # ── Distributed Training ────────────────
    distributed_backend: Literal["deepspeed", "fsdp", "ddp", "none"] = "none"
    zero_stage: int = 2                    # DeepSpeed ZeRO stage (0, 1, 2, 3)
    cpu_offload: bool = False              # CPU offload for optimizer states
    activation_checkpointing: bool = False  # Activation checkpointing (FSDP)
    shard_strategy: str = "full"           # Sharding strategy for FSDP

    # ── Multilingual ───────────────────────
    multilingual_tokens: list = field(default_factory=lambda: [
        "<|en|>", "<|fil|>", "<|ja|>", "<|ko|>", "<|zh|>", "<|es|>"
    ])
    multilingual_training: bool = False

    # ── SEO Specialization ──────────────────
    seo_tokens: list = field(default_factory=lambda: [
        "<|seo|>", "<|meta|>", "<|schema|>", "<|accessibility|>", "<|lighthouse|>"
    ])
    seo_training: bool = False

    # ── Regularization ──────────────────────
    embd_dropout:  float = 0.1
    attn_dropout:  float = 0.1
    resid_dropout: float = 0.1

    # ── Training ────────────────────────────
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0
    batch_size:     int   = 8
    grad_accum_steps: int = 4
    max_iters:      int   = 20_000
    eval_interval:  int   = 500
    eval_iters:     int   = 50
    warmup_iters:   int   = 500
    lr_decay_iters: int   = 20_000
    min_lr:         float = 3e-5

    # ── Checkpointing ───────────────────────
    checkpoint_dir:           str = "checkpoints"
    checkpoint_every:         int = 2000
    sharded_checkpoints:      bool = False
    save_optimizer_state:     bool = True

    # ── Data ─────────────────────────────────
    data_dir: str = "data/processed"

    # ── Inference ────────────────────────────
    kv_cache_enabled: bool = True
    speculative_decoding: bool = False
    speculative_tokens: int = 5
    quantization_bits: Optional[int] = None  # None, 8, or 4
    max_batch_tokens: int = 4096


def compute_rope_scaling(cfg: ModelConfig) -> tuple:
    """
    Compute RoPE scaling parameters based on config.
    Returns (base_freq, scale_factor) for the rotary embeddings.

    Supports:
      - linear scaling: uniform frequency adjustment
      - dynamic NTK-aware scaling: frequency interpolation
      - none: no scaling (original RoPE)

    Reference: https://arxiv.org/abs/2306.15595 (YaRN)
    """
    target_seq_len = cfg.max_seq_len
    base_seq_len = cfg.block_size

    if base_seq_len >= target_seq_len or cfg.rope_scaling == "none":
        return cfg.rope_base, 1.0

    scale = target_seq_len / base_seq_len

    if cfg.rope_scaling == "linear":
        return cfg.rope_base * scale, scale
    elif cfg.rope_scaling == "dynamic":
        # NTK-aware scaling: adjust base frequency
        new_base = cfg.rope_base * (scale ** (cfg.n_embd / (cfg.n_embd - 2)))
        return new_base, 1.0
    elif cfg.rope_scaling == "ntk":
        # NTK-aware with partial linear scaling
        scale_factor = 0.1 * math.log(scale) + 1.0
        new_base = cfg.rope_base * (scale ** (cfg.n_embd / (cfg.n_embd - 2)))
        return new_base, scale_factor
    return cfg.rope_base, 1.0


@dataclass
class ContextPreset:
    """Predefined context length configuration."""
    length: int
    label: str
    recommended_configs: list
    vram_a100_80gb: str
    vram_h100: str


CONTEXT_PRESETS = {
    "4k":  ContextPreset(4096,  "Standard",  ["small", "medium", "large"], "~8-12 GB", "~6-10 GB"),
    "8k":  ContextPreset(8192,  "Extended",  ["large", "xl"],              "~16-28 GB", "~12-22 GB"),
    "16k": ContextPreset(16384, "Long",      ["large", "xl"],              "~24-40 GB", "~18-32 GB"),
    "32k": ContextPreset(32768, "Very Long", ["xl", "xxl"],                "~48-64 GB", "~36-52 GB"),
}


def get_context_preset(length: int) -> Optional[ContextPreset]:
    """Get context preset by length value."""
    for preset in CONTEXT_PRESETS.values():
        if preset.length == length:
            return preset
    return None


def estimate_memory(cfg: ModelConfig) -> dict:
    """
    Estimate memory usage for a given configuration.
    Returns dict with estimated VRAM usage in GB.
    """
    n_params = estimate_param_count(cfg)
    bytes_per_param = 4  # float32

    # Model weights
    weights_gb = n_params * bytes_per_param / 1e9

    # Optimizer states (AdamW: 2 states + 1 momentum)
    optimizer_gb = weights_gb * 3 if cfg.save_optimizer_state else 0

    # Activations (rough estimate)
    max_seq = max(cfg.block_size, cfg.max_seq_len)
    activation_gb = (
        cfg.batch_size * max_seq * cfg.n_embd * cfg.n_layer * 4 / 1e9
        * (1 + cfg.grad_accum_steps * 0.5)  # accumulation overhead
    )

    # Gradient checkpointing reduces activation memory
    if cfg.use_grad_checkpoint:
        activation_gb /= cfg.grad_checkpoint_segments

    total_gb = weights_gb + optimizer_gb + activation_gb

    return {
        "params_m": n_params / 1e6,
        "weights_gb": round(weights_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "total_gb": round(total_gb, 2),
        "recommended_gpu": recommend_gpu(total_gb),
    }


def estimate_param_count(cfg: ModelConfig) -> int:
    """Estimate parameter count for a configuration."""
    n_embd = cfg.n_embd
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    head_dim = n_embd // n_head
    ffn_hidden = cfg.ffn_mult * n_embd

    # Token embedding + LM head (weight tied, count once)
    emb_params = cfg.vocab_size * n_embd

    # Per layer: attention (QKV + output) + FFN (gate, up, down)
    attn_params = 3 * n_embd * n_embd + n_embd * n_embd  # QKV + proj
    ffn_params = 2 * n_embd * ffn_hidden + ffn_hidden * n_embd  # gate/up + down
    layer_params = attn_params + ffn_params + n_embd * 2  # RMS norms

    # MoE adds extra experts
    if cfg.use_moe:
        moe_layers = n_layer // (cfg.moe_layer_interval or 1)
        non_moe_layers = n_layer - moe_layers
        # Each MoE layer has num_experts FFNs (but sparse routing)
        # Count all expert parameters for total model capacity
        expert_count = cfg.moe_num_experts
        moe_extra = moe_layers * (expert_count - 1) * ffn_params
        layer_params = (
            non_moe_layers * layer_params
            + moe_layers * (attn_params + n_embd * 2)  # attention part
            + moe_layers * expert_count * ffn_params   # all experts
        )
    else:
        layer_params = n_layer * layer_params

    return emb_params + layer_params


def recommend_gpu(total_gb: float) -> str:
    """Recommend a GPU based on estimated memory."""
    if total_gb < 15:
        return "T4 (Colab Free) ✓"
    elif total_gb < 24:
        return "T4 (tight) ⚠"
    elif total_gb < 40:
        return "A100 40GB (Colab Pro) ✓"
    elif total_gb < 80:
        return "A100 80GB ✓"
    else:
        return "Multi-GPU A100/H100 cluster required ✗"


# ── Presets ──────────────────────────────────

def small_config() -> ModelConfig:
    return ModelConfig(
        n_embd=512, n_head=8, n_layer=8,
        block_size=256, max_seq_len=4096,
        rope_scaling="none", rope_scale_factor=1.0,
        batch_size=8, grad_accum_steps=4,
        max_iters=20_000, warmup_iters=500, lr_decay_iters=20_000,
    )


def medium_config() -> ModelConfig:
    return ModelConfig(
        n_embd=768, n_head=12, n_layer=12,
        block_size=256, max_seq_len=4096,
        rope_scaling="none", rope_scale_factor=1.0,
        batch_size=4, grad_accum_steps=4,
        max_iters=40_000, warmup_iters=1000,
        lr_decay_iters=40_000, learning_rate=2e-4,
    )


def large_config() -> ModelConfig:
    return ModelConfig(
        n_embd=1024, n_head=16, n_layer=24,
        block_size=512, max_seq_len=8192,
        rope_scaling="dynamic", rope_scale_factor=2.0,
        batch_size=4, grad_accum_steps=8,
        max_iters=100_000, warmup_iters=2000,
        lr_decay_iters=100_000, learning_rate=1e-4,
        vocab_size=32000,
    )


def xl_config() -> ModelConfig:
    return ModelConfig(
        n_embd=2048, n_head=32, n_layer=48,
        block_size=1024, max_seq_len=16384,
        rope_scaling="ntk", rope_scale_factor=4.0,
        batch_size=2, grad_accum_steps=4,
        max_iters=200_000, warmup_iters=2000,
        lr_decay_iters=200_000, learning_rate=5e-5,
        vocab_size=64000,
        use_moe=True, moe_num_experts=16,
        moe_top_k=4, moe_layer_interval=2,
        use_grad_checkpoint=True,
        flash_attn_fallback=True,
    )


def xxl_config() -> ModelConfig:
    return ModelConfig(
        n_embd=4096, n_head=64, n_layer=48,
        block_size=2048, max_seq_len=32768,
        rope_scaling="ntk", rope_scale_factor=8.0,
        batch_size=1, grad_accum_steps=4,
        max_iters=400_000, warmup_iters=4000,
        lr_decay_iters=400_000, learning_rate=3e-5,
        vocab_size=128000,
        use_moe=True, moe_num_experts=32,
        moe_top_k=4, moe_layer_interval=1,
        use_grad_checkpoint=True,
        flash_attn_fallback=True,
        distributed_backend="deepspeed",
        zero_stage=3, cpu_offload=True,
        sharded_checkpoints=True,
    )


def longcontext_4k_config() -> ModelConfig:
    """4k context — good for medium-length files, works on T4."""
    base = large_config()
    base.block_size = 4096
    base.max_seq_len = 4096
    base.rope_scaling = "linear"
    base.rope_scale_factor = 8.0
    base.batch_size = 2
    base.grad_accum_steps = 8
    return base


def longcontext_8k_config() -> ModelConfig:
    """8k context — good for full functions/classes, requires A100."""
    base = xl_config()
    base.block_size = 8192
    base.max_seq_len = 8192
    base.rope_scaling = "dynamic"
    base.rope_scale_factor = 8.0
    base.batch_size = 1
    base.grad_accum_steps = 8
    return base


def longcontext_16k_config() -> ModelConfig:
    """16k context — full file understanding, requires A100."""
    base = xl_config()
    base.block_size = 16384
    base.max_seq_len = 16384
    base.rope_scaling = "ntk"
    base.rope_scale_factor = 16.0
    base.batch_size = 1
    base.grad_accum_steps = 4
    base.use_flash_attn = True
    return base


def longcontext_32k_config() -> ModelConfig:
    """32k context — repository-level understanding, multi-GPU only."""
    base = xxl_config()
    base.block_size = 32768
    base.max_seq_len = 32768
    base.rope_scaling = "ntk"
    base.rope_scale_factor = 32.0
    base.batch_size = 1
    base.grad_accum_steps = 2
    base.use_flash_attn = True
    return base