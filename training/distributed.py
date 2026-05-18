"""
APEXAI — Distributed Training Utilities
Supports DeepSpeed ZeRO (Stage 2/3), FSDP, and DDP backends.
All heavy infrastructure targets cloud GPUs — never auto-installed locally.

RULES:
  - DeepSpeed is optional — only imported when explicitly configured
  - All distributed backends require user configuration, never auto-enabled
  - Colab fallback: single-GPU mode with no distributed overhead
"""

import os
import sys
import torch
import logging

logger = logging.getLogger("apexai.distributed")


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def is_deepspeed_available() -> bool:
    """Check if DeepSpeed is installed (without importing)."""
    try:
        import deepspeed
        return True
    except ImportError:
        return False


def get_device() -> torch.device:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_world_info() -> dict:
    """Get distributed world information.
    Returns dict with rank, world_size, local_rank for the current process.
    Falls back to single-process defaults when not distributed.
    """
    info = {
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "is_distributed": False,
        "is_main": True,
    }

    if "RANK" in os.environ:
        info["rank"] = int(os.environ["RANK"])
        info["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
        info["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))
        info["is_distributed"] = info["world_size"] > 1
        info["is_main"] = info["rank"] == 0

    return info


def setup_distributed(backend: str = "nccl") -> dict:
    """
    Initialize distributed process group.
    Returns world info dict.

    Args:
        backend: 'nccl' (GPU), 'gloo' (CPU fallback)

    Should be called before model creation.
    """
    world_info = get_world_info()

    if not world_info["is_distributed"]:
        logger.info("Single-process mode (no distributed setup needed)")
        return world_info

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
        )
        torch.cuda.set_device(world_info["local_rank"])
        logger.info(
            f"Distributed initialized: rank={world_info['rank']}/"
            f"{world_info['world_size']}  local_rank={world_info['local_rank']}"
        )
    except Exception as e:
        logger.warning(f"Distributed init failed: {e}. Falling back to single GPU.")
        world_info["is_distributed"] = False
        world_info["world_size"] = 1

    return world_info


def cleanup_distributed():
    """Clean up distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# ── DeepSpeed Configuration ──────────────────

def create_deepspeed_config(cfg) -> dict:
    """
    Create a DeepSpeed configuration dict from ModelConfig.

    Supports ZeRO Stage 0, 2, and 3 with optional CPU offload.
    Returns a dict ready to be passed to deepspeed.initialize().

    Cloud-first: CPU offload targets remote CPU nodes, not local machine.
    """
    zero_config = {
        "stage": cfg.zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
    }

    # Stage 3 specific
    if cfg.zero_stage == 3:
        zero_config.update({
            "stage3_max_live_parameters": 1e9,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_gather_16bit_weights_on_model_save": True,
        })

    # CPU offload
    if cfg.cpu_offload:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "ratio": 0.5,
        }
        if cfg.zero_stage == 3:
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    ds_config = {
        "train_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "gradient_clipping": cfg.grad_clip,
        "zero_optimization": zero_config,
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {
            "enabled": False,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg.learning_rate,
                "betas": [cfg.beta1, cfg.beta2],
                "eps": 1e-8,
                "weight_decay": cfg.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": cfg.min_lr,
                "warmup_max_lr": cfg.learning_rate,
                "warmup_num_steps": cfg.warmup_iters,
                "total_num_steps": cfg.max_iters,
            },
        },
        "activation_checkpointing": {
            "partition_activations": cfg.activation_checkpointing,
            "cpu_checkpointing": cfg.cpu_offload,
            "number_checkpoints": cfg.grad_checkpoint_segments,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
        "wall_clock_breakdown": False,
        "steps_per_print": 100,
        "tensorboard": {
            "enabled": False,
        },
    }

    return ds_config


# ── FSDP Configuration ──────────────────────

def create_fsdp_config(cfg) -> dict:
    """
    Create FSDP configuration from ModelConfig.

    Sharding strategies:
      - 'full': Full shard (ZeRO-3 equivalent)
      - 'hybrid': Hybrid shard (FSDP + DDP)
      - 'no_shard': DDP (ZeRO-1 equivalent)
    """
    from torch.distributed.fsdp import (
        ShardingStrategy,
        BackwardPrefetch,
        MixedPrecision,
    )

    shard_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }

    strategy = shard_map.get(cfg.shard_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    config = {
        "sharding_strategy": strategy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "mixed_precision": mp_policy,
        "cpu_offload": cfg.cpu_offload,
        "use_orig_params": True,
        "sync_module_states": True,
        "forward_prefetch": True,
        "limit_all_gathers": True,
    }

    return config


# ── Distributed DataLoader wrapper ──────────

class DistributedDataLoaderWrapper:
    """Wraps a DataLoader for distributed sampling."""

    def __init__(self, dataloader, rank: int = 0, world_size: int = 1):
        self.dataloader = dataloader
        self.rank = rank
        self.world_size = world_size
        self._iter = None

    def __iter__(self):
        self._iter = enumerate(self.dataloader)
        return self

    def __next__(self):
        while True:
            try:
                idx, batch = next(self._iter)
                if idx % self.world_size == self.rank:
                    return batch
            except StopIteration:
                raise StopIteration

    def __len__(self):
        return len(self.dataloader) // self.world_size


# ── Launch Utilities ────────────────────────

def get_launch_command(config_name: str, num_gpus: int = None) -> str:
    """
    Generate the CLI command to launch distributed training.

    Examples:
        torchrun --nproc_per_node=4 main.py train --config xl
        deepspeed --num_gpus=8 main.py train --config xxl
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    return (
        f"torchrun --nproc_per_node={num_gpus} main.py train "
        f"--config {config_name}"
    )


def print_distributed_info(cfg):
    """Print distributed training configuration summary."""
    info = get_world_info()
    lines = [
        "\n" + "=" * 55,
        "  APEXAI — Distributed Training Config",
        "=" * 55,
        f"  Backend:     {cfg.distributed_backend}",
        f"  GPUs:        {info['world_size']}",
        f"  ZeRO Stage:  {cfg.zero_stage}",
        f"  CPU Offload: {cfg.cpu_offload}",
        f"  Grad CKPT:   {cfg.use_grad_checkpoint}",
        f"  Act CKPT:    {cfg.activation_checkpointing}",
        f"  Batch/GPU:   {cfg.batch_size}",
        f"  Grad Accum:  {cfg.grad_accum_steps}",
        f"  Eff Batch:   {cfg.batch_size * cfg.grad_accum_steps * info['world_size']}",
        "=" * 55,
    ]
    for line in lines:
        print(line)