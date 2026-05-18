"""
APEXAI — DeepSpeed Configuration Generator
Generates DeepSpeed JSON config files for ZeRO Stage 2/3 training.

This config file is loaded by DeepSpeed for multi-GPU training on cloud.
It is NOT auto-loaded — must be explicitly passed as --deepspeed_config.

RULES:
  - Never auto-install DeepSpeed locally
  - DeepSpeed is for cloud GPU clusters (A100/H100)
  - Colab fallback: use standard PyTorch training
"""

import json
import os
from models.config import ModelConfig


def generate_deepspeed_config(cfg: ModelConfig, output_path: str = None) -> dict:
    """
    Generate a complete DeepSpeed JSON configuration from ModelConfig.

    Args:
        cfg: Model configuration
        output_path: Optional file path to save the JSON config

    Returns:
        DeepSpeed config dict
    """
    # ── ZeRO optimization ──
    zero_opt = {
        "stage": cfg.zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    }

    if cfg.zero_stage >= 2:
        zero_opt["round_robin_gradients"] = True

    if cfg.zero_stage >= 3:
        zero_opt.update({
            "stage3_max_live_parameters": 3e8,
            "stage3_prefetch_bucket_size": 3e8,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_gather_16bit_weights_on_model_save": True,
            "sub_group_size": 1e9,
        })

    if cfg.cpu_offload:
        zero_opt["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        if cfg.zero_stage >= 3:
            zero_opt["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    # ── Scheduler (WarmupCosineLR) ──
    scheduler = {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": cfg.min_lr,
            "warmup_max_lr": cfg.learning_rate,
            "warmup_num_steps": cfg.warmup_iters,
            "total_num_steps": cfg.max_iters,
        }
    }

    # ── Complete config ──
    ds_config = {
        "train_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "gradient_clipping": cfg.grad_clip,
        "zero_optimization": zero_opt,
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
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
        "scheduler": scheduler,
        "activation_checkpointing": {
            "partition_activations": cfg.activation_checkpointing,
            "cpu_checkpointing": cfg.cpu_offload,
            "number_checkpoints": cfg.grad_checkpoint_segments,
            "profile": False,
        },
        "wall_clock_breakdown": False,
        "steps_per_print": 100,
        "comms_logger": {
            "enabled": False,
            "verbose": False,
            "prof_all": False,
            "debug": False,
        },
        "data_types": {
            "grad_accum_dtype": "fp32",
        },
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
        },
        "tensorboard": {
            "enabled": False,
        },
        "csv_monitor": {
            "enabled": False,
        },
        "wandb": {
            "enabled": True,
            "project": "apexai",
            "group": f"zero{cfg.zero_stage}_bs{cfg.batch_size}_ctx{cfg.block_size}",
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        print(f"[DeepSpeed] Config saved to {output_path}")

    return ds_config


def generate_launcher_script(config_name: str, num_gpus: int = 8) -> str:
    """
    Generate a shell script to launch distributed training on cloud.

    Returns script content as string (not auto-executed).
    """
    return f"""#!/bin/bash
# APEXAI — Distributed Training Launcher
# Generated for config={config_name}, GPUs={num_gpus}
# WARNING: Run only on cloud GPU clusters (A100/H100)
# DO NOT run on local machine without explicit permission

# Set environment
export HF_TOKEN="${{HF_TOKEN:-}}"
export WANDB_API_KEY="${{WANDB_API_KEY:-}}"
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

# Launch with DeepSpeed
deepspeed \\
    --num_gpus={num_gpus} \\
    --master_port=29500 \\
    main.py train \\
    --config {config_name} \\
    --deepspeed_config deepspeed_config.json

echo "Training completed."
"""


# ── Colab setup helper ──

def colab_setup_single_gpu():
    """
    Colab-friendly setup: single GPU, no distributed overhead.
    Returns the device to use for training.
    """
    import torch
    if torch.cuda.is_available():
        print("[Colab] GPU detected — using single GPU mode")
        return torch.device("cuda")
    print("[Colab] No GPU detected — using CPU (warning: very slow)")
    return torch.device("cpu")


# ── DeepSpeed diagnostic ──

def check_deepspeed_installed() -> dict:
    """
    Check if DeepSpeed is installed and report version.
    Returns dict with status info.
    Does NOT auto-install.
    """
    result = {
        "installed": False,
        "version": None,
        "error": None,
    }

    try:
        import deepspeed
        result["installed"] = True
        result["version"] = deepspeed.__version__
    except ImportError:
        result["error"] = (
            "DeepSpeed not installed.\n"
            "Install with: pip install deepspeed\n"
            "Or use Colab fallback mode with standard PyTorch."
        )
    except Exception as e:
        result["error"] = str(e)

    return result