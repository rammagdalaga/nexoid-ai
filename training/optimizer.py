"""
APEXAI — Optimizer
AdamW with weight decay only on weight matrices (not biases / norms).
"""

import torch
from models.config import ModelConfig


def make_optimizer(model, cfg: ModelConfig):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or any(x in name for x in ["norm", "bias", "rope"]):
            no_decay.append(param)
        else:
            decay.append(param)

    n_d  = sum(p.numel() for p in decay)
    n_nd = sum(p.numel() for p in no_decay)
    print(f"[Optimizer] decay={n_d:,}  no-decay={n_nd:,}")

    groups = [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"fused": True} if torch.cuda.is_available() else {}
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2), **kwargs
    )