"""
APEXAI — Training Script
Features:
  - Cosine LR schedule with linear warmup
  - Gradient accumulation (cfg.grad_accum_steps)
  - Gradient clipping
  - Mixed precision (AMP) on CUDA
  - Checkpoint save & resume (including sharded)
  - Live loss display with tqdm
  - Distributed training support (DeepSpeed, FSDP, DDP)
  - Distributed DataLoader wrapper
  - Wandb logging integration (optional)
  - Gradient checkpointing
  - Long context training support
  - Cloud-first: distributed only on cloud GPUs

RULES:
  - DeepSpeed/FSDP only activated when explicitly configured
  - Single GPU Colab fallback with zero distributed overhead
  - All heavy dependencies are optional imports
"""

import os
import sys
import math
import time
import json
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.config      import ModelConfig
from models.transformer import GPT
from training.dataset   import make_loader
from training.optimizer import make_optimizer


# ── Optional dependencies ────────────────────

def _try_import_wandb():
    """Try to import wandb. Returns module or None."""
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def _try_import_deepspeed():
    """Try to import deepspeed. Returns module or None."""
    try:
        import deepspeed
        return deepspeed
    except ImportError:
        return None


# ── LR schedule ──────────────────────────────
def get_lr(step, cfg):
    if step < cfg.warmup_iters:
        return cfg.learning_rate * max(step, 1) / cfg.warmup_iters
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    t = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * t))


# ── Evaluation ───────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, cfg, device, use_amp):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_iters:
            break
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ── Checkpoints ──────────────────────────────
def save_checkpoint(model, optimizer, step, val_loss, cfg, is_distributed=False):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if cfg.sharded_checkpoints and is_distributed:
        # Save sharded checkpoint per GPU
        path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:06d}")
        os.makedirs(path, exist_ok=True)
        torch.save({
            "step":      step,
            "model":     model.state_dict(),
            "val_loss":  val_loss,
            "config":    cfg.__dict__,
        }, os.path.join(path, f"rank_{torch.distributed.get_rank()}.pt"))
        if torch.distributed.get_rank() == 0:
            print(f"\n  [ckpt shard] saved → {path}/  val_loss={val_loss:.4f}")
    else:
        path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:06d}.pt")
        torch.save({
            "step":      step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict() if cfg.save_optimizer_state else None,
            "val_loss":  val_loss,
            "config":    cfg.__dict__,
        }, path)
        print(f"\n  [ckpt] saved → {path}  val_loss={val_loss:.4f}")
    return path


def load_checkpoint(path, model, optimizer, device):
    if os.path.isdir(path):
        # Sharded checkpoint directory
        rank = int(os.environ.get("LOCAL_RANK", 0))
        shard_path = os.path.join(path, f"rank_{rank}.pt")
        ckpt = torch.load(shard_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Resume] step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    return ckpt["step"]


# ── Main ─────────────────────────────────────
def train(cfg: ModelConfig, resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    grad_accum_steps = getattr(cfg, "grad_accum_steps", 1)

    # ── Distributed setup ──
    is_main = True
    if cfg.distributed_backend != "none" and torch.cuda.device_count() > 1:
        from training.distributed import setup_distributed, get_world_info, print_distributed_info
        world_info = setup_distributed()
        is_main = world_info["is_main"]
        if is_main:
            print_distributed_info(cfg)
        # Override device for distributed
        if world_info["is_distributed"]:
            device = torch.device("cuda", world_info["local_rank"])
            use_amp = True
    else:
        world_info = {"is_distributed": False, "world_size": 1, "rank": 0}

    print(f"[APEXAI] device={device}  amp={use_amp}  "
          f"grad_accum={grad_accum_steps}  "
          f"effective_batch={cfg.batch_size * grad_accum_steps * world_info['world_size']}")

    # sync vocab_size from manifest
    manifest_path = os.path.join(cfg.data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        real_vocab = m.get("vocab_size", cfg.vocab_size)
        if real_vocab != cfg.vocab_size:
            if is_main:
                print(f"[APEXAI] vocab_size {cfg.vocab_size} → {real_vocab}")
            cfg.vocab_size = real_vocab

    # data
    train_loader = make_loader(cfg.data_dir, "train", cfg.block_size, cfg.batch_size)
    val_loader   = make_loader(cfg.data_dir, "val",   cfg.block_size, cfg.batch_size)

    # model + optimizer
    model = GPT(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # ── DeepSpeed init (if configured) ──
    deepspeed_engine = None
    if cfg.distributed_backend == "deepspeed":
        ds = _try_import_deepspeed()
        if ds is not None:
            from training.deepspeed_config import generate_deepspeed_config
            ds_config = generate_deepspeed_config(cfg)
            model, optimizer, _, _ = ds.initialize(
                model=model,
                optimizer=optimizer,
                config=ds_config,
                model_parameters=model.parameters(),
            )
            deepspeed_engine = model  # model is now the DeepSpeed engine
            if is_main:
                print("[DeepSpeed] Engine initialized ✓")
        else:
            if is_main:
                print("[DeepSpeed] Not installed. Falling back to standard training.")

    start_step = 0
    if resume_path:
        if deepspeed_engine is not None:
            # DeepSpeed handles its own checkpoint loading
            start_step = load_checkpoint(resume_path, deepspeed_engine, None, device)
        else:
            start_step = load_checkpoint(resume_path, model, optimizer, device)

    # logging
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "training.log"), "a", encoding="utf-8")

    def log(msg):
        if is_main:
            print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n{'='*65}")
    log(f"[APEXAI] Training  params={model.num_params()/1e6:.1f}M  "
        f"vocab={cfg.vocab_size}  ctx={cfg.block_size}  "
        f"max_seq={cfg.max_seq_len}  "
        f"iters={cfg.max_iters}  device={device}  "
        f"dist={world_info['is_distributed']}  "
        f"effective_batch={cfg.batch_size * grad_accum_steps * world_info['world_size']}")

    if cfg.use_moe:
        log(f"         MoE enabled: experts={cfg.moe_num_experts} top_k={cfg.moe_top_k}")
    if cfg.rope_scaling != "none":
        log(f"         RoPE scaling: {cfg.rope_scaling} factor={cfg.rope_scale_factor}")
    if cfg.use_grad_checkpoint:
        log(f"         Gradient checkpointing: segments={cfg.grad_checkpoint_segments}")
    if cfg.use_flash_attn:
        log(f"         Flash Attention: enabled")
    if cfg.distributed_backend != "none":
        log(f"         Distributed: {cfg.distributed_backend} "
            f"ZeRO={cfg.zero_stage} offload={cfg.cpu_offload}")
    log(f"{'='*65}")

    # ── Wandb ──
    wandb_run = None
    wandb_mod = _try_import_wandb()
    if wandb_mod and is_main:
        try:
            wandb_run = wandb_mod.init(
                project="apexai",
                config={
                    "params_m": model.num_params() / 1e6,
                    "vocab_size": cfg.vocab_size,
                    "block_size": cfg.block_size,
                    "max_seq_len": cfg.max_seq_len,
                    "n_layer": cfg.n_layer,
                    "n_embd": cfg.n_embd,
                    "n_head": cfg.n_head,
                    "batch_size": cfg.batch_size,
                    "grad_accum": cfg.grad_accum_steps,
                    "learning_rate": cfg.learning_rate,
                    "max_iters": cfg.max_iters,
                    "use_moe": cfg.use_moe,
                    "rope_scaling": cfg.rope_scaling,
                    "distributed": cfg.distributed_backend,
                    "use_grad_checkpoint": cfg.use_grad_checkpoint,
                    "use_flash_attn": cfg.use_flash_attn,
                },
                save_code=True,
            )
            log("[Wandb] Logging enabled ✓")
        except Exception as e:
            log(f"[Wandb] Init failed: {e} (skipping)")

    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    t0 = time.time()

    pbar = tqdm(range(start_step, cfg.max_iters),
                desc="Training", unit="step", dynamic_ncols=True,
                disable=not is_main)

    for step in pbar:
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss.backward()

            accum_loss += loss.item()

        if deepspeed_engine is not None:
            # DeepSpeed handles gradient clipping and stepping
            deepspeed_engine.step()
        elif use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.1e}")

        if step % 100 == 0 and is_main:
            elapsed = time.time() - t0
            ms_step = elapsed * 1000 / max(step - start_step + 1, 1)
            log(f"step {step:6d} | loss {accum_loss:.4f} | "
                f"lr {lr:.2e} | {ms_step:.0f} ms/step")

        if step > 0 and step % cfg.eval_interval == 0 and is_main:
            val_loss = evaluate(model, val_loader, cfg, device, use_amp)
            log(f"  ↳ val_loss={val_loss:.4f}")
            if wandb_run:
                wandb_run.log({
                    "train_loss": accum_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "step": step,
                }, step=step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, cfg,
                                is_distributed=world_info["is_distributed"])

        if step > 0 and step % cfg.checkpoint_every == 0 and is_main:
            save_checkpoint(model, optimizer, step, accum_loss, cfg,
                            is_distributed=world_info["is_distributed"])

    pbar.close()
    log(f"\n[APEXAI] Training complete.  best_val_loss={best_val_loss:.4f}")
    log_file.close()

    if wandb_run:
        wandb_run.finish()

    # ── Cleanup distributed ──
    if world_info["is_distributed"]:
        from training.distributed import cleanup_distributed
        cleanup_distributed()