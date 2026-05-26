"""
APEXAI — Code Generation
Loads a trained checkpoint and generates code.
Supports temperature, top-k, top-p (nucleus) sampling, and KV cache.
"""

import os
import sys
import glob
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.config       import ModelConfig, estimate_param_count
from models.transformer  import GPT
from tokenizer.tokenizer import CodeTokenizer
from training.security_reasoning import analyze_security_reasoning


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    if not files:
        raise FileNotFoundError(
            f"No checkpoints in '{ckpt_dir}'. Run training first."
        )
    return files[-1]


def load_model(ckpt_path, device):
    print(f"[Generate] Loading: {ckpt_path}")
    if os.path.isdir(ckpt_path):
        # Sharded checkpoint directory
        rank = int(os.environ.get("LOCAL_RANK", 0))
        shard_path = os.path.join(ckpt_path, f"rank_{rank}.pt")
        ckpt = torch.load(shard_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ModelConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Generate] step={ckpt['step']}  "
          f"val_loss={ckpt['val_loss']:.4f}  "
          f"params={model.num_params()/1e6:.1f}M  "
          f"ctx={cfg.block_size}  max_seq={cfg.max_seq_len}")
    return model, cfg


def generate(prompt, model, tokenizer, cfg, device,
             max_new_tokens=256, temperature=0.8,
             top_k=40, top_p=0.95, use_cache=True):
    ids = tokenizer.encode(prompt)
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )
    new_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(new_ids)


def interactive_loop(model, tokenizer, cfg, device):
    print("\n" + "═" * 58)
    print("  APEXAI — Code Generation Agent")
    print(f"  Context: {cfg.block_size}  Max seq: {cfg.max_seq_len}")
    print("  Enter a prompt.  Type 'quit' to exit.")
    print("═" * 58)

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye]")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("[Bye]")
            break

        result = generate(prompt, model, tokenizer, cfg, device)
        print("\n" + "─" * 50)
        print(prompt + result)
        print("─" * 50)

def generate_security_analysis(code_snippet: str) -> dict:
    """Inference-side 4-stage defensive security analysis."""
    reasoning = analyze_security_reasoning(code_snippet)
    return reasoning.stage_outputs
