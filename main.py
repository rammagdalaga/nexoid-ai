"""
APEXAI — Main Entry Point

Commands:
    python main.py tokenize                        -- train BPE tokenizer on data/raw/*.py
    python main.py preprocess                      -- tokenize corpus into data/processed/
    python main.py train [--config small/medium/large/xl/xxl]  -- train the model
    python main.py train --resume checkpoints/ckpt_*.pt       -- resume training
    python main.py generate                        -- interactive code generation (REPL)
    python main.py generate --prompt "def foo():"  -- generate from a prompt
    python main.py serve                           -- start REST inference server
    python main.py evaluate [--benchmark]          -- run benchmarks
    python main.py profile                         -- show GPU/memory profile
    python main.py memory                          -- show GPU memory summary
    python main.py estimate [--config xl]          -- estimate memory for a config
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


HELP = """
╔══════════════════════════════════════════════════════╗
║            APEXAI  —  Code AI Agent                  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  TRAINING PIPELINE:                                  ║
║                                                      ║
║  Step 1:  python main.py tokenize                    ║
║           Train BPE tokenizer on data/raw/*.py       ║
║                                                      ║
║  Step 2:  python main.py preprocess                  ║
║           Tokenize corpus → data/processed/          ║
║                                                      ║
║  Step 3:  python main.py train                       ║
║           Train the model (default: small config)    ║
║           --config small | medium | large | xl | xxl ║
║           --resume checkpoints/ckpt_XXXXXX.pt        ║
║                                                      ║
║  INFERENCE:                                          ║
║                                                      ║
║  Step 4:  python main.py generate                    ║
║           Interactive code generation (REPL)         ║
║           --prompt "def merge_sort(arr):"            ║
║           --max_tokens 256                           ║
║           --temp 0.8  --top_k 40                     ║
║                                                      ║
║  Step 4b: python main.py serve                       ║
║           Start REST API server (--port 8080)        ║
║                                                      ║
║  UTILITIES:                                          ║
║                                                      ║
║  python main.py evaluate [--benchmark humaneval]     ║
║  python main.py profile                              ║
║  python main.py memory                               ║
║  python main.py estimate --config xl                 ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""


def cmd_tokenize(argv):
    """Train the BPE tokenizer on .py files in data/raw/."""
    from tokenizer.tokenizer import CodeTokenizer
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(prog="main.py tokenize")
    parser.add_argument("--data_dir",   default=os.path.join(ROOT, "data", "raw"),
                        help="Folder containing training files")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--output",     default=os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    args = parser.parse_args(argv)

    py_files = list(Path(args.data_dir).rglob("*.py"))
    if not py_files:
        print(f"[Error] No .py files found in: {args.data_dir}")
        print(f"        Make sure your training files are inside data/raw/")
        sys.exit(1)

    print(f"[Tokenizer] Found {len(py_files)} files in {args.data_dir}")
    tok = CodeTokenizer()
    tok.train([str(p) for p in py_files], vocab_size=args.vocab_size)
    tok.save(args.output)


def cmd_preprocess(argv):
    """Tokenize the corpus into binary shards for training."""
    from training.dataset import preprocess
    import argparse

    parser = argparse.ArgumentParser(prog="main.py preprocess")
    parser.add_argument("--data_dir",   default=os.path.join(ROOT, "data", "raw"))
    parser.add_argument("--out_dir",    default=os.path.join(ROOT, "data", "processed"))
    parser.add_argument("--tokenizer",  default=os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    parser.add_argument("--shard_size", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    preprocess(args.data_dir, args.out_dir, args.tokenizer, args.shard_size)
    print(f"\n[APEXAI] Preprocessing complete. Run 'python main.py train' to train.")


def cmd_train(argv):
    """Train the model from scratch (or resume from checkpoint)."""
    from training.train import train
    from models.config import (
        small_config, medium_config, large_config,
        xl_config, xxl_config,
        longcontext_4k_config, longcontext_8k_config,
        longcontext_16k_config, longcontext_32k_config,
    )
    import argparse

    parser = argparse.ArgumentParser(prog="main.py train")
    parser.add_argument("--config", choices=[
        "small", "medium", "large", "xl", "xxl",
        "4k", "8k", "16k", "32k",
    ], default="small")
    parser.add_argument("--resume",    default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--deepspeed_config", default=None,
                        help="Path to DeepSpeed JSON config (optional)")
    args = parser.parse_args(argv)

    configs = {
        "small": small_config, "medium": medium_config,
        "large": large_config, "xl": xl_config, "xxl": xxl_config,
        "4k": longcontext_4k_config, "8k": longcontext_8k_config,
        "16k": longcontext_16k_config, "32k": longcontext_32k_config,
    }
    cfg = configs[args.config]()

    cfg.data_dir       = os.path.join(ROOT, "data", "processed")
    cfg.checkpoint_dir = os.path.join(ROOT, "checkpoints")

    if args.max_iters:
        cfg.max_iters = args.max_iters

    train(cfg, resume_path=args.resume)


def cmd_generate(argv):
    """Run the code generation agent (interactive or single prompt)."""
    import argparse
    import torch
    from models.config       import ModelConfig
    from models.transformer  import GPT
    from tokenizer.tokenizer import CodeTokenizer
    from inference.generate  import find_latest_checkpoint, load_model, generate, interactive_loop

    parser = argparse.ArgumentParser(prog="main.py generate")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint (default: latest in checkpoints/)")
    parser.add_argument("--tokenizer",  default=os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    parser.add_argument("--prompt",     default=None)
    parser.add_argument("--max_tokens", type=int,   default=256)
    parser.add_argument("--temp",       type=float, default=0.8)
    parser.add_argument("--top_k",      type=int,   default=40)
    parser.add_argument("--no-cache",   action="store_true",
                        help="Disable KV cache")
    args = parser.parse_args(argv)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or find_latest_checkpoint(os.path.join(ROOT, "checkpoints"))
    model, cfg = load_model(ckpt_path, device)
    tokenizer  = CodeTokenizer.load(args.tokenizer)

    use_cache = not args.no_cache

    if args.prompt:
        result = generate(
            args.prompt, model, tokenizer, cfg, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            use_cache=use_cache,
        )
        print("\n" + args.prompt + result)
    else:
        interactive_loop(model, tokenizer, cfg, device)


def cmd_serve(argv):
    """Start the REST inference server."""
    import argparse
    from inference.server import start_server

    parser = argparse.ArgumentParser(prog="main.py serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    start_server(args.host, args.port, args.checkpoint)


def cmd_evaluate(argv):
    """Run evaluation benchmarks."""
    import argparse
    import torch
    from evaluation.benchmarks import get_benchmark, run_benchmark
    from evaluation.humaneval import run_humaneval
    from evaluation.seo_eval import run_seo_benchmark
    from models.config import ModelConfig
    from models.transformer import GPT
    from tokenizer.tokenizer import CodeTokenizer
    from inference.generate import find_latest_checkpoint, load_model, generate

    parser = argparse.ArgumentParser(prog="main.py evaluate")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "seo", "all"],
                        default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or find_latest_checkpoint(os.path.join(ROOT, "checkpoints"))
    model, cfg = load_model(ckpt_path, device)
    tokenizer = CodeTokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))

    def gen_fn(prompt):
        return generate(prompt, model, tokenizer, cfg, device,
                        max_new_tokens=args.max_tokens)

    if args.benchmark in ("humaneval", "all"):
        print("\n" + "=" * 55)
        result = run_humaneval(gen_fn)
        print(f"[HumanEval] Accuracy: {result.accuracy:.1%}  "
              f"Passed: {result.passed}/{result.total}")
        if result.errors:
            for e in result.errors[:3]:
                print(f"  Error: {e}")

    if args.benchmark in ("mbpp", "all"):
        print("\n" + "=" * 55)
        cases = get_benchmark("mbpp")
        result = run_benchmark(gen_fn, cases, name="mbpp")
        print(f"[MBPP] Accuracy: {result.accuracy:.1%}  "
              f"Passed: {result.passed}/{result.total}")

    if args.benchmark in ("seo", "all"):
        print("\n" + "=" * 55)
        result = run_seo_benchmark(gen_fn)
        print(f"[SEO] SEO Score: {result.accuracy:.1%}  "
              f"Passed: {result.passed}/{result.total}")


def cmd_profile(argv):
    """Show GPU training profile."""
    from training.profiler import Profiler, estimate_flops
    import argparse

    parser = argparse.ArgumentParser(prog="main.py profile")
    parser.add_argument("--config", default="small")
    args = parser.parse_args(argv)

    from models.config import ModelConfig
    configs = {
        "small": lambda: ModelConfig(),
        "xl": lambda: ModelConfig(n_embd=2048, n_head=32, n_layer=48),
    }
    cfg = configs.get(args.config, configs["small"])()

    flops = estimate_flops(cfg)
    print(f"\n[APEXAI] FLOPs estimate for '{args.config}' config:")
    print(f"  Forward:   {flops['forward_tflop']:.2f} TFLOPs")
    print(f"  Backward:  {flops['backward_tflop']:.2f} TFLOPs")
    print(f"  Total:     {flops['total_tflop']:.2f} TFLOPs")

    from models.config import estimate_memory
    mem = estimate_memory(cfg)
    print(f"\n  Memory Estimate:")
    print(f"  Params:    {mem['params_m']:.0f}M")
    print(f"  Weights:   {mem['weights_gb']:.1f} GB")
    print(f"  Opt States:{mem['optimizer_gb']:.1f} GB")
    print(f"  Activations:{mem['activation_gb']:.1f} GB")
    print(f"  Total:     {mem['total_gb']:.1f} GB")
    print(f"  GPU:       {mem['recommended_gpu']}")


def cmd_memory(argv):
    """Show GPU memory summary."""
    from utils.memory import print_memory_summary
    print_memory_summary()


def cmd_estimate(argv):
    """Estimate memory for a configuration."""
    import argparse
    parser = argparse.ArgumentParser(prog="main.py estimate")
    parser.add_argument("--config", default="small",
                        choices=["small", "medium", "large", "xl", "xxl"])
    args = parser.parse_args(argv)

    from models.config import (
        small_config, medium_config, large_config,
        xl_config, xxl_config, estimate_memory
    )
    configs = {
        "small": small_config, "medium": medium_config,
        "large": large_config, "xl": xl_config, "xxl": xxl_config,
    }
    cfg = configs[args.config]()
    mem = estimate_memory(cfg)

    print(f"\n[APEXAI] Memory Estimate — {args.config.upper()} Config")
    print(f"  Parameters:     {mem['params_m']:.0f}M")
    print(f"  Weights:        {mem['weights_gb']:.1f} GB")
    print(f"  Optimizer:      {mem['optimizer_gb']:.1f} GB")
    print(f"  Activations:    {mem['activation_gb']:.1f} GB")
    print(f"  Total VRAM:     {mem['total_gb']:.1f} GB")
    print(f"  Recommended GPU: {mem['recommended_gpu']}")


# ── Command dispatch ─────────────────────────

COMMANDS = {
    "tokenize":   cmd_tokenize,
    "preprocess": cmd_preprocess,
    "train":      cmd_train,
    "generate":   cmd_generate,
    "serve":      cmd_serve,
    "evaluate":   cmd_evaluate,
    "profile":    cmd_profile,
    "memory":     cmd_memory,
    "estimate":   cmd_estimate,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(HELP)
        sys.exit(0)

    command  = sys.argv[1]
    leftover = sys.argv[2:]

    print(f"\n[APEXAI] Running: {command}\n")
    COMMANDS[command](leftover)