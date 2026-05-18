"""
APEXAI — Byte-Level BPE Tokenizer
Uses the HuggingFace `tokenizers` library (fast Rust implementation).
Trains a GPT-style byte-level BPE tokenizer on Python source files.

Usage:
    python tokenizer/tokenizer.py --data_file data/raw/python_code.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Special tokens ──────────────────────────
PAD_TOKEN  = "<pad>"
UNK_TOKEN  = "<unk>"
BOS_TOKEN  = "<s>"
EOS_TOKEN  = "</s>"
SEP_TOKEN  = "<|endoftext|>"

SPECIAL_TOKENS = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]


# ─────────────────────────────────────────────
# Tokenizer wrapper
# ─────────────────────────────────────────────
class CodeTokenizer:
    """
    Wraps HuggingFace ByteLevelBPETokenizer so the rest of APEXAI
    can call .encode() / .decode() exactly as before.
    """

    def __init__(self):
        self._tok = None   # set after train() or load()

    # ── training ─────────────────────────────
    def train(self, files: list, vocab_size: int = 32000,
              min_frequency: int = 2, verbose: bool = True):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                "Run:  pip install tokenizers"
            )

        print(f"[Tokenizer] Training ByteLevel BPE on {len(files)} file(s)  "
              f"vocab_size={vocab_size}")

        tok = ByteLevelBPETokenizer()
        tok.train(
            files         = [str(f) for f in files],
            vocab_size    = vocab_size,
            min_frequency = min_frequency,
            special_tokens = SPECIAL_TOKENS,
        )
        self._tok = tok
        print(f"[Tokenizer] Done. Vocab size = {self._tok.get_vocab_size()}")

    # ── encode / decode ───────────────────────
    def encode(self, text: str) -> list:
        if self._tok is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        return self._tok.encode(text).ids

    def decode(self, ids: list) -> str:
        if self._tok is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        return self._tok.decode(ids, skip_special_tokens=True)

    # ── save / load ───────────────────────────
    def save(self, path: str):
        """Save vocab.json + merges.txt to a directory."""
        os.makedirs(path, exist_ok=True)
        if self._tok is None:
            raise RuntimeError("Nothing to save — train first.")
        self._tok.save_model(path)
        # also save a metadata file so we know vocab_size
        meta = {
            "vocab_size":     self._tok.get_vocab_size(),
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Tokenizer] Saved to {path}/  "
              f"(vocab={self._tok.get_vocab_size()})")

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        """Load from a directory containing vocab.json + merges.txt."""
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError("Run:  pip install tokenizers")

        vocab_file  = os.path.join(path, "vocab.json")
        merges_file = os.path.join(path, "merges.txt")

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"vocab.json not found in '{path}'.\n"
                f"Run tokenizer training first."
            )

        tok = cls()
        tok._tok = ByteLevelBPETokenizer(
            vocab  = vocab_file,
            merges = merges_file,
        )
        tok._tok.add_special_tokens(SPECIAL_TOKENS)
        meta_path = os.path.join(path, "meta.json")
        vs = tok._tok.get_vocab_size()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                vs = json.load(f).get("vocab_size", vs)
        print(f"[Tokenizer] Loaded from {path}/  (vocab={vs})")
        return tok

    # ── properties ───────────────────────────
    @property
    def vocab_size(self) -> int:
        if self._tok is None:
            return 0
        return self._tok.get_vocab_size()

    @property
    def eos_id(self) -> int:
        if self._tok is None:
            return 2
        return self._tok.token_to_id(EOS_TOKEN) or 2

    @property
    def bos_id(self) -> int:
        if self._tok is None:
            return 0
        return self._tok.token_to_id(BOS_TOKEN) or 0

    @property
    def pad_id(self) -> int:
        if self._tok is None:
            return 1
        return self._tok.token_to_id(PAD_TOKEN) or 1

    # for HuggingFace GPT2TokenizerFast compatibility (used in generate)
    def batch_encode(self, texts: list) -> list:
        return [self.encode(t) for t in texts]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train APEXAI tokenizer")
    parser.add_argument("--data_file",  required=True,
                        help="Path to python_code.txt (all code concatenated)")
    parser.add_argument("--out_dir",    default="tokenizer/saved",
                        help="Where to save vocab.json + merges.txt")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_freq",   type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"[Error] File not found: {args.data_file}")
        sys.exit(1)

    tok = CodeTokenizer()
    tok.train(
        files         = [args.data_file],
        vocab_size    = args.vocab_size,
        min_frequency = args.min_freq,
    )
    tok.save(args.out_dir)