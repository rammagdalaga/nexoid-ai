"""
APEXAI — Dataset Pipeline
Downloads from The Stack (bigcode/the-stack) via HuggingFace streaming,
supports multiple languages for training a coding-first intelligence system.

Steps:
  1. download()   — streams files from The Stack subsets → data/raw/{lang}_code.txt
  2. preprocess() — tokenizes code files → data/processed/shard_XXXX.pt

Enhanced for multi-language streaming with balanced sampling and quality filtering.
"""

import os
import sys
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hashlib
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────
# Configuration for Multi-Language Streaming
# ─────────────────────────────────────────────

# Language configurations for The Stack dataset
LANGUAGE_CONFIGS = {
    "python": {
        "data_dir": "data/python",
        "patterns": ["def ", "class ", "import ", "from "],
        "weight": 0.35,  # Primary focus - maintain Python strength
    },
    "javascript": {
        "data_dir": "data/javascript",
        "patterns": ["function ", "const ", "let ", "var ", "=>"],
        "weight": 0.25,  # Web development focus
    },
    "typescript": {
        "data_dir": "data/typescript",
        "patterns": ["interface ", "type ", "function ", "const ", "let "],
        "weight": 0.15,  # Type-safe JavaScript
    },
    "html": {
        "data_dir": "data/html",
        "patterns": ["<html", "<div", "<script", "<!DOCTYPE"],
        "weight": 0.15,  # Web markup
    },
    "css": {
        "data_dir": "data/css",
        "patterns": ["{", "}", ":", ";", "@media"],
        "weight": 0.10,  # Styling
    }
}

# SEO-specific patterns for enhanced web development understanding
SEO_PATTERNS = [
    "meta name=", "meta property=", "structured-data", "schema.org",
    "rel=canonical", "h1", "h2", "h3", "alt=", "title>",
    "viewport", "mobile-friendly", "loading=", "async"
]

# Minimum quality thresholds per language
QUALITY_THRESHOLDS = {
    "python":    {"min_chars": 200, "max_chars": 10000, "required_patterns": 2},
    "javascript": {"min_chars": 150, "max_chars": 8000,  "required_patterns": 2},
    "typescript": {"min_chars": 150, "max_chars": 8000,  "required_patterns": 2},
    "html":      {"min_chars": 100, "max_chars": 5000,   "required_patterns": 2},
    "css":       {"min_chars": 50,  "max_chars": 3000,   "required_patterns": 2}
}

# ─────────────────────────────────────────────
# Tokenizer and Utilities
# ─────────────────────────────────────────────

# FIX: os.environ.get() takes the KEY name as a string, not the token value itself.
# The old code did: os.environ.get("hf_HSMUsMcJk...") which looks up a key that
# doesn't exist, so TOKEN was always None.
# Correct pattern: store your token under a real env var name like HF_TOKEN,
# then read it with os.environ.get("HF_TOKEN").
TOKEN = os.environ.get("HF_TOKEN")


def get_language_weight(language):
    """Get sampling weight for a language."""
    return LANGUAGE_CONFIGS.get(language, {}).get("weight", 0.1)


def calculate_code_quality_score(code, language):
    """Calculate a quality score for code based on language-specific heuristics."""
    config = QUALITY_THRESHOLDS.get(language, QUALITY_THRESHOLDS["python"])

    # Length check
    if not (config["min_chars"] <= len(code) <= config["max_chars"]):
        return 0.0

    # Pattern matching
    patterns = LANGUAGE_CONFIGS.get(language, {}).get("patterns", [])
    pattern_matches = sum(1 for pattern in patterns if pattern in code)
    pattern_score = min(pattern_matches / max(len(patterns), 1), 1.0)

    # SEO bonus for web languages
    seo_bonus = 0.0
    if language in ["html", "css", "javascript", "typescript"]:
        seo_matches = sum(1 for pattern in SEO_PATTERNS if pattern in code.lower())
        seo_bonus = min(seo_matches * 0.1, 0.3)  # Max 30% bonus

    # Structure indicators (brackets, semicolons, etc.)
    structure_chars = code.count('{') + code.count('}') + code.count('(') + code.count(')') + code.count(';')
    structure_score = min(structure_chars / 100, 0.5)  # Normalize

    # Combine scores
    total_score = (pattern_score * 0.4) + (structure_score * 0.3) + (seo_bonus * 0.3)
    return min(total_score, 1.0)


def detect_language_from_content(code):
    """Detect programming language from code content."""
    scores = {}
    for lang, config in LANGUAGE_CONFIGS.items():
        patterns = config["patterns"]
        score = sum(1 for pattern in patterns if pattern in code)
        scores[lang] = score

    if max(scores.values()) == 0:
        return "unknown"
    return max(scores, key=scores.get)


def deduplicate_code(code, seen_hashes):
    """Check if code is duplicate using SHA-256 hashing."""
    code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
    if code_hash in seen_hashes:
        return True
    seen_hashes.add(code_hash)
    return False


# ─────────────────────────────────────────────
# Step 1 — Download from The Stack (Multi-Language)
# ─────────────────────────────────────────────

def download(
    out_dir:     str   = "data/raw",
    max_files_per_lang: int = 5_000,
    min_quality_score: float = 0.3,
    hf_token:    str   = None,
    languages:   list  = None,
):
    """
    Streams files from multiple language subsets of bigcode/the-stack on HuggingFace.
    Saves them separated by language with <|endoftext|> delimiters.

    Authentication:
      Option A — pass hf_token= directly to this function
      Option B — set HF_TOKEN environment variable before running
      Option C — run `huggingface-cli login` once (token cached by HF)

    Args:
        out_dir: Directory to save language-specific code files
        max_files_per_lang: Maximum files to download per language
        min_quality_score: Minimum quality threshold (0.0-1.0) for accepting code
        hf_token: HuggingFace token for authenticated access
        languages: List of languages to download (None for all configured)

    Returns:
        Dict mapping language codes to output file paths
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run:  pip install datasets")

    # FIX: resolve token with proper priority chain
    try:
        from huggingface_hub import get_token as _get_hf_token
        _cached = _get_hf_token()
    except Exception:
        _cached = None

    resolved_token = hf_token or TOKEN or _cached

    if resolved_token is None:
        print("[Dataset] WARNING: No HuggingFace token found.")
        print("          If the dataset is gated, set HF_TOKEN env var or run:")
        print("          huggingface-cli login")
    else:
        print("[Dataset] HuggingFace token resolved ✓")

    # Determine which languages to process
    if languages is None:
        languages = list(LANGUAGE_CONFIGS.keys())

    os.makedirs(out_dir, exist_ok=True)

    # Track downloaded files and duplicates per language
    results = {}
    total_seen_hashes = set()

    print(f"[Dataset] Streaming The Stack for languages: {', '.join(languages)}")
    print(f"          Target: {max_files_per_lang:,} files per language")
    print(f"          Min quality score: {min_quality_score}")
    print(f"          Output directory → {out_dir}")

    for language in languages:
        if language not in LANGUAGE_CONFIGS:
            print(f"[Dataset] WARNING: Unknown language '{language}', skipping")
            continue

        config = LANGUAGE_CONFIGS[language]
        out_file = os.path.join(out_dir, f"{language}_code.txt")

        print(f"\n[Dataset] Processing {language.upper()} ({config['data_dir']})")

        # Load dataset stream for this language
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=config["data_dir"],
            split="train",
            streaming=True,
            token=resolved_token,
        )

        saved   = 0
        skipped = 0
        lang_seen_hashes = set()

        with open(out_file, "w", encoding="utf-8", errors="ignore") as fout:
            bar = tqdm(total=max_files_per_lang, unit="files", desc=f"Downloading {language}")
            for sample in ds:
                code = sample.get("content", "")

                # Skip if too short/long or duplicate
                if deduplicate_code(code, lang_seen_hashes):
                    skipped += 1
                    continue

                # Calculate quality score
                quality_score = calculate_code_quality_score(code, language)
                if quality_score < min_quality_score:
                    skipped += 1
                    continue

                # Write code with language marker and end token
                fout.write(f"<|{language}|>\n")
                fout.write(code)
                fout.write("\n<|endoftext|>\n")
                saved += 1
                bar.update(1)

                if saved >= max_files_per_lang:
                    break
            bar.close()

        size_mb = os.path.getsize(out_file) / 1024 / 1024
        print(f"[Dataset] {language.upper()}: Saved {saved:,} files  ({size_mb:.1f} MB)  "
              f"skipped {skipped:,}")

        results[language] = out_file
        total_seen_hashes.update(lang_seen_hashes)

    print(f"\n[Dataset] Download complete. Total unique files processed: {sum(len(v) for v in results.values()):,}")
    return results


# ─────────────────────────────────────────────
# Step 2 — Tokenize → Shards (Multi-Language Aware)
# ─────────────────────────────────────────────
def preprocess(
    data_dir:      str,
    out_dir:       str,
    tokenizer_path: str,
    shard_size:    int = 2_000_000,
    languages:     list = None,
):
    """
    Reads language-specific code files, tokenizes with CodeTokenizer,
    and saves as flat LongTensor shards in out_dir.
    Maintains language markers for training awareness.
    """
    from tokenizer.tokenizer import CodeTokenizer

    os.makedirs(out_dir, exist_ok=True)
    tok = CodeTokenizer.load(tokenizer_path)

    # Determine which language files to process
    if languages is None:
        # Auto-detect language files in data_dir
        languages = []
        for filename in os.listdir(data_dir):
            if filename.endswith("_code.txt"):
                lang = filename.replace("_code.txt", "")
                if lang in LANGUAGE_CONFIGS:
                    languages.append(lang)

    print(f"[Dataset] Tokenizing files for languages: {', '.join(languages)}")

    all_tokens = []
    shard_idx  = 0
    language_token_counts = {lang: 0 for lang in languages}
    total_tokens = 0

    def flush(tokens, idx):
        arr  = torch.tensor(tokens, dtype=torch.long)
        path = os.path.join(out_dir, f"shard_{idx:04d}.pt")
        torch.save(arr, path)
        print(f"  [shard {idx:04d}] {len(tokens):,} tokens → {path}")
        return []

    CHUNK  = 500_000

    for language in languages:
        data_file = os.path.join(data_dir, f"{language}_code.txt")

        if not os.path.exists(data_file):
            print(f"[Dataset] WARNING: Data file not found for {language}: {data_file}")
            continue

        file_size = os.path.getsize(data_file) / 1024 / 1024
        print(f"[Dataset] Tokenizing {language} ({data_file})  ({file_size:.1f} MB)")

        buffer = ""

        with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
            pbar = tqdm(total=int(file_size), unit="MB", desc=f"Tokenizing {language}")
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                buffer += chunk
                parts  = buffer.split("<|endoftext|>")
                buffer = parts[-1]

                for doc in parts[:-1]:
                    if doc.strip():
                        # Extract language marker if present
                        lang_marker = ""
                        code_content = doc.strip()

                        if code_content.startswith("<|") and "|\n" in code_content:
                            end_marker = code_content.find("|\n")
                            lang_marker = code_content[2:end_marker]
                            code_content = code_content[end_marker + 2:]

                        # Tokenize the code content
                        ids = tok.encode(code_content)

                        # Prepend language marker tokens if we detected one
                        if lang_marker and lang_marker in LANGUAGE_CONFIGS:
                            # Add special language tokens (we'll need to extend tokenizer for this)
                            # For now, we'll rely on the model learning from context
                            pass

                        all_tokens.extend(ids)
                        language_token_counts[lang_marker if lang_marker in languages else language] += len(ids)
                        total_tokens += len(ids)

                if len(all_tokens) >= shard_size:
                    all_tokens = flush(all_tokens[:shard_size], shard_idx)
                    shard_idx += 1
                pbar.update(len(chunk) / 1024 / 1024)
            pbar.close()

        # Handle remaining buffer
        if buffer.strip():
            # Process any remaining content
            parts = buffer.split("<|endoftext|>")
            for doc in parts[:-1]:
                if doc.strip():
                    ids = tok.encode(doc.strip())
                    all_tokens.extend(ids)
                    total_tokens += len(ids)

        if all_tokens:
            flush(all_tokens, shard_idx)
            shard_idx += 1
            all_tokens = []

    n_shards = shard_idx
    if n_shards == 0:
        print("[Dataset] No tokens produced — check your data files.")
        return {}

    # Create train/validation split (90/10)
    split_at = max(1, int(n_shards * 0.9))
    train_shards = [f"shard_{i:04d}.pt" for i in range(split_at)]
    val_shards   = [f"shard_{i:04d}.pt" for i in range(split_at, n_shards)]

    manifest = {
        "train":        train_shards,
        "val":          val_shards,
        "vocab_size":   tok.vocab_size,
        "total_tokens": total_tokens,
        "language_distribution": language_token_counts,
        "languages": languages,
        "shards": n_shards
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Dataset] Done.")
    print(f"  total tokens : {total_tokens:,}")
    print(f"  shards       : {n_shards}  (train={len(train_shards)}, val={len(val_shards)})")
    print(f"  vocab size   : {tok.vocab_size}")
    print(f"  language distribution:")
    for lang, count in language_token_counts.items():
        percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
        print(f"    {lang}: {count:,} tokens ({percentage:.1f}%)")

    return manifest

# ─────────────────────────────────────────────
# PyTorch Dataset + DataLoader
# ─────────────────────────────────────────────
class CodeDataset(Dataset):
    def __init__(self, data_dir: str, split: str, block_size: int):
        self.block_size = block_size

        with open(os.path.join(data_dir, "manifest.json")) as f:
            manifest = json.load(f)

        shard_names = manifest[split]
        shards = [
            torch.load(os.path.join(data_dir, s), weights_only=True)
            for s in shard_names
        ]
        self.data     = torch.cat(shards, dim=0)
        self.n_chunks = (len(self.data) - 1) // block_size

        if self.n_chunks == 0:
            raise ValueError(
                f"Not enough tokens ({len(self.data)}) for "
                f"block_size={block_size}. Add more data or reduce block_size."
            )
        print(f"[Dataset] {split:5s}  tokens={len(self.data):,}  "
              f"chunks={self.n_chunks:,}")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        s = idx * self.block_size
        return (self.data[s     : s + self.block_size],
                self.data[s + 1 : s + self.block_size + 1])


def make_loader(data_dir, split, block_size, batch_size,
                num_workers=0) -> DataLoader:
    ds = CodeDataset(data_dir, split, block_size)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )