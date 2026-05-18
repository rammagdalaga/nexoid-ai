"""
APEXAI — Memory Utilities
GPU memory profiling, estimation, and optimization helpers.
"""

import os
import gc
import time
import torch
from typing import Dict, Optional


def get_gpu_info() -> Dict:
    """
    Get GPU information including memory stats.

    Returns dict with:
      - available: bool
      - name: GPU name
      - total_memory_gb: total VRAM in GB
      - free_memory_gb: free VRAM in GB
      - allocated_memory_gb: allocated VRAM in GB
      - cuda_version: CUDA version string
    """
    info = {
        "available": torch.cuda.is_available(),
        "name": None,
        "total_memory_gb": 0,
        "free_memory_gb": 0,
        "allocated_memory_gb": 0,
        "cuda_version": torch.version.cuda,
    }

    if info["available"]:
        info["name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

        if torch.cuda.is_allocated():
            info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / 1e9
            info["free_memory_gb"] = info["total_memory_gb"] - info["allocated_memory_gb"]
        else:
            info["free_memory_gb"] = info["total_memory_gb"]

    return info


def print_memory_summary():
    """Print a formatted summary of current GPU memory usage."""
    info = get_gpu_info()

    print("\n" + "=" * 50)
    print("  APEXAI — GPU Memory Summary")
    print("=" * 50)

    if not info["available"]:
        print("  GPU: Not available (CPU mode)")
        print("=" * 50)
        return

    print(f"  GPU:          {info['name']}")
    print(f"  CUDA:         {info['cuda_version']}")
    print(f"  Total VRAM:   {info['total_memory_gb']:.2f} GB")
    print(f"  Allocated:    {info['allocated_memory_gb']:.2f} GB")
    print(f"  Free:         {info['free_memory_gb']:.2f} GB")

    if info["total_memory_gb"] > 0:
        pct = (info["allocated_memory_gb"] / info["total_memory_gb"]) * 100
        print(f"  Utilization:  {pct:.1f}%")

    # Show top memory-consuming tensors
    if torch.cuda.is_allocated():
        print("\n  Top tensors by memory:")
        tensors = [
            (tensor_size(t), t.shape, t.dtype)
            for t in get_tensor_snapshots()
            if t.is_cuda
        ]
        tensors.sort(reverse=True)
        for size_bytes, shape, dtype in tensors[:5]:
            print(f"    {size_bytes / 1e6:.1f} MB  {shape}  {dtype}")

    print("=" * 50)


def get_tensor_snapshots() -> list:
    """Get a snapshot of all tracked tensors for analysis."""
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append(obj)
        except Exception:
            pass
    return tensors


def tensor_size(t: torch.Tensor) -> int:
    """Get the memory size of a tensor in bytes."""
    return t.numel() * t.element_size()


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_available_batch_size(cfg, target_memory_gb: float = 14.0) -> int:
    """
    Estimate the maximum batch size that fits in available GPU memory.

    Args:
        cfg: Model config
        target_memory_gb: Target memory usage in GB (default: 14 for T4 16GB)

    Returns:
        Recommended batch size
    """
    from models.config import estimate_memory

    current_batch = cfg.batch_size

    # Binary search for max batch size
    lo, hi = 1, 64
    best = current_batch

    while lo <= hi:
        mid = (lo + hi) // 2
        test_cfg = cfg
        test_cfg.batch_size = mid
        mem = estimate_memory(test_cfg)
        if mem["total_gb"] < target_memory_gb:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def optimize_memory_settings(cfg) -> Dict:
    """
    Apply recommended memory optimization settings.

    Returns dict of applied optimizations.
    """
    applied = {}

    # Enable expandable segments for PyTorch allocator
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        applied["expandable_segments"] = True

    # Set cuDNN to deterministic mode for reproducibility
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True

    # Enable TF32 on Ampere+ GPUs
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability(0)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied["tf32_enabled"] = True

    # Gradient checkpointing if configured
    if cfg.use_grad_checkpoint:
        applied["gradient_checkpointing"] = True

    return applied


def log_memory_usage(step: int, prefix: str = ""):
    """Log current GPU memory usage to stdout."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{prefix} step {step}: allocated={allocated:.2f}GB  reserved={reserved:.2f}GB")

class KVCacheManager:
    """Custom LRU-like KV cache manager with trimming and eviction."""
    def __init__(self, max_entries: int = 16, max_seq_len: int = 32768):
        self.max_entries = max_entries
        self.max_seq_len = max_seq_len
        self._store = {}   # key -> {cache, last_access}

    def put(self, key: str, cache_obj):
        self._store[key] = {"cache": cache_obj, "last_access": time.time()}
        self._evict_if_needed()

    def get(self, key: str):
        rec = self._store.get(key)
        if not rec:
            return None
        rec["last_access"] = time.time()
        return rec["cache"]

    def trim_kv_cache(self, cache_obj, keep_last_tokens: int = 32768):
        if cache_obj is None or not hasattr(cache_obj, "_cache"):
            return cache_obj
        keep = max(1, min(keep_last_tokens, self.max_seq_len))
        for layer_idx, kv in list(cache_obj._cache.items()):
            k, v = kv
            if k.shape[2] > keep:
                cache_obj._cache[layer_idx] = (k[:, :, -keep:, :], v[:, :, -keep:, :])
        cache_obj.seen_tokens = min(getattr(cache_obj, "seen_tokens", keep), keep)
        return cache_obj

    def sliding_window_context(self, input_ids, window: int = 32768):
        if input_ids is None:
            return input_ids
        win = max(1, min(window, self.max_seq_len))
        if hasattr(input_ids, "shape") and input_ids.shape[-1] > win:
            return input_ids[..., -win:]
        return input_ids

    def memory_pressure_detected(self, threshold: float = 0.9) -> bool:
        if not torch.cuda.is_available():
            return False
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return (reserved / max(total, 1)) >= threshold

    def _evict_if_needed(self):
        while len(self._store) > self.max_entries:
            oldest = min(self._store.items(), key=lambda kv: kv[1]["last_access"])[0]
            del self._store[oldest]
