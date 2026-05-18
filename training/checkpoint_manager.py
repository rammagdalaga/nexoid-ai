"""
APEXAI MODULE STATUS
Phase: 2 HARDENING COMPLETE
Role: Atomic checkpoint persistence and recovery manager
Dependencies: training loop + SystemManager recovery logic
System Integration: ACTIVE
Thread Safety: ENFORCED

Responsibilities:
- Save/load versioned checkpoints with integrity validation.
- Rotate older artifacts while preserving latest reliable recovery points.
"""

import hashlib
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = max(1, keep_last_n)
        self._lock = threading.RLock()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _ckpt_path(self, step: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{step:08d}.pt")

    def _tmp_ckpt_path(self, step: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{step:08d}.tmp")

    def _hash_path(self, step: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{step:08d}.sha256")

    def save(self, step: int, model_state: Dict[str, Any], optimizer_state: Optional[Dict[str, Any]] = None,
             meta: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            path = self._ckpt_path(step)
            tmp_path = self._tmp_ckpt_path(step)
            payload = {
                "step": step,
                "saved_at": int(time.time()),
                "model": model_state,
                "optimizer": optimizer_state,
                "meta": meta or {},
            }
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            digest = self._sha256_file(path)
            with open(self._hash_path(step), "w", encoding="utf-8") as f:
                f.write(digest)
            self._rotate()
            return path

    def load(self, step: int, verify_integrity: bool = True) -> Dict[str, Any]:
        with self._lock:
            path = self._ckpt_path(step)
            if verify_integrity and not self.verify(step):
                raise ValueError(f"Checkpoint integrity failed for step={step}")
            return torch.load(path, map_location="cpu", weights_only=False)

    def latest_step(self) -> Optional[int]:
        with self._lock:
            steps = []
            for f in os.listdir(self.checkpoint_dir):
                if f.startswith("ckpt_") and f.endswith(".pt"):
                    try:
                        steps.append(int(f[len("ckpt_"):-3]))
                    except ValueError:
                        pass
            return max(steps) if steps else None

    def resume_latest(self, verify_integrity: bool = True) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        step = self.latest_step()
        if step is None:
            return None, None
        return step, self.load(step, verify_integrity=verify_integrity)

    def verify(self, step: int) -> bool:
        with self._lock:
            path = self._ckpt_path(step)
            hash_path = self._hash_path(step)
            if not (os.path.exists(path) and os.path.exists(hash_path)):
                return False
            actual = self._sha256_file(path)
            with open(hash_path, encoding="utf-8") as f:
                expected = f.read().strip()
            return actual == expected

    def _rotate(self):
        entries = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("ckpt_") and f.endswith(".pt"):
                try:
                    entries.append(int(f[len("ckpt_"):-3]))
                except ValueError:
                    pass
        entries.sort(reverse=True)
        for step in entries[self.keep_last_n:]:
            pt = self._ckpt_path(step)
            hs = self._hash_path(step)
            if os.path.exists(pt):
                os.remove(pt)
            if os.path.exists(hs):
                os.remove(hs)

    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
