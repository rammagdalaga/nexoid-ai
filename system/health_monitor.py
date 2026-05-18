import os
import time
from collections import deque
from typing import Deque, Dict, Optional

import torch


class HealthMonitor:
    def __init__(self, window: int = 100):
        self.window = max(10, window)
        self.inference_latencies: Deque[float] = deque(maxlen=self.window)
        self.training_tokens_per_sec: Deque[float] = deque(maxlen=self.window)
        self.error_flags: Deque[int] = deque(maxlen=self.window)

    def record_inference_latency(self, seconds: float):
        self.inference_latencies.append(max(0.0, seconds))

    def record_training_throughput(self, tokens: int, elapsed_seconds: float):
        if elapsed_seconds > 0:
            self.training_tokens_per_sec.append(tokens / elapsed_seconds)

    def record_error(self, has_error: bool):
        self.error_flags.append(1 if has_error else 0)

    def _gpu_mem(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"gpu_total_gb": 0.0, "gpu_allocated_gb": 0.0, "gpu_reserved_gb": 0.0}
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc = torch.cuda.memory_allocated(0) / 1e9
        res = torch.cuda.memory_reserved(0) / 1e9
        return {"gpu_total_gb": total, "gpu_allocated_gb": alloc, "gpu_reserved_gb": res}

    def snapshot(self) -> Dict:
        avg_latency = sum(self.inference_latencies) / len(self.inference_latencies) if self.inference_latencies else 0.0
        avg_tps = sum(self.training_tokens_per_sec) / len(self.training_tokens_per_sec) if self.training_tokens_per_sec else 0.0
        err_rate = (sum(self.error_flags) / len(self.error_flags)) if self.error_flags else 0.0
        return {
            "ts": int(time.time()),
            "pid": os.getpid(),
            "avg_inference_latency_s": avg_latency,
            "avg_training_tokens_per_sec": avg_tps,
            "error_rate": err_rate,
            **self._gpu_mem(),
            "healthy": err_rate < 0.2,
        }
