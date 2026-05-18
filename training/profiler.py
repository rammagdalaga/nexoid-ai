"""
APEXAI — Training Profiler
Performance monitoring, GPU utilization tracking, and profiling utilities.
"""

import os
import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ProfileEvent:
    """A single profiling event."""
    name: str
    start_time: float
    end_time: float
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0


@dataclass
class TrainingProfile:
    """Aggregated training profile data."""
    total_steps: int = 0
    avg_step_time_ms: float = 0.0
    avg_loss: float = 0.0
    gpu_utilization_pct: float = 0.0
    memory_peak_gb: float = 0.0
    events: List[ProfileEvent] = field(default_factory=list)


class Profiler:
    """
    Lightweight training profiler.
    Tracks step times, memory usage, and GPU utilization.
    """

    def __init__(self, enabled: bool = True, log_interval: int = 10):
        self.enabled = enabled
        self.log_interval = log_interval
        self.events: List[ProfileEvent] = []
        self._current_event: Optional[ProfileEvent] = None
        self._step_times: List[float] = []
        self._peak_memory = 0.0

    def start_event(self, name: str):
        """Start a profiling event."""
        if not self.enabled:
            return
        mem_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        mem_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        self._current_event = ProfileEvent(
            name=name,
            start_time=time.time(),
            end_time=0.0,
            gpu_memory_allocated=mem_allocated,
            gpu_memory_reserved=mem_reserved,
        )

    def end_event(self):
        """End the current profiling event."""
        if not self.enabled or self._current_event is None:
            return
        self._current_event.end_time = time.time()
        self.events.append(self._current_event)
        self._current_event = None

    def record_step(self, loss: float):
        """Record a training step."""
        if not self.enabled:
            return

        self._step_times.append(time.time())
        if len(self._step_times) > 1:
            # Keep last N step times for moving average
            if len(self._step_times) > 100:
                self._step_times = self._step_times[-100:]

        # Track peak memory
        if torch.cuda.is_available():
            current_mem = torch.cuda.max_memory_allocated() / 1e9
            self._peak_memory = max(self._peak_memory, current_mem)

    def get_step_time_ms(self) -> float:
        """Get average step time in milliseconds."""
        if len(self._step_times) < 2:
            return 0.0
        times = [
            self._step_times[i + 1] - self._step_times[i]
            for i in range(len(self._step_times) - 1)
        ]
        return (sum(times) / len(times)) * 1000

    def get_gpu_utilization(self) -> float:
        """Get approximate GPU utilization as percentage."""
        if not torch.cuda.is_available():
            return 0.0

        try:
            # Use nvidia-ml if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except (ImportError, Exception):
            pass

        # Fallback: estimate from memory usage
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return allocated / max(total, 1)

    def get_profile(self) -> TrainingProfile:
        """Get the aggregated training profile."""
        n_steps = len(self._step_times)
        if n_steps == 0:
            return TrainingProfile()

        return TrainingProfile(
            total_steps=n_steps,
            avg_step_time_ms=self.get_step_time_ms(),
            gpu_utilization_pct=self.get_gpu_utilization() * 100,
            memory_peak_gb=self._peak_memory,
            events=self.events[-50:],  # Keep last 50 events
        )

    def print_summary(self):
        """Print a formatted profiling summary."""
        profile = self.get_profile()

        print("\n" + "=" * 50)
        print("  APEXAI — Training Profile")
        print("=" * 50)
        print(f"  Steps Tracked:    {profile.total_steps}")
        print(f"  Avg Step Time:    {profile.avg_step_time_ms:.1f} ms")
        print(f"  Steps/sec:        {1000 / max(profile.avg_step_time_ms, 0.01):.1f}")
        print(f"  GPU Utilization:  {profile.gpu_utilization_pct:.1f}%")
        print(f"  Peak GPU Memory:  {profile.memory_peak_gb:.2f} GB")
        print(f"  Events Logged:    {len(profile.events)}")

        if profile.events:
            print("\n  Recent Events:")
            for event in profile.events[-5:]:
                duration = (event.end_time - event.start_time) * 1000
                print(f"    {event.name}: {duration:.1f} ms")

        print("=" * 50)


class GPUMonitor:
    """
    Real-time GPU monitoring.
    Logs GPU temperature, power, and memory usage.
    """

    def __init__(self, interval_seconds: int = 10):
        self.interval = interval_seconds
        self._last_check = 0.0
        self._metrics: List[Dict] = []

    def check(self) -> Optional[Dict]:
        """
        Check GPU stats. Returns dict if interval has elapsed.
        """
        now = time.time()
        if now - self._last_check < self.interval:
            return None

        self._last_check = now
        metrics = {"timestamp": now, "memory_gb": 0.0, "temperature_c": 0.0, "power_w": 0.0}

        if not torch.cuda.is_available():
            return metrics

        metrics["memory_gb"] = torch.cuda.memory_allocated() / 1e9

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            metrics["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            metrics["power_w"] = power / 1000.0
        except ImportError:
            pass
        except Exception:
            pass

        self._metrics.append(metrics)
        return metrics

    def get_average_power(self) -> float:
        """Get average power consumption in watts."""
        if not self._metrics:
            return 0.0
        powers = [m.get("power_w", 0) for m in self._metrics if m.get("power_w")]
        return sum(powers) / max(len(powers), 1)


def estimate_flops(cfg, seq_len: int = None) -> Dict:
    """
    Estimate FLOPs per training step.

    Returns dict with forward, backward, and total FLOPs.
    """
    if seq_len is None:
        seq_len = cfg.block_size

    n_layer = cfg.n_layer
    n_embd = cfg.n_embd
    n_head = cfg.n_head
    head_dim = n_embd // n_head
    ffn_hidden = cfg.ffn_mult * n_embd
    batch = cfg.batch_size

    # Attention FLOPs per layer
    # QKV projection: 3 * seq_len * n_embd * n_embd
    attn_qkv = 3 * seq_len * n_embd * n_embd

    # Attention scores: seq_len * seq_len * n_head * head_dim * 2
    attn_scores = 2 * seq_len * seq_len * n_head * head_dim

    # Attention output: attn_scores + projection
    attn_output = attn_scores + seq_len * n_embd * n_embd

    attn_total = attn_qkv + attn_output

    # FFN FLOPs per layer (SwiGLU: gate + up + down)
    ffn_total = seq_len * (2 * n_embd * ffn_hidden + ffn_hidden * n_embd)

    # Per layer total
    layer_flops = attn_total + ffn_total

    # Total forward FLOPs
    forward_flops = batch * n_layer * layer_flops

    # Backward is ~2x forward
    total_flops = forward_flops * 3  # forward + backward

    return {
        "forward_tflop": forward_flops / 1e12,
        "backward_tflop": (forward_flops * 2) / 1e12,
        "total_tflop": total_flops / 1e12,
    }