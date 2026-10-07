"""Runtime resource and inference-performance metrics for the loaded model.

Every source is optional: a missing GPU, `psutil` or NVML turns its fields into `None`
instead of failing the whole report.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

log = logging.getLogger(__name__)

STARTED_AT = time.time()

# Requests per minute is measured over this trailing window.
RPM_WINDOW_S = 60


class LatencyRecorder:
    """Rolling window of the most recent inference latencies."""

    def __init__(self, size: int = 200):
        self._samples: deque[tuple[float, int]] = deque(maxlen=size)
        self._total = 0
        self._lock = threading.Lock()

    def record(self, latency_ms: int) -> None:
        with self._lock:
            self._samples.append((time.time(), latency_ms))
            self._total += 1

    def snapshot(self) -> dict:
        with self._lock:
            samples = list(self._samples)
            total = self._total
        latencies = sorted(ms for _, ms in samples)
        now = time.time()
        return {
            "total": total,
            "last_ms": samples[-1][1] if samples else None,
            "avg_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
            "max_ms": latencies[-1] if latencies else None,
            "per_minute": sum(1 for t, _ in samples if now - t <= RPM_WINDOW_S),
            "recent_ms": [ms for _, ms in samples],
        }


def _percentile(sorted_values: list[int], pct: int) -> int | None:
    if not sorted_values:
        return None
    # Nearest-rank: the smallest value with at least `pct`% of the samples at or below it.
    rank = max(1, -(-len(sorted_values) * pct // 100))
    return sorted_values[rank - 1]


def _models(classifier) -> list[dict]:
    """One entry per checkpoint loaded in the Laya router."""
    agents = getattr(getattr(classifier, "router", None), "_agents", None) or {}
    models = []
    for name, agent in agents.items():
        try:
            params = list(agent.model.parameters())
            models.append(
                {
                    "name": name,
                    "device": str(agent.device),
                    "dtype": str(agent.dtype).removeprefix("torch."),
                    "parameters": sum(p.numel() for p in params),
                    "weights_bytes": sum(p.numel() * p.element_size() for p in params),
                }
            )
        except Exception:
            log.debug("Could not inspect loaded model %r", name, exc_info=True)
    return models


def _cuda_memory() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "name": torch.cuda.get_device_name(),
            "total_bytes": total,
            "free_bytes": free,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    except Exception:
        log.debug("CUDA memory query failed", exc_info=True)
        return None


def _host() -> dict | None:
    try:
        import psutil

        process = psutil.Process()
        memory = psutil.virtual_memory()
        return {
            "process_rss_bytes": process.memory_info().rss,
            "ram_used_bytes": memory.total - memory.available,
            "ram_total_bytes": memory.total,
            # Non-blocking: the first call returns 0.0 and later calls measure since the previous one.
            "process_cpu_percent": process.cpu_percent(interval=None),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_threads": psutil.cpu_count(logical=True),
        }
    except Exception:
        log.debug("Host metrics unavailable", exc_info=True)
        return None


_nvml_lock = threading.Lock()
_nvml_ready: bool | None = None


def _nvml_available() -> bool:
    """Initialises NVML once; a failed init is remembered so it is not retried on every poll."""
    global _nvml_ready
    with _nvml_lock:
        if _nvml_ready is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                _nvml_ready = True
            except Exception:
                log.debug("NVML unavailable", exc_info=True)
                _nvml_ready = False
        return _nvml_ready


def _gpu_utilization() -> dict | None:
    if not _nvml_available():
        return None
    try:
        import pynvml
        import torch

        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device() if torch.cuda.is_available() else 0)
        return {
            "gpu_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            # NVML reports milliwatts.
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1),
            "power_limit_w": round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000),
        }
    except Exception:
        log.debug("GPU utilization query failed", exc_info=True)
        return None


def collect(classifier) -> dict:
    latency = getattr(classifier, "latency", None)
    return {
        "uptime_s": round(time.time() - STARTED_AT),
        "models": _models(classifier),
        "gpu_memory": _cuda_memory(),
        "gpu": _gpu_utilization(),
        "host": _host(),
        "performance": latency.snapshot() if latency else None,
    }
