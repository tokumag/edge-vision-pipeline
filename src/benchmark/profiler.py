"""
Edge inference benchmarking suite for latency, FPS throughput, and memory profiling.
"""

from __future__ import annotations

import time
import os
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from src.inference.base import BaseEngine


class EdgeProfiler:
    """
    Precision hardware profiler for edge inference runtimes.
    Calculates tail latencies (p50, p95, p99), throughput (FPS), and peak memory usage.
    """

    def __init__(
        self,
        engine: BaseEngine,
        warmup_runs: int = 15,
        test_runs: int = 100,
        input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    ) -> None:
        self.engine = engine
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.input_shape = input_shape

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Execute profiling runs and return latency percentiles, throughput, and memory metrics.
        """
        dummy_tensor = np.random.uniform(0.0, 1.0, self.input_shape).astype(np.float32)

        # 1. Warmup passes
        for _ in range(self.warmup_runs):
            self.engine.infer(dummy_tensor)

        # Memory before
        process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        mem_before_mb = (process.memory_info().rss / (1024 * 1024)) if process else 0.0

        # 2. Timed benchmark passes
        latencies_ms: List[float] = []
        start_overall = time.perf_counter()

        for _ in range(self.test_runs):
            t0 = time.perf_counter()
            self.engine.infer(dummy_tensor)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        total_duration = time.perf_counter() - start_overall
        mem_after_mb = (process.memory_info().rss / (1024 * 1024)) if process else 0.0

        arr = np.array(latencies_ms)
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean_lat = float(np.mean(arr))
        std_lat = float(np.std(arr))
        fps = float(self.test_runs / total_duration)

        return {
            "backend": self.engine.backend_name,
            "test_runs": self.test_runs,
            "input_shape": list(self.input_shape),
            "latency": {
                "mean_ms": round(mean_lat, 2),
                "std_ms": round(std_lat, 2),
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "min_ms": round(float(np.min(arr)), 2),
                "max_ms": round(float(np.max(arr)), 2),
            },
            "throughput_fps": round(fps, 2),
            "memory": {
                "rss_mb": round(mem_after_mb, 2),
                "delta_mb": round(max(0.0, mem_after_mb - mem_before_mb), 2),
            },
        }

    @staticmethod
    def compare_runtimes(results: List[Dict[str, Any]]) -> str:
        """Format multiple runtime benchmark runs into a clean markdown comparison table."""
        header = "| Backend | Mean Latency | p95 Latency | Throughput (FPS) | Memory RSS |\n|---|---|---|---|---|\n"
        rows = []
        for r in results:
            b = r["backend"].upper()
            mean_l = f"{r['latency']['mean_ms']} ms"
            p95_l = f"{r['latency']['p95_ms']} ms"
            fps = f"{r['throughput_fps']} FPS"
            mem = f"{r['memory']['rss_mb']} MB"
            rows.append(f"| **{b}** | {mean_l} | {p95_l} | {fps} | {mem} |")
        return header + "\n".join(rows)
