"""
Thread-safe metrics and statistical telemetry collector for edge pipelines.
"""

from __future__ import annotations

import threading
from typing import Dict, Any, List, Optional
import numpy as np


class MetricsCollector:
    """
    Collects per-frame telemetry (latency, detection counts, class distribution)
    and computes statistical rollups (p50, p90, p95, p99, defect rate).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._latencies: List[float] = []
            self._class_counts: Dict[str, int] = {}
            self._total_detections: int = 0
            self._defect_frames: int = 0
            self._total_frames: int = 0

    def record(
        self,
        latency_ms: float,
        num_detections: int,
        class_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record telemetry for a processed frame."""
        with self._lock:
            self._total_frames += 1
            self._latencies.append(latency_ms)
            self._total_detections += num_detections

            has_defect = False
            if class_counts:
                for cname, count in class_counts.items():
                    self._class_counts[cname] = self._class_counts.get(cname, 0) + count
                    if cname != "normal_component" and count > 0:
                        has_defect = True

            if has_defect:
                self._defect_frames += 1

    def summary(self) -> Dict[str, Any]:
        """Compute statistical summary of collected metrics."""
        with self._lock:
            if not self._latencies:
                return {
                    "total_frames": 0,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "avg_fps": 0.0,
                    "total_detections": 0,
                    "class_distribution": {},
                    "defect_rate": 0.0,
                }

            arr = np.array(self._latencies, dtype=np.float32)
            avg_lat = float(np.mean(arr))
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            avg_fps = float(1000.0 / max(avg_lat, 1e-4))
            defect_rate = float(self._defect_frames / max(self._total_frames, 1))

            return {
                "total_frames": self._total_frames,
                "avg_latency_ms": round(avg_lat, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "avg_fps": round(avg_fps, 1),
                "total_detections": self._total_detections,
                "class_distribution": dict(self._class_counts),
                "defect_rate": round(defect_rate, 4),
            }
