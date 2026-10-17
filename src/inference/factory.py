"""
Factory for creating inference engine backends with graceful fallback.
"""

import os
from typing import Dict, Any, Optional
from src.inference.base import BaseEngine
from src.inference.mock_engine import MockEngine


def create_engine(config: Dict[str, Any], force_mock: bool = False) -> BaseEngine:
    """
    Construct an inference engine according to configuration.
    Falls back gracefully to MockEngine if model files or runtime libraries are absent.
    """
    backend = config.get("backend", "onnx").lower()
    model_path = config.get("model_path", "weights/yolov8n_defect.onnx")

    if force_mock or backend == "mock" or not os.path.exists(model_path):
        return MockEngine(
            model_path=model_path,
            num_classes=len(config.get("classes", [0, 1, 2, 3, 4, 5])),
            simulated_latency_ms=14.0,
        )

    if backend == "onnx":
        try:
            from src.inference.onnx_engine import ONNXEngine
            return ONNXEngine(
                model_path=model_path,
                providers=config.get("providers", ["CPUExecutionProvider"]),
                num_threads=config.get("num_threads", 4),
            )
        except Exception:
            # Fallback to MockEngine if onnxruntime fails to load checkpoint
            return MockEngine(model_path=model_path)

    raise ValueError(f"Unsupported inference backend: {backend}")
