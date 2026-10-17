"""
Abstract base class for inference engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseEngine(ABC):
    """
    Standard interface for all edge inference runtimes.
    Ensures plug-and-play interchangeability between ONNX Runtime, PyTorch, OpenVINO, etc.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        self.model_path = model_path
        self.backend_name = "base"

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Execute forward pass.

        Args:
            input_tensor: Normalized float32 array with shape [1, 3, H, W].

        Returns:
            Raw model output array, typically [1, 4 + num_classes, num_proposals]
            or [1, num_proposals, 4 + num_classes].
        """
        pass

    @abstractmethod
    def warmup(self, input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640), iterations: int = 5) -> None:
        """Warm up engine caches and JIT compilers."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return engine metadata and execution provider status."""
        return {
            "backend": self.backend_name,
            "model_path": self.model_path,
        }
