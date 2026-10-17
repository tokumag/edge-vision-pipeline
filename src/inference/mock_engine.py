"""
Mock / Synthetic inference engine for local testing, CI/CD, and offline verification.
"""

import time
from typing import Tuple, Dict, Any
import numpy as np
from src.inference.base import BaseEngine


class MockEngine(BaseEngine):
    """
    Simulates a YOLOv8-like detector producing standard raw tensors of shape:
    [1, 4 + num_classes, num_anchors] where num_anchors = 8400.
    Simulates edge inference latency (configurable, e.g. 12-18ms).
    """

    def __init__(
        self,
        model_path: str = "mock_model.onnx",
        num_classes: int = 6,
        simulated_latency_ms: float = 12.0,
        **kwargs,
    ) -> None:
        super().__init__(model_path)
        self.backend_name = "mock"
        self.num_classes = num_classes
        self.simulated_latency_ms = simulated_latency_ms
        self.num_anchors = 8400

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Produce a simulated YOLO detection tensor:
        Format: [batch, 4 + num_classes, num_anchors]
        Channels 0..3: cx, cy, w, h
        Channels 4..4+C: class probabilities
        """
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        batch_size = input_tensor.shape[0]
        output = np.zeros((batch_size, 4 + self.num_classes, self.num_anchors), dtype=np.float32)

        # Create 2-4 synthetic detections per batch
        # 1. Main component detection (class 0: normal_component)
        output[:, 0, 100] = 320.0 # cx
        output[:, 1, 100] = 320.0 # cy
        output[:, 2, 100] = 220.0 # w
        output[:, 3, 100] = 160.0 # h
        output[:, 4, 100] = 0.92  # class 0 confidence

        # 2. Defect detection
        mean_r = float(np.mean(input_tensor[:, 0, :, :]))
        if mean_r > 0.05 and self.num_classes > 1:
            output[:, 0, 250] = 340.0
            output[:, 1, 250] = 330.0
            output[:, 2, 250] = 70.0
            output[:, 3, 250] = 35.0
            defect_cls_1 = 4 + min(2, self.num_classes - 1)
            output[:, defect_cls_1, 250] = 0.86

            # Secondary minor scratch if classes allow
            if self.num_classes > 2:
                output[:, 0, 400] = 260.0
                output[:, 1, 400] = 290.0
                output[:, 2, 400] = 40.0
                output[:, 3, 400] = 20.0
                defect_cls_2 = 4 + min(1, self.num_classes - 1)
                output[:, defect_cls_2, 400] = 0.74

        return output

    def warmup(self, input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640), iterations: int = 5) -> None:
        """No-op warmup for mock engine."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "num_classes": self.num_classes,
            "simulated_latency_ms": self.simulated_latency_ms,
        })
        return meta
