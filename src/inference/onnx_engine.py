"""
ONNX Runtime inference backend optimized for edge CPU and accelerator deployment.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.inference.base import BaseEngine

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


class ONNXEngine(BaseEngine):
    """
    ONNX Runtime engine with graph optimizations and dynamic thread allocation.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        num_threads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(model_path)
        self.backend_name = "onnx"

        if not _HAS_ONNX:
            raise ImportError("onnxruntime is not installed. Run `pip install onnxruntime`.")

        # Setup session options
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = num_threads
        session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        requested_providers = providers or ["CPUExecutionProvider"]
        active_providers = [p for p in requested_providers if p in available_providers]
        if not active_providers:
            active_providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_opts,
            providers=active_providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.active_providers = self.session.get_providers()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Execute forward inference through ONNX Runtime."""
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor},
        )
        return outputs[0]

    def warmup(self, input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640), iterations: int = 5) -> None:
        """Run warmup passes to populate memory pools and execution caches."""
        dummy_tensor = np.zeros(input_shape, dtype=np.float32)
        for _ in range(iterations):
            self.infer(dummy_tensor)

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "input_name": self.input_name,
            "output_names": self.output_names,
            "providers": self.active_providers,
        })
        return meta
