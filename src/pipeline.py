"""
Unified edge detection pipeline orchestrating preprocessing, inference, tracking, visualization, and defect heatmaps.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.core.preprocessor import LetterboxPreprocessor
from src.inference.factory import create_engine
from src.inference.base import BaseEngine
from src.postprocess.nms import YOLOOutputDecoder
from src.postprocess.tracker import EdgeTracker
from src.postprocess.visualizer import FrameVisualizer
from src.postprocess.heatmap import DefectHeatmap
from src.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured result returned for each processed video frame."""
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Dict[str, Any]]
    latency_ms: float
    fps: float
    metadata: Dict[str, Any]
    hotspots: List[Dict[str, Any]] = field(default_factory=list)


class DetectionPipeline:
    """
    Production-grade end-to-end edge computer vision pipeline.
    Encapsulates preprocessing, deep learning inference, NMS suppression,
    persistent tracking, spatial defect heatmaps, and visual annotation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        force_mock: bool = False,
    ) -> None:
        self.config = config
        model_cfg = config.get("model", {})
        self.target_size = tuple(model_cfg.get("input_size", [640, 640]))

        self.preprocessor = LetterboxPreprocessor(target_size=self.target_size)
        self.engine: BaseEngine = create_engine(model_cfg, force_mock=force_mock)
        self.decoder = YOLOOutputDecoder(
            conf_threshold=model_cfg.get("confidence_threshold", 0.45),
            iou_threshold=model_cfg.get("iou_threshold", 0.50),
            class_names=config.get("classes", {}),
        )
        self.tracker = EdgeTracker()
        self.visualizer = FrameVisualizer()
        self.heatmap = DefectHeatmap(resolution=self.target_size)
        self.metrics = MetricsCollector()

        logger.info(
            "Initialized DetectionPipeline with backend=%s, input_size=%s",
            self.engine.backend_name,
            self.target_size,
        )

    def process_frame(
        self, frame: np.ndarray, apply_heatmap: bool = False
    ) -> PipelineResult:
        """
        Execute full inference pipeline on a single frame.

        Args:
            frame: Input image array [H, W, 3] in BGR format.
            apply_heatmap: Whether to blend spatial defect density onto output frame.

        Returns:
            PipelineResult containing detections, annotated frame, and timing diagnostics.
        """
        t0 = time.perf_counter()

        # 1. Preprocess (Letterbox & Normalization)
        tensor, meta = self.preprocessor.preprocess(frame)

        # 2. Forward Inference
        raw_output = self.engine.infer(tensor)

        # 3. Postprocess (NMS & Coordinate Scaling)
        detections = self.decoder.decode(raw_output, meta)

        # 4. Association & Persistent Tracking
        detections = self.tracker.update(detections)

        # 5. Spatial Defect Density Accumulation
        self.heatmap.update(detections)
        hotspots = self.heatmap.get_top_hotspots(k=3)

        # Latency calculation
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-4)

        # 6. Visual Rendering
        annotated = self.visualizer.render(
            frame=frame,
            detections=detections,
            fps=fps,
            latency_ms=latency_ms,
            backend=self.engine.backend_name,
        )

        if apply_heatmap:
            annotated = self.heatmap.render_overlay(annotated, alpha=0.40)

        # 7. Record Telemetry
        class_counts: Dict[str, int] = {}
        for d in detections:
            cname = d.get("class_name", "unknown")
            class_counts[cname] = class_counts.get(cname, 0) + 1

        self.metrics.record(
            latency_ms=latency_ms,
            num_detections=len(detections),
            class_counts=class_counts,
        )

        return PipelineResult(
            raw_frame=frame,
            annotated_frame=annotated,
            detections=detections,
            latency_ms=round(latency_ms, 2),
            fps=round(fps, 1),
            metadata=meta,
            hotspots=hotspots,
        )

    def get_heatmap_frame(self, base_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Render standalone thermal heatmap overlay."""
        if base_frame is None:
            base_frame = np.full((self.target_size[1], self.target_size[0], 3), 40, dtype=np.uint8)
        return self.heatmap.render_overlay(base_frame, alpha=0.60)

    def reset_state(self) -> None:
        """Reset tracker, heatmap, and telemetry metrics."""
        self.tracker = EdgeTracker()
        self.heatmap.reset()
        self.metrics.reset()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Fetch statistical rollup from metrics collector."""
        return self.metrics.summary()
