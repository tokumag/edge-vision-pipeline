"""
End-to-end integration and unit tests for Edge Vision Pipeline.
Run via: pytest tests/ -v
"""

import numpy as np
import pytest

from src.core.stream import SyntheticStream, VideoStream
from src.core.preprocessor import LetterboxPreprocessor
from src.inference.mock_engine import MockEngine
from src.inference.factory import create_engine
from src.postprocess.nms import YOLOOutputDecoder
from src.postprocess.tracker import EdgeTracker
from src.postprocess.visualizer import FrameVisualizer
from src.benchmark.profiler import EdgeProfiler


class TestStream:
    def test_synthetic_stream_shape(self):
        stream = SyntheticStream(resolution=(640, 480))
        frame = stream.generate_frame()
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_stream_generator_count(self):
        stream = SyntheticStream(resolution=(320, 320))
        frames = list(stream.frames(count=5))
        assert len(frames) == 5


class TestPreprocessor:
    def test_letterbox_output_shape(self):
        prep = LetterboxPreprocessor(target_size=(640, 640))
        dummy_img = np.zeros((480, 800, 3), dtype=np.uint8)
        tensor, meta = prep.preprocess(dummy_img)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min() <= tensor.max() <= 1.0
        assert meta["orig_shape"] == (480, 800)

    def test_scale_boxes_back_identity(self):
        prep = LetterboxPreprocessor(target_size=(640, 640))
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, meta = prep.preprocess(dummy_img)

        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scaled = prep.scale_boxes_back(boxes, meta)
        assert np.allclose(boxes, scaled, atol=1.0)


class TestInferenceEngine:
    def test_mock_engine_infer(self):
        engine = MockEngine(num_classes=6, simulated_latency_ms=0)
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        out = engine.infer(tensor)

        assert out.shape == (1, 10, 8400)
        assert engine.get_metadata()["backend"] == "mock"

    def test_engine_factory_mock(self):
        cfg = {"backend": "mock", "classes": [0, 1, 2]}
        engine = create_engine(cfg)
        assert engine.backend_name == "mock"


class TestPostprocess:
    def test_nms_suppression(self):
        # Two overlapping boxes of same class with high IoU
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205], # Near identical
            [400, 400, 500, 500], # Distinct
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
        class_ids = np.array([0, 0, 0], dtype=np.int32)

        keep = YOLOOutputDecoder.non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)
        # Should keep box 0 and box 2, suppressing box 1
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_decoder_synthetic(self):
        decoder = YOLOOutputDecoder(
            conf_threshold=0.5,
            iou_threshold=0.5,
            class_names={0: "normal", 1: "defect"},
        )
        engine = MockEngine(num_classes=2, simulated_latency_ms=0)
        # High red channel to trigger defect detection
        tensor = np.full((1, 3, 640, 640), 0.8, dtype=np.float32)
        raw_output = engine.infer(tensor)

        meta = {
            "orig_shape": (640, 640),
            "scale": 1.0,
            "pad": (0, 0),
            "target_shape": (640, 640),
        }
        detections = decoder.decode(raw_output, meta)
        assert len(detections) > 0
        assert "bbox" in detections[0]
        assert "confidence" in detections[0]


class TestTracker:
    def test_tracker_assignment(self):
        tracker = EdgeTracker(max_lost_frames=5)
        dets_frame1 = [{"bbox": [100, 100, 200, 200], "class_id": 0, "confidence": 0.9}]
        tracked1 = tracker.update(dets_frame1)
        assert tracked1[0]["track_id"] == 1

        # Frame 2 with slight movement
        dets_frame2 = [{"bbox": [104, 102, 204, 202], "class_id": 0, "confidence": 0.92}]
        tracked2 = tracker.update(dets_frame2)
        # Should retain same track ID
        assert tracked2[0]["track_id"] == 1
        assert tracked2[0]["track_age"] == 2


class TestProfiler:
    def test_benchmark_run(self):
        engine = MockEngine(num_classes=6, simulated_latency_ms=2.0)
        profiler = EdgeProfiler(engine=engine, warmup_runs=2, test_runs=10)
        results = profiler.run_benchmark()

        assert "throughput_fps" in results
        assert "latency" in results
        assert results["latency"]["p50_ms"] > 0
        assert results["latency"]["p95_ms"] >= results["latency"]["p50_ms"]
