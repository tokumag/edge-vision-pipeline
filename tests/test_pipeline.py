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
from src.pipeline import DetectionPipeline, PipelineResult
from src.metrics import MetricsCollector


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

    def test_synthetic_stream_deterministic(self):
        stream1 = SyntheticStream(resolution=(640, 640))
        stream2 = SyntheticStream(resolution=(640, 640))
        f1 = stream1.generate_frame()
        f2 = stream2.generate_frame()
        assert np.array_equal(f1, f2)


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

    def test_non_square_high_res_input(self):
        prep = LetterboxPreprocessor(target_size=(640, 640))
        dummy_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        tensor, meta = prep.preprocess(dummy_img)

        assert tensor.shape == (1, 3, 640, 640)
        assert meta["orig_shape"] == (1080, 1920)
        assert "pad" in meta
        assert "scale" in meta

    def test_empty_boxes_scale_back(self):
        prep = LetterboxPreprocessor(target_size=(640, 640))
        meta = {"orig_shape": (480, 640), "scale": 1.0, "pad": (0, 0)}
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        scaled = prep.scale_boxes_back(empty_boxes, meta)
        assert len(scaled) == 0


class TestInferenceEngine:
    def test_mock_engine_infer(self):
        engine = MockEngine(num_classes=6, simulated_latency_ms=0)
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        out = engine.infer(tensor)

        assert out.shape == (1, 10, 8400)
        assert engine.get_metadata()["backend"] == "mock"

    def test_mock_engine_warmup_no_error(self):
        engine = MockEngine(num_classes=6, simulated_latency_ms=0)
        engine.warmup(iterations=3)

    def test_engine_factory_mock(self):
        cfg = {"backend": "mock", "classes": [0, 1, 2]}
        engine = create_engine(cfg)
        assert engine.backend_name == "mock"


class TestPostprocess:
    def test_nms_suppression(self):
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205],
            [400, 400, 500, 500],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
        class_ids = np.array([0, 0, 0], dtype=np.int32)

        keep = YOLOOutputDecoder.non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_nms_different_classes_retained(self):
        # Two identical boxes but belonging to DIFFERENT classes must NOT be suppressed
        boxes = np.array([
            [100, 100, 200, 200],
            [100, 100, 200, 200],
        ], dtype=np.float32)
        scores = np.array([0.95, 0.90], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)

        keep = YOLOOutputDecoder.non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(keep) == 2

    def test_decoder_high_threshold_empty(self):
        decoder = YOLOOutputDecoder(conf_threshold=0.999)
        engine = MockEngine(num_classes=2, simulated_latency_ms=0)
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        raw_output = engine.infer(tensor)
        meta = {"orig_shape": (640, 640), "scale": 1.0, "pad": (0, 0)}

        detections = decoder.decode(raw_output, meta)
        assert len(detections) == 0


class TestTracker:
    def test_tracker_assignment(self):
        tracker = EdgeTracker(max_lost_frames=5)
        dets_frame1 = [{"bbox": [100, 100, 200, 200], "class_id": 0, "confidence": 0.9}]
        tracked1 = tracker.update(dets_frame1)
        assert tracked1[0]["track_id"] == 1

        dets_frame2 = [{"bbox": [104, 102, 204, 202], "class_id": 0, "confidence": 0.92}]
        tracked2 = tracker.update(dets_frame2)
        assert tracked2[0]["track_id"] == 1
        assert tracked2[0]["track_age"] == 2

    def test_tracker_distant_detection_new_id(self):
        tracker = EdgeTracker(max_lost_frames=5, iou_distance_thresh=0.3)
        dets_frame1 = [{"bbox": [100, 100, 200, 200], "class_id": 0, "confidence": 0.9}]
        tracked1 = tracker.update(dets_frame1)
        assert tracked1[0]["track_id"] == 1

        # Far away detection should spawn track_id 2
        dets_frame2 = [{"bbox": [500, 500, 600, 600], "class_id": 0, "confidence": 0.9}]
        tracked2 = tracker.update(dets_frame2)
        assert tracked2[0]["track_id"] == 2


class TestProfiler:
    def test_benchmark_run(self):
        engine = MockEngine(num_classes=6, simulated_latency_ms=1.0)
        profiler = EdgeProfiler(engine=engine, warmup_runs=2, test_runs=10)
        results = profiler.run_benchmark()

        assert "throughput_fps" in results
        assert "latency" in results
        assert results["latency"]["p50_ms"] > 0
        assert results["latency"]["p95_ms"] >= results["latency"]["p50_ms"]


class TestPipelineAndMetrics:
    def test_detection_pipeline_end_to_end(self):
        cfg = {
            "model": {
                "backend": "mock",
                "input_size": [640, 640],
                "confidence_threshold": 0.45,
                "iou_threshold": 0.50,
            },
            "classes": {0: "normal_component", 1: "surface_scratch", 2: "structural_crack"},
        }
        pipeline = DetectionPipeline(config=cfg, force_mock=True)
        stream = SyntheticStream(resolution=(640, 640))
        frame = stream.generate_frame()

        result = pipeline.process_frame(frame)
        assert isinstance(result, PipelineResult)
        assert result.annotated_frame.shape == (640, 640, 3)
        assert result.latency_ms > 0
        assert result.fps > 0
        assert len(result.detections) > 0

    def test_metrics_collector(self):
        collector = MetricsCollector()
        for i in range(10):
            collector.record(
                latency_ms=15.0 + i,
                num_detections=2,
                class_counts={"normal_component": 1, "structural_crack": 1 if i % 2 == 0 else 0},
            )

        summary = collector.summary()
        assert summary["total_frames"] == 10
        assert summary["total_detections"] == 20
        assert summary["avg_latency_ms"] > 15.0
        assert summary["p50_latency_ms"] >= 15.0
        assert summary["defect_rate"] == 0.5
