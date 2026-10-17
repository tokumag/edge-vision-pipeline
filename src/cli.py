"""
Command-Line Interface for Edge Vision Pipeline.
Provides detection streaming, hardware benchmarking, and model export commands.
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import time
import yaml
from pathlib import Path
import numpy as np

from src.core.stream import VideoStream, SyntheticStream
from src.core.preprocessor import LetterboxPreprocessor
from src.inference.factory import create_engine
from src.postprocess.nms import YOLOOutputDecoder
from src.postprocess.tracker import EdgeTracker
from src.postprocess.visualizer import FrameVisualizer
from src.benchmark.profiler import EdgeProfiler


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(path: str = "configs/default.yaml") -> dict:
    """Safely load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_detect(args: argparse.Namespace) -> None:
    """Run real-time edge defect detection on camera, file, or synthetic stream."""
    config = load_config(args.config)
    print("\n" + "=" * 60)
    print("  [*] EDGE VISION PIPELINE - REAL-TIME INFERENCE")
    print("=" * 60)

    # Initialize components
    target_size = tuple(config["model"]["input_size"])
    preprocessor = LetterboxPreprocessor(target_size=target_size)
    engine = create_engine(config["model"], force_mock=args.mock)
    decoder = YOLOOutputDecoder(
        conf_threshold=args.conf or config["model"]["confidence_threshold"],
        iou_threshold=config["model"]["iou_threshold"],
        class_names=config.get("classes", {}),
    )
    tracker = EdgeTracker()
    visualizer = FrameVisualizer()

    print(f"Backend:         {engine.backend_name.upper()}")
    print(f"Input size:      {target_size}")
    print(f"Conf threshold:  {decoder.conf_threshold}")
    print(f"Frames to run:   {args.frames}")
    print("-" * 60)

    stream = SyntheticStream(resolution=(640, 640))
    total_detections = 0
    start_time = time.perf_counter()

    for idx in range(args.frames):
        frame = stream.generate_frame()
        t0 = time.perf_counter()

        # 1. Preprocess
        tensor, meta = preprocessor.preprocess(frame)

        # 2. Infer
        raw_output = engine.infer(tensor)

        # 3. Postprocess & Track
        detections = decoder.decode(raw_output, meta)
        detections = tracker.update(detections)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-4)

        # 4. Visualize
        annotated = visualizer.render(
            frame, detections, fps=fps, latency_ms=latency_ms, backend=engine.backend_name
        )
        total_detections += len(detections)

        if (idx + 1) % 10 == 0 or idx == args.frames - 1:
            print(f"  Frame [{idx + 1:3d}/{args.frames}] | Latency: {latency_ms:5.1f}ms | FPS: {fps:4.1f} | Detections: {len(detections)}")

    elapsed = time.perf_counter() - start_time
    avg_fps = args.frames / max(elapsed, 1e-4)
    print("-" * 60)
    print(f"[OK] Pipeline run complete in {elapsed:.2f}s (Avg {avg_fps:.1f} FPS)")
    print(f"[OK] Total defect/component instances tracked: {tracker.next_track_id - 1}")
    print("=" * 60 + "\n")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Execute high-precision hardware inference profiler."""
    config = load_config(args.config)
    print("\n" + "=" * 60)
    print("  [*] EDGE HARDWARE BENCHMARK & LATENCY PROFILER")
    print("=" * 60)

    engine = create_engine(config["model"], force_mock=args.mock)
    target_size = tuple(config["model"]["input_size"])

    profiler = EdgeProfiler(
        engine=engine,
        warmup_runs=args.warmup,
        test_runs=args.runs,
        input_shape=(1, 3, target_size[0], target_size[1]),
    )

    print(f"Backend:     {engine.backend_name.upper()}")
    print(f"Resolution:  {target_size}")
    print(f"Iterations:  {args.runs} (warmup: {args.warmup})")
    print("\nExecuting profiling passes...")

    results = profiler.run_benchmark()

    print("\n" + "-" * 40)
    print("BENCHMARK RESULTS")
    print("-" * 40)
    print(f"  Throughput:     {results['throughput_fps']} FPS")
    print(f"  Mean Latency:   {results['latency']['mean_ms']} ms (+/-{results['latency']['std_ms']} ms)")
    print(f"  Median (p50):   {results['latency']['p50_ms']} ms")
    print(f"  95th % (p95):   {results['latency']['p95_ms']} ms")
    print(f"  99th % (p99):   {results['latency']['p99_ms']} ms")
    print(f"  Memory (RSS):   {results['memory']['rss_mb']} MB")
    print("-" * 40)

    if args.json:
        print("\nJSON Output:")
        print(json.dumps(results, indent=2))
    print("=" * 60 + "\n")


def cmd_export(args: argparse.Namespace) -> None:
    """Generate model export commands and deployment configuration."""
    print("\n" + "=" * 60)
    print("  [*] EDGE MODEL DEPLOYMENT & EXPORT GENERATOR")
    print("=" * 60)
    print(f"Target Format: {args.format.upper()}")
    print(f"Model Path:    {args.model}")

    if args.format == "onnx":
        print("\nExporting PyTorch YOLO to ONNX:")
        print(f"  yolo export model={args.model} format=onnx imgsz=640 simplify=True dynamic=False")
        print("\nOptimizing ONNX for CPU with onnxruntime:")
        print("  python -m onnxruntime.tools.convert_onnx_models_to_ort weights/model.onnx")
    elif args.format == "openvino":
        print("\nConverting ONNX to OpenVINO IR (FP16):")
        print(f"  mo --input_model {args.model} --compress_to_fp16 --output_dir weights/openvino_fp16/")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge Computer Vision Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: detect
    p_detect = subparsers.add_parser("detect", help="Run real-time defect detection")
    p_detect.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_detect.add_argument("--frames", type=int, default=30, help="Number of frames to process")
    p_detect.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    p_detect.add_argument("--mock", action="store_true", help="Force synthetic mock engine")
    p_detect.set_defaults(func=cmd_detect)

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark edge inference latency and FPS")
    p_bench.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_bench.add_argument("--runs", type=int, default=50, help="Number of timed test iterations")
    p_bench.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_bench.add_argument("--mock", action="store_true", help="Force synthetic mock engine")
    p_bench.add_argument("--json", action="store_true", help="Print structured JSON metrics")
    p_bench.set_defaults(func=cmd_benchmark)

    # Subcommand: export
    p_export = subparsers.add_parser("export", help="Model optimization and export helper")
    p_export.add_argument("--model", default="weights/yolov8n.pt", help="Source model weights")
    p_export.add_argument("--format", choices=["onnx", "openvino"], default="onnx", help="Export target")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
