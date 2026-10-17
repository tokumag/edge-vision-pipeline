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
from typing import Dict, Any

from src.core.stream import SyntheticStream
from src.pipeline import DetectionPipeline
from src.benchmark.profiler import EdgeProfiler
from src.inference.factory import create_engine

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Safely load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_detect(args: argparse.Namespace) -> None:
    """Run real-time edge defect detection on camera, file, or synthetic stream."""
    config = load_config(args.config)
    if args.conf is not None:
        config["model"]["confidence_threshold"] = args.conf

    pipeline = DetectionPipeline(config=config, force_mock=args.mock)
    stream = SyntheticStream(resolution=(640, 640))

    print("\n" + "=" * 60)
    print("  [*] EDGE VISION PIPELINE - REAL-TIME INFERENCE")
    print("=" * 60)
    print(f"Backend:         {pipeline.engine.backend_name.upper()}")
    print(f"Input size:      {config['model']['input_size']}")
    print(f"Conf threshold:  {config['model']['confidence_threshold']}")
    print(f"Frames to run:   {args.frames}")
    print("-" * 60)

    start_time = time.perf_counter()

    for idx in range(args.frames):
        frame = stream.generate_frame()
        res = pipeline.process_frame(frame)

        if (idx + 1) % 10 == 0 or idx == args.frames - 1:
            print(
                f"  Frame [{idx + 1:3d}/{args.frames}] | "
                f"Latency: {res.latency_ms:5.1f}ms | "
                f"FPS: {res.fps:4.1f} | "
                f"Detections: {len(res.detections)}"
            )

    elapsed = time.perf_counter() - start_time
    summary = pipeline.get_summary_metrics()

    print("-" * 60)
    print(f"[OK] Pipeline run complete in {elapsed:.2f}s (Avg {summary['avg_fps']} FPS)")
    print(f"[OK] Median latency: {summary['p50_latency_ms']} ms | p95: {summary['p95_latency_ms']} ms")
    print(f"[OK] Total detections: {summary['total_detections']} | Defect frame rate: {summary['defect_rate']:.1%}")
    print("=" * 60 + "\n")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Execute high-precision hardware inference profiler."""
    config = load_config(args.config)
    engine = create_engine(config["model"], force_mock=args.mock)
    target_size = tuple(config["model"]["input_size"])

    print("\n" + "=" * 60)
    print("  [*] EDGE HARDWARE BENCHMARK & LATENCY PROFILER")
    print("=" * 60)
    print(f"Backend:     {engine.backend_name.upper()}")
    print(f"Resolution:  {target_size}")
    print(f"Iterations:  {args.runs} (warmup: {args.warmup})")
    print("\nExecuting profiling passes...")

    profiler = EdgeProfiler(
        engine=engine,
        warmup_runs=args.warmup,
        test_runs=args.runs,
        input_shape=(1, 3, target_size[0], target_size[1]),
    )
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
