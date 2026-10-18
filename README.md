# Edge Vision Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/runtime-ONNX--Runtime-purple.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-24%2F24%20passing-brightgreen.svg)](tests/)

An end-to-end computer vision and industrial defect detection pipeline optimized for low-latency edge deployment. Built with **ONNX Runtime**, **vectorized multi-class NMS**, **centroid/IoU tracking**, **spatial defect density heatmaps**, and a **clean Streamlit monitoring interface**.

---

## Architecture Overview

```text
+-------------------+      +----------------------+      +----------------------+
|   Video Stream    | ---> | Thread-Safe Capture  | ---> | Letterbox Resizer    |
| (Webcam/File/AOI) |      | Non-blocking Buffer  |      | Normalize (0-1.0)    |
+-------------------+      +----------------------+      +----------------------+
                                                                    |
                                                                    v
+-------------------+      +----------------------+      +----------------------+
| Real-Time Display | <--- | Centroid & IoU       | <--- | ONNX Runtime Engine  |
| Streamlit / CLI   |      | Persistent Tracker   |      | Vectorized NMS       |
+-------------------+      +----------------------+      +----------------------+
```

### Core Engineering Design
- **Unified Pipeline API**: `DetectionPipeline` encapsulates preprocessing, deep learning inference, Non-Maximum Suppression, and tracking into a single call returning a typed `PipelineResult`.
- **Decoupled Frame Capture**: Background threading prevents video source buffering and eliminates latency queuing on high-speed feeds.
- **Aspect-Ratio Preserving Letterboxing**: Tracks sub-pixel coordinate scaling so predicted bounding boxes map accurately back to original image dimensions.
- **Vectorized NMS**: Multi-class Non-Maximum Suppression implemented purely in NumPy for maximum edge throughput without PyTorch overhead.
- **Spatial Defect Density Heatmap**: Gaussian accumulation identifying localized recurrent failure hotspots across conveyor belts and manufacturing dies.
- **Media Upload & Live Inspection**: Inspect live video streams or upload custom images (.jpg, .png) for immediate optical inspection and hot-zone analysis.
- **Hardware Telemetry Profiler**: Precision edge benchmarking measuring tail latencies ($p_{50}, p_{90}, p_{95}, p_{99}$), memory RSS, and throughput (FPS).

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/tokumag/edge-vision-pipeline.git
cd edge-vision-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Install dependencies (or install as package in editable mode)
pip install -e .
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with the live inspection loop, upload images, inspect defect heatmaps, and run the hardware profiler.

### 3. CLI Commands

```bash
# Real-time inspection (synthetic conveyor feed)
python -m src.cli detect --mock --frames 30

# Hardware latency profiler
python -m src.cli benchmark --mock --runs 100 --warmup 15

# Export model helper
python -m src.cli export --model weights/yolov8n.pt --format onnx
```

---

## Automated Test Suite

The test suite covers streaming, letterbox preprocessing, inference backends, vectorized NMS, tracking persistence, spatial defect heatmaps, hardware profiling, and telemetry collection:

```bash
pytest tests/ -v
```

```text
tests/test_heatmap.py::TestDefectHeatmap::test_heatmap_initialization PASSED
tests/test_heatmap.py::TestDefectHeatmap::test_heatmap_accumulation_ignores_normal PASSED
tests/test_heatmap.py::TestDefectHeatmap::test_heatmap_accumulation_defect PASSED
tests/test_heatmap.py::TestDefectHeatmap::test_heatmap_hotspots_identification PASSED
tests/test_heatmap.py::TestDefectHeatmap::test_render_overlay_preserves_shape PASSED
tests/test_heatmap.py::TestDefectHeatmap::test_reset_clears_density PASSED
tests/test_pipeline.py::TestStream::test_synthetic_stream_shape PASSED
tests/test_pipeline.py::TestStream::test_stream_generator_count PASSED
tests/test_pipeline.py::TestStream::test_synthetic_stream_deterministic PASSED
tests/test_pipeline.py::TestPreprocessor::test_letterbox_output_shape PASSED
tests/test_pipeline.py::TestPreprocessor::test_scale_boxes_back_identity PASSED
tests/test_pipeline.py::TestPreprocessor::test_non_square_high_res_input PASSED
tests/test_pipeline.py::TestPreprocessor::test_empty_boxes_scale_back PASSED
tests/test_pipeline.py::TestInferenceEngine::test_mock_engine_infer PASSED
tests/test_pipeline.py::TestInferenceEngine::test_mock_engine_warmup_no_error PASSED
tests/test_pipeline.py::TestInferenceEngine::test_engine_factory_mock PASSED
tests/test_pipeline.py::TestPostprocess::test_nms_suppression PASSED
tests/test_pipeline.py::TestPostprocess::test_nms_different_classes_retained PASSED
tests/test_pipeline.py::TestPostprocess::test_decoder_high_threshold_empty PASSED
tests/test_pipeline.py::TestTracker::test_tracker_assignment PASSED
tests/test_pipeline.py::TestTracker::test_tracker_distant_detection_new_id PASSED
tests/test_pipeline.py::TestProfiler::test_benchmark_run PASSED
tests/test_pipeline.py::TestPipelineAndMetrics::test_detection_pipeline_end_to_end PASSED
tests/test_pipeline.py::TestPipelineAndMetrics::test_metrics_collector PASSED

============================= 24 passed in 2.55s =============================
```

---

## Project Structure

```text
edge-vision-pipeline/
|-- app.py                     # Streamlit live inspection, media upload & benchmark dashboard
|-- pyproject.toml             # PEP 621 package metadata & configuration
|-- Dockerfile                 # Containerized edge deployment
|-- docker-compose.yml         # Container service configuration
|-- configs/
|   `-- default.yaml           # Pipeline, model, camera, and class taxonomy
|-- src/
|   |-- __init__.py            # Root package exports
|   |-- pipeline.py            # Unified DetectionPipeline orchestrator
|   |-- metrics.py             # MetricsCollector telemetry aggregator
|   |-- cli.py                 # Production CLI (detect, benchmark, export)
|   |-- core/
|   |   |-- stream.py          # Threaded camera & synthetic stream generator
|   |   `-- preprocessor.py    # Letterbox aspect-ratio preserving preprocessor
|   |-- inference/
|   |   |-- base.py            # BaseEngine abstract interface
|   |   |-- onnx_engine.py     # ONNX Runtime provider with sequential threads
|   |   |-- mock_engine.py     # Deterministic mock engine for CI/CD & testing
|   |   `-- factory.py         # Dynamic engine factory with fallback
|   |-- postprocess/
|   |   |-- nms.py             # Vectorized multi-class Non-Maximum Suppression
|   |   |-- tracker.py         # Centroid & IoU persistent defect tracker
|   |   |-- heatmap.py         # Spatial defect density & Gaussian thermal accumulator
|   |   `-- visualizer.py      # HUD telemetry overlay & bounding box renderer
|   `-- benchmark/
|       `-- profiler.py        # Latency percentiles (p50-p99) & memory profiler
|-- tests/
|   |-- test_heatmap.py        # DefectHeatmap density & overlay tests
|   `-- test_pipeline.py       # End-to-end pipeline & component unit tests
`-- weights/                   # Checkpoint storage (.onnx / .pt)
```

---

## Docker Deployment

```bash
docker-compose up --build
```
Access the dashboard at `http://localhost:8501`.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Tokuma Gebissa** — AI Engineering Student
- GitHub: [@tokumag](https://github.com/tokumag)
- Email: tokuma.gebissa-ug@aau.edu.et
