"""
Edge Computer Vision Pipeline
=============================
High-performance real-time object & defect detection optimized for edge hardware.
"""

from src.pipeline import DetectionPipeline, PipelineResult
from src.metrics import MetricsCollector

__version__ = "1.0.0"
__author__ = "Tokuma Gebissa"

__all__ = ["DetectionPipeline", "PipelineResult", "MetricsCollector"]
