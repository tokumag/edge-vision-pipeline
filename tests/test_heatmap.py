"""
Unit tests for DefectHeatmap spatial density accumulator.
"""

import numpy as np
import pytest

from src.postprocess.heatmap import DefectHeatmap


class TestDefectHeatmap:
    def test_heatmap_initialization(self):
        hm = DefectHeatmap(resolution=(640, 480), gaussian_radius=20)
        assert hm.width == 640
        assert hm.height == 480
        assert hm.get_density_map().shape == (480, 640)
        assert np.max(hm.get_density_map()) == 0.0

    def test_heatmap_accumulation_ignores_normal(self):
        hm = DefectHeatmap(resolution=(640, 640), gaussian_radius=20)
        # Normal component should NOT add to defect heatmap
        normal_dets = [{"bbox": [100, 100, 200, 200], "class_name": "normal_component"}]
        hm.update(normal_dets)
        assert np.max(hm.get_density_map()) == 0.0

    def test_heatmap_accumulation_defect(self):
        hm = DefectHeatmap(resolution=(640, 640), gaussian_radius=20)
        defect_dets = [{"bbox": [200, 200, 300, 300], "class_name": "structural_crack", "confidence": 0.9}]
        hm.update(defect_dets)

        density = hm.get_density_map()
        assert np.max(density) == pytest.approx(1.0, rel=1e-3)
        # Peak density should be near centroid (250, 250)
        assert density[250, 250] > 0.8

    def test_heatmap_hotspots_identification(self):
        hm = DefectHeatmap(resolution=(640, 640), gaussian_radius=25)
        # Add clustered defects around (400, 400)
        for _ in range(3):
            hm.update([{"bbox": [380, 380, 420, 420], "class_name": "dent_defect", "confidence": 0.9}])

        hotspots = hm.get_top_hotspots(k=2)
        assert len(hotspots) >= 1
        cx, cy = hotspots[0]["center"]
        assert abs(cx - 400) <= 2
        assert abs(cy - 400) <= 2
        assert hotspots[0]["density"] == pytest.approx(1.0, rel=1e-2)

    def test_render_overlay_preserves_shape(self):
        hm = DefectHeatmap(resolution=(640, 480), gaussian_radius=20)
        base = np.zeros((480, 640, 3), dtype=np.uint8)
        hm.update([{"bbox": [100, 100, 150, 150], "class_name": "surface_scratch", "confidence": 0.8}])

        overlay = hm.render_overlay(base, alpha=0.5)
        assert overlay.shape == base.shape
        assert overlay.dtype == np.uint8

    def test_reset_clears_density(self):
        hm = DefectHeatmap(resolution=(320, 320))
        hm.update([{"bbox": [50, 50, 100, 100], "class_name": "misalignment", "confidence": 0.8}])
        assert np.max(hm.get_density_map()) > 0.0

        hm.reset()
        assert np.max(hm.get_density_map()) == 0.0
