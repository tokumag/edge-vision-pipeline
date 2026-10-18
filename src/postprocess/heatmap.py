"""
Defect spatial density mapping and Gaussian thermal heatmap accumulator.
Useful for identifying recurring stress points and localized defect clusters in manufacturing.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class DefectHeatmap:
    """
    Accumulates defect centroids across frames and computes a spatial density heatmap.
    Applies Gaussian splatting to highlight recurring failure hot-zones on components.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 640),
        decay: float = 1.0,
        gaussian_radius: int = 25,
    ) -> None:
        self.width, self.height = resolution
        self.decay = decay
        self.gaussian_radius = gaussian_radius
        self._accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        self._kernel = self._create_gaussian_kernel(gaussian_radius)

    def _create_gaussian_kernel(self, radius: int) -> np.ndarray:
        """Construct normalized 2D Gaussian circular kernel."""
        size = 2 * radius + 1
        x = np.arange(-radius, radius + 1)
        y = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, y)
        sigma = radius / 2.5
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel.astype(np.float32)

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """
        Incorporate defect bounding boxes into the density grid.
        Ignores 'normal_component' labels to focus exclusively on anomalies.
        """
        if self.decay < 1.0:
            self._accumulator *= self.decay

        k_rad = self.gaussian_radius
        k_size = 2 * k_rad + 1

        for det in detections:
            cname = det.get("class_name", "unknown")
            if cname == "normal_component":
                continue

            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))

            if not (0 <= cx < self.width and 0 <= cy < self.height):
                continue

            # Compute bounding box inside grid
            gx1 = max(0, cx - k_rad)
            gx2 = min(self.width, cx + k_rad + 1)
            gy1 = max(0, cy - k_rad)
            gy2 = min(self.height, cy + k_rad + 1)

            # Compute slice within kernel
            kx1 = gx1 - (cx - k_rad)
            kx2 = kx1 + (gx2 - gx1)
            ky1 = gy1 - (cy - k_rad)
            ky2 = ky1 + (gy2 - gy1)

            confidence = float(det.get("confidence", 1.0))
            self._accumulator[gy1:gy2, gx1:gx2] += (
                self._kernel[ky1:ky2, kx1:kx2] * confidence
            )

    def get_density_map(self) -> np.ndarray:
        """
        Return normalized spatial density matrix in range [0.0, 1.0].
        """
        max_val = float(np.max(self._accumulator))
        if max_val <= 1e-6:
            return np.zeros_like(self._accumulator)
        return self._accumulator / max_val

    def render_overlay(
        self,
        base_frame: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Blend thermal color map onto base image.

        Args:
            base_frame: [H, W, 3] image array (BGR or RGB).
            alpha: Transparency factor (0.0 = only base, 1.0 = only heatmap).

        Returns:
            Blended uint8 array of same shape as base_frame.
        """
        norm_density = self.get_density_map()
        density_u8 = (norm_density * 255).astype(np.uint8)

        if _HAS_CV2:
            heatmap_colored = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)
            if base_frame.shape[:2] != (self.height, self.width):
                heatmap_colored = cv2.resize(
                    heatmap_colored, (base_frame.shape[1], base_frame.shape[0])
                )
            # Suppress black/cold regions in overlay
            mask = (density_u8 > 15).astype(np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]

            blended = (
                base_frame.astype(np.float32) * (1.0 - alpha * mask)
                + heatmap_colored.astype(np.float32) * (alpha * mask)
            )
            return np.clip(blended, 0, 255).astype(np.uint8)
        else:
            # Fallback simple RGB thermal gradient
            r = density_u8
            g = np.clip(255 - np.abs(density_u8.astype(int) - 128) * 2, 0, 255).astype(np.uint8)
            b = 255 - density_u8
            heatmap_colored = np.stack([b, g, r], axis=-1)

            blended = (
                base_frame.astype(np.float32) * (1.0 - alpha)
                + heatmap_colored.astype(np.float32) * alpha
            )
            return np.clip(blended, 0, 255).astype(np.uint8)

    def get_top_hotspots(self, k: int = 3) -> List[Dict[str, Any]]:
        """
        Identify top-k defect cluster locations with coordinate and peak density.
        """
        norm_map = self.get_density_map()
        if np.max(norm_map) <= 1e-4:
            return []

        hotspots = []
        temp_map = norm_map.copy()

        for _ in range(min(k, 10)):
            max_idx = np.unravel_index(np.argmax(temp_map), temp_map.shape)
            peak_val = float(temp_map[max_idx])
            if peak_val < 0.1:
                break

            cy, cx = int(max_idx[0]), int(max_idx[1])
            hotspots.append({
                "center": (cx, cy),
                "density": round(peak_val, 3),
            })

            # Suppress surrounding neighborhood to locate distinct clusters
            r = self.gaussian_radius
            y1 = max(0, cy - r)
            y2 = min(self.height, cy + r + 1)
            x1 = max(0, cx - r)
            x2 = min(self.width, cx + r + 1)
            temp_map[y1:y2, x1:x2] = 0.0

        return hotspots

    def reset(self) -> None:
        """Clear density accumulation."""
        self._accumulator.fill(0.0)
