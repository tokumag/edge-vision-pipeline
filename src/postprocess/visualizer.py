"""
Bounding box annotation and performance telemetry overlay visualizer.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


DEFAULT_COLORS = {
    "normal_component": (46, 204, 113),  # Green
    "surface_scratch": (241, 196, 15),   # Yellow
    "structural_crack": (231, 76, 60),   # Red
    "dent_defect": (230, 126, 34),       # Orange
    "misalignment": (155, 89, 182),      # Purple
    "foreign_particle": (52, 152, 219),  # Blue
}


class FrameVisualizer:
    """
    Renders styled bounding boxes, diagnostic HUDs, latency telemetry, and FPS metrics.
    """

    def __init__(
        self,
        color_map: Dict[str, Tuple[int, int, int]] = None,
        draw_fps: bool = True,
        draw_latency: bool = True,
    ) -> None:
        self.color_map = color_map or DEFAULT_COLORS
        self.draw_fps = draw_fps
        self.draw_latency = draw_latency

    def render(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float = 0.0,
        latency_ms: float = 0.0,
        backend: str = "onnx",
    ) -> np.ndarray:
        """
        Draw bounding boxes and status HUD onto the frame.
        """
        out_frame = frame.copy()

        # Render bounding boxes
        for det in detections:
            bbox = det["bbox"]
            cname = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            track_id = det.get("track_id")

            color = self.color_map.get(cname, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in bbox]

            label = f"#{track_id} {cname} {conf:.0%}" if track_id else f"{cname} {conf:.0%}"

            if _HAS_CV2:
                # Main bounding box
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)

                # Label tag banner
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                tag_y1 = max(0, y1 - text_h - 6)
                tag_y2 = y1
                cv2.rectangle(out_frame, (x1, tag_y1), (x1 + text_w + 8, tag_y2), color, -1)
                cv2.putText(
                    out_frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
            else:
                # Basic array boundary drawing for environments without cv2
                h, w = out_frame.shape[:2]
                x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
                y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                out_frame[y1 : y1 + 2, x1:x2] = color
                out_frame[y2 - 2 : y2, x1:x2] = color
                out_frame[y1:y2, x1 : x1 + 2] = color
                out_frame[y1:y2, x2 - 2 : x2] = color

        # Render Diagnostics HUD
        if _HAS_CV2:
            hud_text = f"Backend: {backend.upper()} | FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Objects: {len(detections)}"
            cv2.rectangle(out_frame, (10, 10), (520, 42), (25, 25, 25), -1)
            cv2.putText(
                out_frame, hud_text, (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 1, cv2.LINE_AA
            )

        return out_frame
