"""
Lightweight edge tracker using Centroid and IoU association.
Maintains continuity of detected components and defect events across frames.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class EdgeTracker:
    """
    Real-time multi-object tracker optimized for minimal CPU overhead.
    Maintains active track IDs and computes defect duration/recurrence.
    """

    def __init__(self, max_lost_frames: int = 10, iou_distance_thresh: float = 0.3) -> None:
        self.max_lost_frames = max_lost_frames
        self.iou_distance_thresh = iou_distance_thresh
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Associate detections to existing tracks or create new tracks.

        Returns:
            List of detections enriched with 'track_id' and 'track_age'.
        """
        # Increment lost frame count for existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["lost"] += 1
            if self.tracks[tid]["lost"] > self.max_lost_frames:
                del self.tracks[tid]

        if not detections:
            return []

        if not self.tracks:
            # Initialize tracks for all incoming detections
            for det in detections:
                det["track_id"] = self.next_track_id
                det["track_age"] = 1
                self.tracks[self.next_track_id] = {
                    "bbox": det["bbox"],
                    "class_id": det["class_id"],
                    "lost": 0,
                    "age": 1,
                }
                self.next_track_id += 1
            return detections

        # Compute IoU cost matrix between current detections and active tracks
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid]["bbox"] for tid in track_ids])
        det_boxes = np.array([det["bbox"] for det in detections])

        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)

        matched_dets = set()
        matched_tracks = set()

        # Greedy association by highest IoU
        for _ in range(min(len(detections), len(track_ids))):
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            d_idx, t_idx = max_idx[0], max_idx[1]
            max_iou = iou_matrix[d_idx, t_idx]

            if max_iou < self.iou_distance_thresh:
                break

            tid = track_ids[t_idx]
            detections[d_idx]["track_id"] = tid
            self.tracks[tid]["age"] += 1
            self.tracks[tid]["lost"] = 0
            self.tracks[tid]["bbox"] = detections[d_idx]["bbox"]
            detections[d_idx]["track_age"] = self.tracks[tid]["age"]

            matched_dets.add(d_idx)
            matched_tracks.add(tid)

            # Invalidate row and column
            iou_matrix[d_idx, :] = -1.0
            iou_matrix[:, t_idx] = -1.0

        # Unmatched detections get newly allocated IDs
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                det["track_id"] = self.next_track_id
                det["track_age"] = 1
                self.tracks[self.next_track_id] = {
                    "bbox": det["bbox"],
                    "class_id": det["class_id"],
                    "lost": 0,
                    "age": 1,
                }
                self.next_track_id += 1

        return detections

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Calculate pairwise IoU matrix."""
        num_a, num_b = len(boxes_a), len(boxes_b)
        matrix = np.zeros((num_a, num_b), dtype=np.float32)

        for i in range(num_a):
            x1_a, y1_a, x2_a, y2_a = boxes_a[i]
            area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)

            for j in range(num_b):
                x1_b, y1_b, x2_b, y2_b = boxes_b[j]
                area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

                xx1 = max(x1_a, x1_b)
                yy1 = max(y1_a, y1_b)
                xx2 = min(x2_a, x2_b)
                yy2 = min(y2_a, y2_b)

                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h

                union = area_a + area_b - inter
                matrix[i, j] = inter / max(union, 1e-6)

        return matrix
