"""
Vectorized Non-Maximum Suppression (NMS) and YOLO tensor decoder.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class YOLOOutputDecoder:
    """
    Decodes raw model tensors (e.g. YOLOv8 format [1, 4 + num_classes, num_anchors])
    into filtered bounding boxes, class indices, and confidence scores.
    """

    def __init__(
        self,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        class_names: Dict[int, str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

    def decode(
        self, raw_output: np.ndarray, meta: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Decodes raw output into candidate detections.

        Args:
            raw_output: Tensor of shape [1, 4 + C, 8400] or [1, 8400, 4 + C]
            meta: Preprocessor metadata dict containing scaling and padding.

        Returns:
            List of detection dicts: [{"bbox": [x1, y1, x2, y2], "confidence": 0.85, "class_id": 1, "class_name": "structural_crack"}]
        """
        # Ensure format is [num_anchors, 4 + C]
        if raw_output.ndim == 3:
            raw_output = raw_output[0]

        if raw_output.shape[0] < raw_output.shape[1]:
            # [4 + C, num_anchors] -> transpose to [num_anchors, 4 + C]
            preds = raw_output.T
        else:
            preds = raw_output

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert cx, cy, w, h to x1, y1, x2, y2 in letterbox coordinates
        boxes_xyxy = np.empty_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Apply multi-class NMS
        keep_indices = self.non_max_suppression(boxes_xyxy, confidences, class_ids, self.iou_threshold)
        if len(keep_indices) == 0:
            return []

        final_boxes = boxes_xyxy[keep_indices]
        final_scores = confidences[keep_indices]
        final_classes = class_ids[keep_indices]

        # Scale coordinates back to original image space
        from src.core.preprocessor import LetterboxPreprocessor
        scaled_boxes = LetterboxPreprocessor.scale_boxes_back(final_boxes, meta)

        detections = []
        for i in range(len(scaled_boxes)):
            cid = int(final_classes[i])
            cname = self.class_names.get(cid, f"class_{cid}")
            detections.append({
                "bbox": [round(float(v), 1) for v in scaled_boxes[i]],
                "confidence": round(float(final_scores[i]), 3),
                "class_id": cid,
                "class_name": cname,
            })

        return detections

    @staticmethod
    def non_max_suppression(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """Vectorized class-aware Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            current_class = class_ids[i]
            other_indices = order[1:]

            # Only compare IoU against boxes of the same class
            same_class_mask = class_ids[other_indices] == current_class

            xx1 = np.maximum(x1[i], x1[other_indices])
            yy1 = np.maximum(y1[i], y1[other_indices])
            xx2 = np.minimum(x2[i], x2[other_indices])
            yy2 = np.minimum(y2[i], y2[other_indices])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[other_indices] - inter + 1e-6)

            # Suppress if IoU > threshold AND of same class
            suppress = (iou > iou_threshold) & same_class_mask
            order = other_indices[~suppress]

        return keep
