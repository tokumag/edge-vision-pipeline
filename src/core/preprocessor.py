"""
High-performance image preprocessor for YOLO and ONNX edge models.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class LetterboxPreprocessor:
    """
    Resizes images while preserving aspect ratio by adding border padding (letterboxing).
    Also handles color conversion (BGR to RGB) and tensor normalization.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        pad_value: int = 114,
        normalize: bool = True,
    ) -> None:
        self.target_w, self.target_h = target_size
        self.pad_value = pad_value
        self.normalize = normalize

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transform raw image to normalized model input tensor [1, 3, H, W].

        Returns:
            tensor: Preprocessed ndarray of shape (1, 3, target_h, target_w)
            meta: Dict containing scale and padding offsets for bbox inverse mapping.
        """
        orig_h, orig_w = img.shape[:2]

        # Calculate scale ratio to fit inside target bounding box
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Padding needed
        pad_x = (self.target_w - new_w) / 2
        pad_y = (self.target_h - new_h) / 2
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))

        # Resize image
        if _HAS_CV2:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(self.pad_value, self.pad_value, self.pad_value)
            )
            # BGR to RGB
            rgb_img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        else:
            # Fallback pure numpy nearest neighbor / padding
            rgb_img = self._numpy_resize_pad(img, new_w, new_h, top, bottom, left, right)

        # Transpose HWC -> CHW
        chw = np.transpose(rgb_img, (2, 0, 1))

        # Convert to float32 and normalize
        tensor = np.ascontiguousarray(chw, dtype=np.float32)
        if self.normalize:
            tensor /= 255.0

        # Add batch dimension: [1, C, H, W]
        tensor = np.expand_dims(tensor, axis=0)

        meta = {
            "orig_shape": (orig_h, orig_w),
            "scale": scale,
            "pad": (left, top),
            "target_shape": (self.target_h, self.target_w),
        }

        return tensor, meta

    def _numpy_resize_pad(
        self, img: np.ndarray, new_w: int, new_h: int, top: int, bottom: int, left: int, right: int
    ) -> np.ndarray:
        """Pure numpy fallback for resizing and padding."""
        h, w = img.shape[:2]
        # Nearest neighbor indexing
        y_indices = (np.linspace(0, h - 1, new_h)).astype(int)
        x_indices = (np.linspace(0, w - 1, new_w)).astype(int)
        resized = img[y_indices[:, None], x_indices]

        # Allocate padded array
        total_h = new_h + top + bottom
        total_w = new_w + left + right
        padded = np.full((total_h, total_w, 3), self.pad_value, dtype=np.uint8)
        padded[top : top + new_h, left : left + new_w] = resized

        # Swap channel 0 and 2 (BGR to RGB)
        return padded[:, :, ::-1]

    @staticmethod
    def scale_boxes_back(
        boxes: np.ndarray, meta: Dict[str, Any]
    ) -> np.ndarray:
        """
        Rescale predicted boxes [x1, y1, x2, y2] from letterboxed image
        back to original image coordinates.
        """
        if len(boxes) == 0:
            return boxes

        boxes_scaled = boxes.copy()
        pad_x, pad_y = meta["pad"]
        scale = meta["scale"]
        orig_h, orig_w = meta["orig_shape"]

        boxes_scaled[:, [0, 2]] = (boxes_scaled[:, [0, 2]] - pad_x) / scale
        boxes_scaled[:, [1, 3]] = (boxes_scaled[:, [1, 3]] - pad_y) / scale

        # Clip coordinates within original boundaries
        boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, orig_w)
        boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, orig_h)

        return boxes_scaled
