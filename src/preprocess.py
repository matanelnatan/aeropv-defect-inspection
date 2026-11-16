"""Frame preprocessing, quality filtering, and mask coordinate restoration utilities."""

import cv2
import numpy as np


def auto_crop_black_borders(frame_bgr: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Remove dark letterboxing/pillarboxing borders from drone video frames."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if mask.mean() < 0.15:
        return frame_bgr
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return frame_bgr[y0:y1, x0:x1]


def compute_sharpness_score(frame_bgr: np.ndarray) -> float:
    """Compute Laplacian variance as a motion-blur quality metric."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def unpad_and_resize_mask(
    pred_padded: np.ndarray, orig_h: int, orig_w: int, target_size: int = 384
) -> np.ndarray:
    """Invert LongestMaxSize + PadIfNeeded transformation back to native frame resolution."""
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    cropped = pred_padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    return cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
