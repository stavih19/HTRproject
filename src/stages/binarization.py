import cv2
import numpy as np

from src.config import BinarizationConfig
from src.image_utils import ensure_odd


def create_binary_mask(
    gray: np.ndarray,
    config: BinarizationConfig,
) -> np.ndarray:
    """Create an analysis-only foreground mask.

    White pixels = likely ink / foreground.
    The original image is never modified.
    """
    block_size = ensure_odd(config.adaptive_block_size, minimum=3)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        config.adaptive_c,
    )

    return binary


def _two_means(values: np.ndarray) -> tuple[float, float]:
    """Split local contrast values into faint and strong populations."""

    low, high = np.percentile(values, [30.0, 75.0]).astype(np.float32)
    for _ in range(12):
        midpoint = (low + high) / 2.0
        low_values = values[values <= midpoint]
        high_values = values[values > midpoint]
        if len(low_values) == 0 or len(high_values) == 0:
            break
        new_low = float(np.mean(low_values))
        new_high = float(np.mean(high_values))
        if abs(new_low - low) + abs(new_high - high) < 0.05:
            low, high = new_low, new_high
            break
        low, high = new_low, new_high
    return float(min(low, high)), float(max(low, high))


def create_front_ink_mask(
    gray: np.ndarray,
    permissive_mask: np.ndarray,
    config: BinarizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate front-page ink relative to the ink strength in each region.

    Returns a uint8 mask and a float32 confidence map in [0, 1]. The existing
    permissive mask remains the source for row discovery.
    """

    block_size = ensure_odd(config.adaptive_block_size, minimum=3)
    background = cv2.GaussianBlur(gray, (block_size, block_size), 0)
    contrast = np.maximum(
        background.astype(np.float32) - gray.astype(np.float32), 0.0
    )

    h, w = gray.shape[:2]
    tile_size = max(32, int(config.front_ink_tile_size))
    overlap = float(np.clip(config.front_ink_tile_overlap, 0.0, 0.9))
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    confidence_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    # Soft tile edges prevent visible threshold seams between windows.
    axis_weight = np.hanning(tile_size).astype(np.float32)
    axis_weight = np.maximum(axis_weight, 0.05)
    full_weight = np.outer(axis_weight, axis_weight)

    for y1 in range(0, h, step):
        y2 = min(h, y1 + tile_size)
        y1 = max(0, y2 - tile_size)
        for x1 in range(0, w, step):
            x2 = min(w, x1 + tile_size)
            x1 = max(0, x2 - tile_size)

            local_contrast = contrast[y1:y2, x1:x2]
            local_permissive = permissive_mask[y1:y2, x1:x2] > 0
            values = local_contrast[local_permissive]
            values = values[values > 0]

            if len(values) >= config.front_ink_min_samples:
                faint_mean, strong_mean = _two_means(values)
                separation = strong_mean - faint_mean
                if separation < config.front_ink_min_cluster_separation:
                    faint_mean = float(np.percentile(values, 35.0))
                    strong_mean = float(np.percentile(values, 80.0))
                scale = max(
                    strong_mean - faint_mean,
                    float(config.front_ink_min_cluster_separation),
                )
                local_confidence = np.clip(
                    (local_contrast - faint_mean) / scale, 0.0, 1.0
                )
            else:
                # Sparse regions are kept neutral rather than being forced
                # into a possibly incorrect front/back classification.
                local_confidence = local_permissive.astype(np.float32) * 0.5

            local_confidence *= local_permissive
            weights = full_weight[: y2 - y1, : x2 - x1]
            confidence_sum[y1:y2, x1:x2] += local_confidence * weights
            weight_sum[y1:y2, x1:x2] += weights

            if x2 == w:
                break
        if y2 == h:
            break

    confidence = confidence_sum / np.maximum(weight_sum, 1e-6)
    confidence *= permissive_mask > 0

    strong = confidence >= config.front_ink_strong_confidence
    weak = (
        (confidence >= config.front_ink_weak_confidence)
        & (permissive_mask > 0)
    ).astype(np.uint8)

    # Hysteresis: retain weak stroke pixels only when their connected
    # component contains a strong front-ink seed.
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    keep_labels = np.unique(labels[strong])
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels):
        front_mask = np.isin(labels, keep_labels).astype(np.uint8) * 255
    else:
        front_mask = np.zeros_like(permissive_mask)

    return front_mask, confidence.astype(np.float32)
