from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.config import RowDetectionConfig
from src.models import Row


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return values.astype(np.float32)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def _text_like_mask(band: np.ndarray) -> np.ndarray:
    """Remove obvious rules and tiny specks before tracing the ink path."""

    mask = (band > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    band_height = max(1, band.shape[0])

    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if area < 5:
            continue
        if height / band_height >= 0.95:
            continue
        if width / max(1.0, float(height)) >= 25.0:
            continue
        cleaned[labels == index] = 1

    return cleaned


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        return values
    positions = np.arange(len(values))
    return np.interp(positions, positions[valid], values[valid])


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0:
        return float(np.quantile(values, quantile))
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = min(len(sorted_values) - 1, int(np.searchsorted(cumulative, target)))
    return float(sorted_values[index])


def _limit_steps(values: np.ndarray, max_step: float) -> np.ndarray:
    result = values.astype(np.float32).copy()
    for i in range(1, len(result)):
        result[i] = np.clip(
            result[i], result[i - 1] - max_step, result[i - 1] + max_step
        )
    for i in range(len(result) - 2, -1, -1):
        result[i] = np.clip(
            result[i], result[i + 1] - max_step, result[i + 1] + max_step
        )
    return result


def _row_search_bounds(rows: Sequence[Row], index: int, image_height: int) -> Tuple[int, int]:
    row = rows[index]
    center = (row.bbox.y1 + row.bbox.y2 - 1) / 2.0
    top = row.bbox.y1
    bottom = row.bbox.y2 - 1

    if index > 0:
        previous = rows[index - 1]
        previous_center = (previous.bbox.y1 + previous.bbox.y2 - 1) / 2.0
        top = max(top, int(np.floor((previous_center + center) / 2.0)) + 1)
    if index + 1 < len(rows):
        following = rows[index + 1]
        following_center = (following.bbox.y1 + following.bbox.y2 - 1) / 2.0
        bottom = min(bottom, int(np.ceil((center + following_center) / 2.0)) - 1)

    return max(0, top), min(image_height - 1, max(top, bottom))


def trace_curved_segments(
    binary: np.ndarray,
    rows: List[Row],
    config: RowDetectionConfig,
    ink_confidence: np.ndarray | None = None,
) -> List[Row]:
    """Attach a curved baseline and polygon to every detected row."""

    image_height, image_width = binary.shape[:2]
    sample_count = max(6, int(config.segment_num_samples))
    window_width = max(
        9, int(round(image_width * config.segment_window_width_ratio))
    )
    half_window = window_width // 2

    for row_index, row in enumerate(rows):
        y1, y2 = _row_search_bounds(rows, row_index, image_height)
        band = binary[y1:y2 + 1, :]
        cleaned = _text_like_mask(band)
        if ink_confidence is None:
            confidence_band = cleaned.astype(np.float32)
        else:
            confidence_band = (
                ink_confidence[y1:y2 + 1, :].astype(np.float32) * cleaned
            )
        ink_y, ink_x = np.nonzero(cleaned)

        if len(ink_x) >= 2:
            x_start = int(np.quantile(ink_x, 0.01))
            x_end = int(np.quantile(ink_x, 0.99))
        else:
            x_start, x_end = row.bbox.x1, row.bbox.x2 - 1

        x_start = max(0, min(x_start, image_width - 1))
        x_end = max(x_start + 1, min(x_end, image_width - 1))
        xs = np.linspace(x_start, x_end, sample_count).astype(np.int32)

        baseline = np.full(sample_count, np.nan, dtype=np.float32)
        upper = np.full(sample_count, np.nan, dtype=np.float32)
        lower = np.full(sample_count, np.nan, dtype=np.float32)

        for i, x in enumerate(xs):
            left = max(0, int(x) - half_window)
            right = min(image_width, int(x) + half_window + 1)
            local_y, local_x = np.nonzero(cleaned[:, left:right])
            if len(local_y) == 0:
                continue
            absolute_y = local_y.astype(np.float32) + y1
            weights = confidence_band[local_y, local_x + left]
            weights = np.maximum(weights, 1e-3)
            baseline[i] = _weighted_quantile(
                absolute_y, weights, config.segment_baseline_quantile
            )
            upper[i] = _weighted_quantile(
                absolute_y, weights, config.segment_top_quantile
            )
            lower[i] = _weighted_quantile(
                absolute_y, weights, config.segment_bottom_quantile
            )

        fallback_center = (y1 + y2) / 2.0
        for values, fallback in (
            (baseline, fallback_center),
            (upper, y1),
            (lower, y2),
        ):
            if np.any(np.isfinite(values)):
                values[:] = _interpolate_missing(values)
            else:
                values[:] = fallback

        max_step = max(
            2.0,
            (y2 - y1 + 1) * float(config.segment_max_step_ratio),
        )
        baseline = _limit_steps(baseline, max_step)
        baseline = _smooth(baseline, config.segment_smoothing_window)
        upper = _smooth(upper, config.segment_smoothing_window)
        lower = _smooth(lower, config.segment_smoothing_window)

        min_half = int(config.segment_min_half_height)
        upper = np.minimum(upper, baseline - min_half)
        lower = np.maximum(lower, baseline + min_half)
        upper = np.clip(upper, y1, y2)
        lower = np.clip(lower, y1, y2)

        baseline_points = [
            [int(x), int(round(y))] for x, y in zip(xs, baseline)
        ]
        top_points = [[int(x), int(round(y))] for x, y in zip(xs, upper)]
        bottom_points = [
            [int(x), int(round(y))]
            for x, y in zip(xs[::-1], lower[::-1])
        ]

        row.baseline = baseline_points
        row.polygon = top_points + bottom_points

    return rows
