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
        # A table rule spans nearly the whole band and is very narrow.  Do
        # not reject a component merely because it spans the band: digits in
        # the right-hand row-number column can touch both search bounds too.
        spans_band_height = height / band_height >= 0.95
        rule_max_width = max(3, int(round(band_height * 0.20)))
        if spans_band_height and width <= rule_max_width:
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
        if previous.bbox.y2 > row.bbox.y1:
            previous_center = (previous.bbox.y1 + previous.bbox.y2 - 1) / 2.0
            top = max(top, int(np.floor((previous_center + center) / 2.0)) + 1)
    if index + 1 < len(rows):
        following = rows[index + 1]
        if row.bbox.y2 > following.bbox.y1:
            following_center = (following.bbox.y1 + following.bbox.y2 - 1) / 2.0
            bottom = min(bottom, int(np.ceil((center + following_center) / 2.0)) - 1)

    return max(0, top), min(image_height - 1, max(top, bottom))


def _stroke_tracking_bounds(
    rows: Sequence[Row], index: int, image_height: int
) -> Tuple[int, int]:
    """Allow stroke following into whitespace, but never past row midpoints."""

    row = rows[index]
    center = (row.bbox.y1 + row.bbox.y2 - 1) / 2.0
    top = row.bbox.y1
    bottom = row.bbox.y2 - 1
    if index > 0:
        previous = rows[index - 1]
        previous_center = (previous.bbox.y1 + previous.bbox.y2 - 1) / 2.0
        top = int(np.floor((previous_center + center) / 2.0)) + 1
    if index + 1 < len(rows):
        following = rows[index + 1]
        following_center = (following.bbox.y1 + following.bbox.y2 - 1) / 2.0
        bottom = int(np.ceil((center + following_center) / 2.0)) - 1
    return max(0, top), min(image_height - 1, max(top, bottom))


def _follow_row_strokes(
    binary: np.ndarray,
    rows: Sequence[Row],
    index: int,
    gap_threshold: int,
) -> Tuple[np.ndarray, int, int]:
    """Keep ink connected (allowing a short break) to the detected row."""

    row = rows[index]
    y1, y2 = _stroke_tracking_bounds(rows, index, binary.shape[0])
    cleaned = _text_like_mask(binary[y1:y2 + 1, :])
    cleaned[:, : row.bbox.x1] = 0
    cleaned[:, row.bbox.x2 :] = 0

    gap = max(0, int(gap_threshold))
    if gap > 0:
        # A kernel of gap+1 lets two stroke fragments bridge a blank interval
        # up to `gap` pixels without retaining the artificial bridge itself.
        size = gap + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        connected_mask = cv2.dilate(cleaned, kernel, iterations=1)
    else:
        connected_mask = cleaned

    _, labels = cv2.connectedComponents(
        (connected_mask > 0).astype(np.uint8), connectivity=8
    )
    seed_top = max(0, row.bbox.y1 - y1)
    seed_bottom = min(cleaned.shape[0], row.bbox.y2 - y1)
    seed = cleaned[seed_top:seed_bottom, row.bbox.x1:row.bbox.x2] > 0
    seed_label_view = labels[
        seed_top:seed_bottom, row.bbox.x1:row.bbox.x2
    ]
    selected_labels = np.unique(seed_label_view[seed])
    selected_labels = selected_labels[selected_labels != 0]
    if len(selected_labels) == 0:
        return cleaned, y1, y2

    followed = cleaned * np.isin(labels, selected_labels).astype(np.uint8)
    return followed, y1, y2


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
        cleaned, y1, y2 = _follow_row_strokes(
            binary,
            rows,
            row_index,
            config.segment_stroke_gap_threshold,
        )
        if ink_confidence is None:
            confidence_band = cleaned.astype(np.float32)
        else:
            confidence_band = (
                ink_confidence[y1:y2 + 1, :].astype(np.float32) * cleaned
            )
        ink_y, ink_x = np.nonzero(cleaned)

        if len(ink_x) >= 2:
            x_start = int(np.quantile(ink_x, 0.01))
            # The row number at the far right is usually much smaller than
            # the main handwritten entry.  A 99th-percentile endpoint can
            # therefore discard the entire digit even after it has passed
            # the component filtering above.  Use the true right edge of
            # the cleaned, text-like ink so RTL segments begin at that digit.
            x_end = int(np.max(ink_x))
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
            # The polygon is an inclusion envelope, not a robust estimate of
            # the typical character height.  Quantiles clipped uncommon tall
            # ascenders, descenders and small row-number digits.  The mask has
            # already had rules and specks removed, so retain its full local
            # vertical extent here.
            upper[i] = float(np.min(absolute_y))
            lower[i] = float(np.max(absolute_y))

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
        raw_upper = upper.copy()
        raw_lower = lower.copy()
        upper = np.minimum(
            _smooth(upper, config.segment_smoothing_window), raw_upper
        )
        lower = np.maximum(
            _smooth(lower, config.segment_smoothing_window), raw_lower
        )

        min_half = int(config.segment_min_half_height)
        upper = np.minimum(upper, baseline - min_half)
        lower = np.maximum(lower, baseline + min_half)
        upper = np.clip(upper, y1, y2)
        lower = np.clip(lower, y1, y2)

        baseline_points = [
            [int(x), int(round(y))] for x, y in zip(xs, baseline)
        ]

        # Build the polygon at pixel-column resolution.  The sampled envelope
        # above is useful for stable baseline estimation, but connecting only
        # those samples with straight lines can cut through a narrow tip that
        # falls between them.  Per-column extrema form a contour that follows
        # every retained letter stroke all the way to its end.
        contour_xs = np.arange(x_start, x_end + 1, dtype=np.int32)
        contour_upper = np.full(len(contour_xs), np.nan, dtype=np.float32)
        contour_lower = np.full(len(contour_xs), np.nan, dtype=np.float32)
        for contour_index, x in enumerate(contour_xs):
            column_y = np.flatnonzero(cleaned[:, x])
            if len(column_y) == 0:
                continue
            contour_upper[contour_index] = float(column_y[0] + y1)
            contour_lower[contour_index] = float(column_y[-1] + y1)

        contour_upper = _interpolate_missing(contour_upper)
        contour_lower = _interpolate_missing(contour_lower)
        contour_baseline = np.interp(contour_xs, xs, baseline)
        contour_upper = np.minimum(
            contour_upper, contour_baseline - min_half
        )
        contour_lower = np.maximum(
            contour_lower, contour_baseline + min_half
        )
        contour_upper = np.clip(contour_upper, y1, y2)
        contour_lower = np.clip(contour_lower, y1, y2)

        top_points = [
            [int(x), int(round(y))]
            for x, y in zip(contour_xs, contour_upper)
        ]
        bottom_points = [
            [int(x), int(round(y))]
            for x, y in zip(contour_xs[::-1], contour_lower[::-1])
        ]

        row.baseline = baseline_points
        row.polygon = top_points + bottom_points

    return rows
