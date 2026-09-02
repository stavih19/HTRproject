from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.config import ColumnConfig
from src.models import BoundingBox, Row


@dataclass
class ColumnAnalysis:
    bounds: List[Tuple[int, int]]
    gutters: List[Tuple[int, int]]
    projection: np.ndarray
    used_fallback: bool = False


def detect_columns(front_ink: np.ndarray, config: ColumnConfig) -> ColumnAnalysis:
    """Find semantic column bounds; returned order follows reading order."""

    height, width = front_ink.shape[:2]
    count = max(1, int(config.expected_columns))
    projection = np.sum(front_ink > 0, axis=0).astype(np.float32)

    if config.detection_mode == "fixed" and config.fixed_boundaries:
        physical = [
            (max(0, int(round(a * width))), min(width, int(round(b * width))))
            for a, b in config.fixed_boundaries
            if b > a
        ]
        fallback = False
        gutters: List[Tuple[int, int]] = []
    elif count == 1:
        physical = [(0, width)]
        gutters = []
        fallback = False
    else:
        smooth_width = max(3, int(round(width * config.min_gutter_width_ratio)))
        kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
        smoothed = np.convolve(projection, kernel, mode="same")
        half_gutter = max(1, smooth_width // 2)
        split_points = []

        # Search around each expected equal-width division. This prevents a
        # narrow index inside a main column from becoming another column.
        for divider in range(1, count):
            expected = divider * width / count
            radius = width / (count * 3.0)
            left = max(half_gutter, int(round(expected - radius)))
            right = min(width - half_gutter, int(round(expected + radius)))
            if right <= left:
                continue
            split_points.append(left + int(np.argmin(smoothed[left:right])))

        if len(split_points) != count - 1:
            return ColumnAnalysis([(0, width)], [], projection, used_fallback=True)

        split_points.sort()
        gutters = [
            (max(0, point - half_gutter), min(width, point + half_gutter))
            for point in split_points
        ]
        physical = []
        start = 0
        for gutter_left, gutter_right in gutters:
            physical.append((start, gutter_left))
            start = gutter_right
        physical.append((start, width))
        fallback = any(x2 <= x1 for x1, x2 in physical)
        if fallback:
            return ColumnAnalysis([(0, width)], [], projection, used_fallback=True)

    if config.reading_order.lower() == "rtl":
        physical = list(reversed(physical))
    return ColumnAnalysis(physical, gutters, projection, used_fallback=fallback)


def split_rows_by_columns(
    rows: Sequence[Row],
    front_ink: np.ndarray,
    columns: ColumnAnalysis,
    config: ColumnConfig,
) -> List[List[Row]]:
    """Create independent row candidates only where a column contains ink."""

    image_height, image_width = front_ink.shape[:2]
    padding = max(0, int(round(image_width * config.column_padding_ratio)))
    result: List[List[Row]] = []
    global_index = 0

    for column_index, (column_x1, column_x2) in enumerate(columns.bounds):
        column_rows: List[Row] = []
        for source_row in rows:
            y1, y2 = source_row.bbox.y1, source_row.bbox.y2
            band = front_ink[y1:y2, column_x1:column_x2] > 0
            ys, xs = np.nonzero(band)
            if len(xs) < config.min_segment_ink:
                continue
            span = int(xs.max() - xs.min() + 1)
            if span < (column_x2 - column_x1) * config.min_segment_width_ratio:
                continue

            ink_x1 = column_x1 + int(np.quantile(xs, 0.01))
            ink_x2 = column_x1 + int(np.quantile(xs, 0.99)) + 1
            x1 = max(column_x1, ink_x1 - padding)
            x2 = min(column_x2, ink_x2 + padding)
            row = Row(
                index=global_index,
                bbox=BoundingBox(x1, y1, max(x1 + 1, x2), y2),
                score=source_row.score,
                column_index=column_index,
                column_row_index=len(column_rows),
                source_row_index=source_row.index,
            )
            column_rows.append(row)
            global_index += 1
        result.append(column_rows)

    return result
