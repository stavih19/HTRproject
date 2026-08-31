from typing import List, Tuple

from src.config import RowDetectionConfig
from src.models import BoundingBox, Row


def merge_nearby_intervals(
    intervals: List[Tuple[int, int]],
    max_gap: int,
) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    merged = [list(intervals[0])]

    for y1, y2 in intervals[1:]:
        prev = merged[-1]
        gap = y1 - prev[1] - 1

        if gap <= max_gap:
            prev[1] = max(prev[1], y2)
        else:
            merged.append([y1, y2])

    return [(a, b) for a, b in merged]


def refine_rows(
    intervals: List[Tuple[int, int]],
    image_width: int,
    image_height: int,
    config: RowDetectionConfig,
) -> List[Row]:
    merged = merge_nearby_intervals(intervals, config.merge_gap)

    filtered = []
    for y1, y2 in merged:
        height = y2 - y1 + 1

        if height < config.min_row_height:
            continue

        # Keep very tall candidates for visual diagnosis rather than silently
        # discarding them; clamp only if an explicit max is configured <= 0.
        if config.max_row_height > 0 and height > config.max_row_height:
            # At this first research stage, keep it. A later splitter can
            # decide whether the candidate contains several logical rows.
            pass

        y1p = max(0, y1 - config.padding_top)
        y2p = min(image_height - 1, y2 + config.padding_bottom)

        filtered.append((y1p, y2p))

    rows = []
    for i, (y1, y2) in enumerate(filtered):
        rows.append(
            Row(
                index=i,
                bbox=BoundingBox(
                    x1=0,
                    y1=int(y1),
                    x2=int(image_width),
                    y2=int(y2 + 1),
                ),
                score=1.0,
            )
        )

    return rows
