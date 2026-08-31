import json
from pathlib import Path
from typing import List, Sequence, Tuple

from src.models import Row


def apply_row_exclusions(
    rows: Sequence[Row],
    annotation_path: Path,
) -> Tuple[List[Row], List[Row]]:
    """Apply reviewed Y-range exclusions without baking page IDs into code."""

    if not annotation_path.exists():
        return list(rows), []

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    ranges = data.get("exclude_y_ranges", [])
    kept: List[Row] = []
    excluded: List[Row] = []

    for row in rows:
        center_y = (row.bbox.y1 + row.bbox.y2 - 1) / 2.0
        reject = any(
            float(item["y1"]) <= center_y <= float(item["y2"])
            for item in ranges
        )
        if reject:
            excluded.append(row)
        else:
            kept.append(row)

    return kept, excluded
