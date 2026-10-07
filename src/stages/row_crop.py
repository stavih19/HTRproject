from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.models import Row


def _segmented_crop(original: np.ndarray, row: Row) -> np.ndarray:
    """Crop to the segment polygon and whiten pixels outside its envelope."""

    if not row.polygon or len(row.polygon) < 3:
        b = row.bbox
        return original[b.y1:b.y2, b.x1:b.x2].copy()

    height, width = original.shape[:2]
    polygon = np.asarray(row.polygon, dtype=np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    x1 = int(np.min(polygon[:, 0]))
    y1 = int(np.min(polygon[:, 1]))
    x2 = int(np.max(polygon[:, 0])) + 1
    y2 = int(np.max(polygon[:, 1])) + 1

    crop = original[y1:y2, x1:x2].copy()
    local_polygon = polygon - np.array([x1, y1], dtype=np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)

    # Keep a conventional rectangular HTR image while making the curved
    # segmentation effective: pixels outside the segment become background.
    segmented = np.full_like(crop, 255)
    if crop.ndim == 2:
        segmented[mask > 0] = crop[mask > 0]
    else:
        segmented[mask > 0, :] = crop[mask > 0, :]
    return segmented


def crop_rows(
    original: np.ndarray,
    rows: List[Row],
    output_dir: Path,
) -> List[np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    crops = []

    for row in rows:
        crop = _segmented_crop(original, row)
        crops.append(crop)

        if row.column_index is None:
            out_path = output_dir / f"row_{row.index:03d}.png"
        else:
            column_dir = output_dir / f"column_{row.column_index:02d}"
            column_dir.mkdir(parents=True, exist_ok=True)
            out_path = column_dir / f"row_{row.column_row_index:03d}.png"
        cv2.imwrite(str(out_path), crop)

    return crops
