from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.models import Row


def crop_rows(
    original: np.ndarray,
    rows: List[Row],
    output_dir: Path,
) -> List[np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    crops = []

    for row in rows:
        b = row.bbox
        crop = original[b.y1:b.y2, b.x1:b.x2].copy()
        crops.append(crop)

        if row.column_index is None:
            out_path = output_dir / f"row_{row.index:03d}.png"
        else:
            column_dir = output_dir / f"column_{row.column_index:02d}"
            column_dir.mkdir(parents=True, exist_ok=True)
            out_path = column_dir / f"row_{row.column_row_index:03d}.png"
        cv2.imwrite(str(out_path), crop)

    return crops
