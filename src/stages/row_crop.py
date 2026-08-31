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

        out_path = output_dir / f"row_{row.index:03d}.png"
        cv2.imwrite(str(out_path), crop)

    return crops
