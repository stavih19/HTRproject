import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models import BoundingBox, Row
from src.stages.row_crop import crop_rows


class RowCropTests(unittest.TestCase):
    def test_crop_uses_segment_polygon_and_whitens_outside(self):
        image = np.zeros((12, 14, 3), dtype=np.uint8)
        row = Row(
            index=0,
            bbox=BoundingBox(0, 0, 14, 12),
            polygon=[[3, 2], [10, 4], [10, 8], [3, 10]],
        )

        with tempfile.TemporaryDirectory() as directory:
            [crop] = crop_rows(image, [row], Path(directory))

        self.assertEqual((9, 8, 3), crop.shape)
        self.assertTrue(np.all(crop[0, -1] == 255))
        self.assertTrue(np.all(crop[4, 4] == 0))


if __name__ == "__main__":
    unittest.main()
