import unittest

import numpy as np

from src.config import RowDetectionConfig
from src.models import BoundingBox, Row
from src.stages.curved_segmentation import (
    _row_search_bounds,
    trace_curved_segments,
)


class CurvedSegmentationTests(unittest.TestCase):
    def test_stroke_following_bridges_short_scan_break(self):
        mask = np.zeros((50, 200), dtype=np.uint8)
        mask[12:19, 30:170] = 255
        # Continuation below the original row box, separated by two blank
        # pixels as can happen in a faded scan.
        mask[21:24, 92:100] = 255
        mask[32:38, 30:170] = 255
        rows = [
            Row(index=0, bbox=BoundingBox(0, 10, 200, 20)),
            Row(index=1, bbox=BoundingBox(0, 30, 200, 40)),
        ]

        traced = trace_curved_segments(
            mask,
            rows,
            RowDetectionConfig(
                segment_num_samples=20,
                segment_stroke_gap_threshold=3,
            ),
        )

        upper_polygon = traced[0].polygon
        self.assertGreaterEqual(max(point[1] for point in upper_polygon), 23)

    def test_midpoint_split_is_used_only_for_overlapping_rows(self):
        separate = [
            Row(index=0, bbox=BoundingBox(0, 2, 200, 12)),
            Row(index=1, bbox=BoundingBox(0, 20, 200, 32)),
        ]
        self.assertEqual((2, 11), _row_search_bounds(separate, 0, 40))
        self.assertEqual((20, 31), _row_search_bounds(separate, 1, 40))

        overlapping = [
            Row(index=0, bbox=BoundingBox(0, 2, 200, 22)),
            Row(index=1, bbox=BoundingBox(0, 16, 200, 34)),
        ]
        upper_bounds = _row_search_bounds(overlapping, 0, 40)
        lower_bounds = _row_search_bounds(overlapping, 1, 40)
        self.assertLess(upper_bounds[1], lower_bounds[0])

    def test_right_endpoint_keeps_small_rtl_row_number(self):
        mask = np.zeros((30, 200), dtype=np.uint8)

        # A large central entry dominates more than 99% of the row ink.
        mask[8:20, 30:150] = 255
        # A small but valid row-number component at the far right.
        mask[10:18, 185:189] = 255

        row = Row(index=0, bbox=BoundingBox(0, 0, 200, 30))
        [traced] = trace_curved_segments(
            mask,
            [row],
            RowDetectionConfig(segment_num_samples=12),
        )

        self.assertEqual(188, traced.baseline[-1][0])
        self.assertEqual(188, max(point[0] for point in traced.polygon))

    def test_right_endpoint_keeps_digit_touching_search_bounds(self):
        mask = np.zeros((30, 200), dtype=np.uint8)
        mask[8:20, 30:150] = 255
        # A table rule should be removed even when it spans the whole band.
        mask[:, 165:167] = 255
        # A wider digit can also span the whole, tightly cropped row band.
        mask[:, 185:193] = 255

        row = Row(index=0, bbox=BoundingBox(0, 0, 200, 30))
        [traced] = trace_curved_segments(
            mask,
            [row],
            RowDetectionConfig(segment_num_samples=12),
        )

        self.assertEqual(192, traced.baseline[-1][0])
        self.assertEqual(192, max(point[0] for point in traced.polygon))

    def test_polygon_keeps_full_ascenders_and_descenders(self):
        mask = np.zeros((36, 200), dtype=np.uint8)
        mask[10:24, 30:170] = 255
        # Sparse extrema represent a tall letter and a deep descender.  They
        # would be outside the old 5th/95th-percentile envelope.
        mask[2:12, 92:98] = 255
        mask[22:34, 112:118] = 255

        row = Row(index=0, bbox=BoundingBox(0, 0, 200, 36))
        [traced] = trace_curved_segments(
            mask,
            [row],
            RowDetectionConfig(segment_num_samples=20),
        )

        half = len(traced.polygon) // 2
        top = traced.polygon[:half]
        bottom = traced.polygon[half:]
        self.assertLessEqual(min(point[1] for point in top), 2)
        self.assertGreaterEqual(max(point[1] for point in bottom), 33)
        top_by_x = {x: y for x, y in top}
        bottom_by_x = {x: y for x, y in bottom}
        self.assertLessEqual(top_by_x[95], 2)
        self.assertGreaterEqual(bottom_by_x[115], 33)


if __name__ == "__main__":
    unittest.main()
