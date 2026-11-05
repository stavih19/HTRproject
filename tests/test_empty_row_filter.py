import unittest

import numpy as np

from src.config import RowDetectionConfig
from src.models import BoundingBox, Row
from src.stages.empty_row_filter import (
    RowContentEvidence,
    filter_empty_rows,
    polygon_text_evidence,
    select_text_rows,
)


class EmptyRowFilterTests(unittest.TestCase):
    def setUp(self):
        self.config = RowDetectionConfig()

    def test_removes_row_without_character_like_components(self):
        ink = np.zeros((30, 40), dtype=np.uint8)
        ink[5:8, 5:8] = 255
        ink[5:8, 11:14] = 255
        ink[20:23, 30:35] = 255  # Outside both row polygons.
        populated = Row(
            index=0,
            bbox=BoundingBox(2, 2, 20, 12),
            polygon=[[2, 2], [19, 2], [19, 11], [2, 11]],
        )
        empty = Row(
            index=1,
            bbox=BoundingBox(2, 14, 20, 25),
            polygon=[[2, 14], [19, 14], [19, 24], [2, 24]],
        )

        self.assertEqual((2, 18, 3), polygon_text_evidence(ink, populated, self.config))
        self.assertEqual((0, 0, 0), polygon_text_evidence(ink, empty, self.config))

    def test_uses_bounding_box_when_polygon_is_missing(self):
        ink = np.zeros((10, 10), dtype=np.uint8)
        ink[3:5, 4:7] = 255
        row = Row(index=0, bbox=BoundingBox(3, 2, 8, 6))

        self.assertEqual((1, 6, 3), polygon_text_evidence(ink, row, self.config))

    def test_rejects_long_horizontal_rule_even_with_many_ink_pixels(self):
        ink = np.zeros((20, 100), dtype=np.uint8)
        ink[9:11, 5:95] = 255
        row = Row(
            index=0,
            bbox=BoundingBox(0, 4, 100, 16),
            polygon=[[0, 4], [99, 4], [99, 15], [0, 15]],
        )

        self.assertEqual([], filter_empty_rows(ink, ink, [row], self.config))

    def test_removes_sustained_weak_run_but_keeps_isolated_weak_row(self):
        rows = [Row(index=i, bbox=BoundingBox(0, i, 10, i + 1)) for i in range(8)]
        good = RowContentEvidence(100, 30, 30.0)
        weak = RowContentEvidence(10, 5, 5.0)
        evidence = [good, weak, good, weak, weak, weak, good, good]

        kept = select_text_rows(rows, evidence, self.config)

        self.assertEqual([0, 1, 2, 6, 7], [row.index for row in kept])

    def test_bridges_one_row_gap_inside_leading_border_sequence(self):
        rows = [Row(index=i, bbox=BoundingBox(0, i, 10, i + 1)) for i in range(7)]
        good = RowContentEvidence(100, 30, 30.0)
        weak = RowContentEvidence(10, 5, 5.0)
        evidence = [weak, good, weak, weak, weak, good, good]

        kept = select_text_rows(rows, evidence, self.config)

        self.assertEqual([5, 6], [row.index for row in kept])

    def test_preserves_plausible_transition_at_start_of_trailing_weak_run(self):
        rows = [Row(index=i, bbox=BoundingBox(0, i, 10, i + 1)) for i in range(5)]
        good = RowContentEvidence(100, 30, 30.0)
        transition = RowContentEvidence(20, 15, 8.0)
        weak = RowContentEvidence(10, 5, 5.0)
        evidence = [good, good, transition, weak, weak]

        kept = select_text_rows(rows, evidence, self.config)

        self.assertEqual([0, 1, 2], [row.index for row in kept])


if __name__ == "__main__":
    unittest.main()
