from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from src.config import RowDetectionConfig
from src.models import Row
from src.stages.candidate_validation import _text_components


@dataclass(frozen=True)
class RowContentEvidence:
    neighborhood_text_ink: int
    polygon_components: int
    polygon_median_component_area: float


def _polygon_ink(ink: np.ndarray, row: Row) -> np.ndarray:
    height, width = ink.shape[:2]
    if row.polygon and len(row.polygon) >= 3:
        polygon = np.asarray(row.polygon, dtype=np.int32)
    else:
        box = row.bbox
        polygon = np.asarray(
            [
                [box.x1, box.y1],
                [box.x2 - 1, box.y1],
                [box.x2 - 1, box.y2 - 1],
                [box.x1, box.y2 - 1],
            ],
            dtype=np.int32,
        )

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    x1, y1 = np.min(polygon, axis=0)
    x2, y2 = np.max(polygon, axis=0)
    if x2 < x1 or y2 < y1:
        return np.zeros((0, 0), dtype=np.uint8)

    local_polygon = polygon - np.asarray([x1, y1], dtype=np.int32)
    mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)
    region = ink[y1:y2 + 1, x1:x2 + 1]
    return np.where(mask > 0, region, 0).astype(np.uint8)


def polygon_text_evidence(
    ink: np.ndarray, row: Row, config: RowDetectionConfig
) -> tuple[int, int, int]:
    """Return character-like component count, ink area, and widest component."""

    return _text_components(_polygon_ink(ink, row), config)


def _component_median_area(mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0.0
    return float(np.median(stats[1:, cv2.CC_STAT_AREA]))


def _remove_long_rules(
    band: np.ndarray, config: RowDetectionConfig
) -> np.ndarray:
    mask = (band > 0).astype(np.uint8) * 255
    height, width = mask.shape
    horizontal_length = max(15, int(round(width * config.final_rule_length_ratio)))
    vertical_length = max(
        5, int(round(height * config.final_vertical_rule_height_ratio))
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_length, 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_length)
    )
    rules = cv2.bitwise_or(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel),
    )
    return cv2.bitwise_and(mask, cv2.bitwise_not(rules))


def collect_row_evidence(
    binary: np.ndarray,
    front_ink: np.ndarray,
    rows: Sequence[Row],
    config: RowDetectionConfig,
) -> List[RowContentEvidence]:
    """Measure text evidence in row neighborhoods and final polygons."""

    if not rows:
        return []

    centers = np.asarray(
        [(row.bbox.y1 + row.bbox.y2) / 2.0 for row in rows], dtype=float
    )
    evidence = []
    for index, row in enumerate(rows):
        y1 = 0 if index == 0 else int(round((centers[index - 1] + centers[index]) / 2))
        y2 = (
            binary.shape[0]
            if index == len(rows) - 1
            else int(round((centers[index] + centers[index + 1]) / 2))
        )
        y1 = min(binary.shape[0], max(0, y1))
        y2 = min(binary.shape[0], max(y1 + 1, y2))
        clean_band = _remove_long_rules(binary[y1:y2, :], config)
        _, neighborhood_text_ink, _ = _text_components(clean_band, config)

        polygon_ink = _polygon_ink(front_ink, row)
        polygon_components, _, _ = _text_components(polygon_ink, config)
        evidence.append(
            RowContentEvidence(
                neighborhood_text_ink=neighborhood_text_ink,
                polygon_components=polygon_components,
                polygon_median_component_area=_component_median_area(polygon_ink),
            )
        )
    return evidence


def _runs(flags: Sequence[bool]) -> List[tuple[int, int]]:
    result = []
    start = None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            result.append((start, index))
            start = None
    return result


def select_text_rows(
    rows: Sequence[Row],
    evidence: Sequence[RowContentEvidence],
    config: RowDetectionConfig,
) -> List[Row]:
    """Remove sustained non-text sequences while retaining isolated weak rows."""

    if not rows:
        return []
    page_median = float(
        np.median([item.neighborhood_text_ink for item in evidence])
    )
    minimum_neighborhood_ink = page_median * config.final_neighborhood_ink_ratio
    weak = [
        item.neighborhood_text_ink < minimum_neighborhood_ink
        or item.polygon_components < config.final_min_component_count
        or item.polygon_median_component_area
        <= config.final_min_component_median_area
        for item in evidence
    ]

    # At the leading page edge, bridge a one-row apparent recovery only when
    # it is followed by a sustained weak run. This catches multi-part borders
    # without swallowing an isolated faint first text row.
    if weak[0]:
        first_run_end = _runs(weak)[0][1]
        gap = first_run_end
        if gap < len(weak) and not weak[gap]:
            next_start = gap + 1
            next_runs = [run for run in _runs(weak) if run[0] == next_start]
            if (
                next_runs
                and next_runs[0][1] - next_start
                >= config.final_min_edge_bridge_run
            ):
                for index in range(gap, next_runs[0][1]):
                    weak[index] = True

    remove = [False] * len(rows)
    for start, end in _runs(weak):
        length = end - start
        touches_edge = start == 0 or end == len(rows)
        if touches_edge or length >= config.final_min_nontext_run:
            for index in range(start, end):
                remove[index] = True

    # A sharply fading first row in a trailing weak run can be the tail of the
    # preceding text region. Preserve it only when it still has several text
    # components; obvious page edges remain rejected.
    trailing_runs = [run for run in _runs(weak) if run[1] == len(rows)]
    if trailing_runs:
        start, _ = trailing_runs[0]
        if start > 0:
            current = evidence[start]
            previous = evidence[start - 1]
            ratio = current.neighborhood_text_ink / max(
                1, previous.neighborhood_text_ink
            )
            if (
                current.polygon_components >= config.final_min_component_count
                and ratio < config.final_transition_score_ratio
            ):
                remove[start] = False

    return [row for index, row in enumerate(rows) if not remove[index]]


def filter_empty_rows(
    binary: np.ndarray,
    front_ink: np.ndarray,
    rows: Sequence[Row],
    config: RowDetectionConfig,
) -> List[Row]:
    """Filter non-text row sequences before final crops and ALTO output."""

    evidence = collect_row_evidence(binary, front_ink, rows, config)
    return select_text_rows(rows, evidence, config)
