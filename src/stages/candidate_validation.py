from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.config import RowDetectionConfig


@dataclass
class CandidateDecision:
    interval: Tuple[int, int]
    keep: bool
    reason: str
    raw_height: int
    text_components: int
    text_ink: int


def _text_components(
    band: np.ndarray,
    config: RowDetectionConfig,
) -> Tuple[int, int, int]:
    """Return count, total ink, and widest text-like connected component."""

    mask = (band > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    band_height = max(1, band.shape[0])

    component_count = 0
    text_ink = 0
    widest = 0

    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < config.validation_min_component_area:
            continue

        aspect_ratio = width / max(1.0, float(height))
        spans_band_height = height / band_height >= config.validation_full_height_ratio

        # A component spanning almost the full candidate height is commonly a
        # vertical table rule. A very wide and flat component is commonly a
        # horizontal rule. Neither counts as evidence of text.
        if spans_band_height:
            continue
        if aspect_ratio >= config.validation_max_line_aspect_ratio:
            continue

        component_count += 1
        text_ink += int(area)
        widest = max(widest, int(width))

    return component_count, text_ink, widest


def validate_candidates(
    binary: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    config: RowDetectionConfig,
) -> Tuple[List[Tuple[int, int]], List[CandidateDecision]]:
    """Filter tiny and non-text row candidates using their unpadded bands."""

    image_height = binary.shape[0]
    kept: List[Tuple[int, int]] = []
    decisions: List[CandidateDecision] = []

    for y1, y2 in intervals:
        y1 = max(0, int(y1))
        y2 = min(image_height - 1, int(y2))
        raw_height = y2 - y1 + 1

        if raw_height < config.validation_min_raw_height:
            decisions.append(
                CandidateDecision(
                    (y1, y2), False, f"height={raw_height}", raw_height, 0, 0
                )
            )
            continue

        components, text_ink, widest = _text_components(
            binary[y1:y2 + 1, :], config
        )

        enough_components = components >= config.validation_min_text_components
        plausible_single = (
            components >= 1
            and widest >= config.validation_single_component_min_width
        )
        enough_ink = text_ink >= config.validation_min_text_ink
        keep = enough_ink and (enough_components or plausible_single)

        if keep:
            reason = f"kept: components={components}, ink={text_ink}"
            kept.append((y1, y2))
        elif text_ink < config.validation_min_text_ink:
            reason = f"ink={text_ink}"
        else:
            reason = f"components={components}, widest={widest}"

        decisions.append(
            CandidateDecision(
                (y1, y2), keep, reason, raw_height, components, text_ink
            )
        )

    return kept, decisions
