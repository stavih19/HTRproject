from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.config import RowDetectionConfig


@dataclass
class ProjectionAnalysis:
    # Existing global signal
    raw_projection: np.ndarray
    smoothed_projection: np.ndarray
    threshold: float

    # Local-strip signal. local_score is normalized so that 1.0 is the
    # per-strip activation threshold.
    local_score: np.ndarray
    global_active_mask: np.ndarray
    local_active_mask: np.ndarray

    # Candidates are detected independently per strip and then clustered by
    # Y. An OR between the strip masks could join unrelated column entries.
    candidate_intervals: List[Tuple[int, int]]

    # Retained for diagnostics/backwards compatibility. Stage 4 uses
    # candidate_intervals directly.
    active_mask: np.ndarray


def _smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _regions_from_mask(active_mask: np.ndarray) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start = None

    for y, is_active in enumerate(active_mask):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            intervals.append((start, y - 1))
            start = None

    if start is not None:
        intervals.append((start, len(active_mask) - 1))

    return intervals


def _cluster_intervals_by_center(
    intervals: List[Tuple[int, int]],
    image_height: int,
    config: RowDetectionConfig,
) -> List[Tuple[int, int]]:
    """Cluster global/local candidates by Y center without mask union."""

    if not intervals:
        return []

    max_center_distance = max(
        2.0,
        float(config.local_cluster_distance_ratio) * image_height,
    )

    items = sorted(
        (
            ((y1 + y2) / 2.0, max(1.0, y2 - y1 + 1.0), y1, y2)
            for y1, y2 in intervals
        ),
        key=lambda item: item[0],
    )

    clusters = []
    for item in items:
        center = item[0]

        if not clusters:
            clusters.append([item])
            continue

        current_center = float(np.median([x[0] for x in clusters[-1]]))
        if abs(center - current_center) <= max_center_distance:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    result = []
    for cluster in clusters:
        centers = np.array([x[0] for x in cluster], dtype=np.float32)
        heights = np.array([x[1] for x in cluster], dtype=np.float32)

        center = float(np.median(centers))
        height = max(
            float(config.local_min_output_height),
            float(np.median(heights)),
        )
        height = min(height, float(config.local_max_output_height))

        half = height / 2.0
        y1 = max(0, int(round(center - half)))
        y2 = min(image_height - 1, int(round(center + half)))
        result.append([y1, y2])

    # Separate neighboring estimates instead of merging overlapping boxes.
    for i in range(len(result) - 1):
        y1a, y2a = result[i]
        y1b, y2b = result[i + 1]
        if y2a >= y1b:
            center_a = (y1a + y2a) / 2.0
            center_b = (y1b + y2b) / 2.0
            split = int(round((center_a + center_b) / 2.0))
            result[i][1] = max(result[i][0], split - 1)
            result[i + 1][0] = min(result[i + 1][1], split + 1)

    return [(int(a), int(b)) for a, b in result if b >= a]


def _local_strip_candidates(
    binary: np.ndarray,
    config: RowDetectionConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Detect candidates in overlapping analysis strips (not columns)."""

    h, w = binary.shape[:2]
    foreground = (binary > 0).astype(np.float32)

    num_strips = max(1, int(config.num_vertical_strips))
    strip_width = max(1, int(np.ceil(w / num_strips)))
    overlap = float(np.clip(config.strip_overlap, 0.0, 0.9))
    step = max(1, int(round(strip_width * (1.0 - overlap))))

    normalized_scores = []
    all_intervals: List[Tuple[int, int]] = []

    x1 = 0
    while x1 < w:
        x2 = min(w, x1 + strip_width)
        if x2 <= x1:
            break

        density = np.mean(foreground[:, x1:x2], axis=1)
        smoothed_density = _smooth_1d(density, config.smoothing_window)
        percentile_value = float(
            np.percentile(smoothed_density, config.local_percentile_threshold)
        )
        local_threshold = max(
            percentile_value,
            float(config.local_min_density),
            1e-6,
        )

        score = smoothed_density / local_threshold
        normalized_scores.append(score)
        strip_intervals = _regions_from_mask(score > 1.0)

        for y1, y2 in strip_intervals:
            band_height = y2 - y1 + 1
            if band_height < config.local_min_band_height:
                continue
            if band_height > config.local_max_band_height:
                continue
            all_intervals.append((y1, y2))

        if x2 == w:
            break
        x1 += step

    if normalized_scores:
        local_score = np.max(np.stack(normalized_scores, axis=0), axis=0)
    else:
        local_score = np.zeros(h, dtype=np.float32)

    local_active = local_score > 1.0
    return local_score.astype(np.float32), local_active, all_intervals


def horizontal_projection(
    binary: np.ndarray,
    config: RowDetectionConfig,
) -> ProjectionAnalysis:
    """Compute global + local horizontal ink evidence for row detection."""

    h, w = binary.shape[:2]

    raw = np.sum(binary > 0, axis=1).astype(np.float32)
    smoothed = _smooth_1d(raw, config.smoothing_window)
    percentile_value = float(
        np.percentile(smoothed, config.percentile_threshold)
    )
    relative_value = float(config.relative_width_threshold * w)
    threshold = max(percentile_value, relative_value)
    global_active = smoothed > threshold
    global_intervals = _regions_from_mask(global_active)

    local_score, local_active, local_intervals = _local_strip_candidates(
        binary, config
    )
    combined_intervals = _cluster_intervals_by_center(
        global_intervals + local_intervals,
        image_height=h,
        config=config,
    )

    combined_active = np.zeros(h, dtype=bool)
    for y1, y2 in combined_intervals:
        combined_active[y1:y2 + 1] = True

    return ProjectionAnalysis(
        raw_projection=raw,
        smoothed_projection=smoothed,
        threshold=threshold,
        local_score=local_score,
        global_active_mask=global_active,
        local_active_mask=local_active,
        candidate_intervals=combined_intervals,
        active_mask=combined_active,
    )


def active_regions(active_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boolean Y mask into inclusive candidate bands."""

    return _regions_from_mask(active_mask)
