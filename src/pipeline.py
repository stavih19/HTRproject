from pathlib import Path
import json
import shutil

import cv2
import numpy as np

from src.config import PipelineConfig
from src.image_utils import read_image, to_gray
from src.models import PageResult
from src.stages.binarization import create_binary_mask, create_front_ink_mask
from src.stages.row_detection import horizontal_projection
from src.stages.row_refinement import refine_rows
from src.stages.row_crop import crop_rows
from src.stages.candidate_validation import validate_candidates
from src.stages.curved_segmentation import trace_curved_segments
from src.stages.column_segmentation import detect_columns, split_rows_by_columns
from src.report import write_image_report
from src.alto import write_alto
from src.visualization import (
    save_original,
    save_grayscale,
    save_binarization,
    save_projection,
    save_candidates,
    save_candidate_validation,
    save_columns,
    save_final_rows,
    save_curved_segments,
    save_row_crops,
)


def _analysis_image(original: np.ndarray, target_height: int) -> np.ndarray:
    """Downsample large pages so pixel-based settings remain consistent."""

    height, width = original.shape[:2]
    target_height = max(1, int(target_height))
    if height <= target_height:
        return original

    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(
        original,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _scale_rows_to_original(
    rows,
    analysis_width: int,
    analysis_height: int,
    original_width: int,
    original_height: int,
):
    """Map boxes, baselines and polygons back to source-image pixels."""

    scale_x = original_width / float(analysis_width)
    scale_y = original_height / float(analysis_height)

    def scale_point(point):
        x = int(round(point[0] * scale_x))
        y = int(round(point[1] * scale_y))
        return [
            min(original_width - 1, max(0, x)),
            min(original_height - 1, max(0, y)),
        ]

    for row in rows:
        box = row.bbox
        box.x1 = min(original_width - 1, max(0, int(round(box.x1 * scale_x))))
        box.y1 = min(original_height - 1, max(0, int(round(box.y1 * scale_y))))
        box.x2 = min(
            original_width,
            max(box.x1 + 1, int(round(box.x2 * scale_x))),
        )
        box.y2 = min(
            original_height,
            max(box.y1 + 1, int(round(box.y2 * scale_y))),
        )
        if row.baseline is not None:
            row.baseline = [scale_point(point) for point in row.baseline]
        if row.polygon is not None:
            row.polygon = [scale_point(point) for point in row.polygon]

    return rows


def run_pipeline(
    image_path: Path,
    output_root: Path,
    config: PipelineConfig,
) -> PageResult:
    image_path = Path(image_path)
    out_dir = output_root / image_path.stem
    rows_dir = out_dir / "rows"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0 — original
    original = read_image(str(image_path))
    h, w = original.shape[:2]
    analysis_original = _analysis_image(original, config.analysis_target_height)
    analysis_h, analysis_w = analysis_original.shape[:2]

    save_original(original, out_dir / "00_original.png")

    # Stage 1 — grayscale
    gray = to_gray(analysis_original)
    save_grayscale(gray, out_dir / "01_grayscale.png")

    # Stage 2 — binarization
    binary = create_binary_mask(gray, config.binarization)
    front_ink, front_confidence = create_front_ink_mask(
        gray,
        binary,
        config.binarization,
    )
    save_binarization(
        original,
        gray,
        binary,
        front_ink,
        front_confidence,
        out_dir / "02_binarization.png",
    )

    # Stage 3 — horizontal projection
    projection = horizontal_projection(binary, config.row_detection)
    save_projection(
        binary,
        projection,
        out_dir / "03_projection.png",
    )

    # Stage 4 — raw row candidates
    # Local-strip candidates were detected independently and clustered by Y.
    # Do not rebuild them from an OR mask, which could join unrelated rows.
    candidates = projection.candidate_intervals
    save_candidates(
        analysis_original,
        candidates,
        out_dir / "04_row_candidates.png",
    )

    # Stage 4.5 — reject tiny bands and candidates without text-like ink.
    candidates, validation_decisions = validate_candidates(
        binary,
        candidates,
        config.row_detection,
    )
    save_candidate_validation(
        analysis_original,
        validation_decisions,
        out_dir / "04_candidate_validation.png",
    )

    # Stage 5 — final rows
    rows = refine_rows(
        candidates,
        image_width=analysis_w,
        image_height=analysis_h,
        config=config.row_detection,
    )

    if config.columns.enabled:
        column_analysis = detect_columns(front_ink, config.columns)
        save_columns(
            analysis_original,
            column_analysis,
            out_dir / "04_columns.png",
        )
        row_groups = split_rows_by_columns(
            rows,
            front_ink,
            column_analysis,
            config.columns,
        )
    else:
        row_groups = [rows]

    # Stage 5.5 — trace a non-horizontal path and polygon inside each row.
    traced_groups = []
    for row_group in row_groups:
        traced_groups.append(
            trace_curved_segments(
                front_ink,
                row_group,
                config.row_detection,
                ink_confidence=front_confidence,
            )
        )
    rows = [row for group in traced_groups for row in group]
    rows = _scale_rows_to_original(
        rows,
        analysis_width=analysis_w,
        analysis_height=analysis_h,
        original_width=w,
        original_height=h,
    )
    save_final_rows(
        original,
        rows,
        out_dir / "05_rows_final.png",
    )
    save_curved_segments(
        original,
        rows,
        out_dir / "05_row_segments.png",
    )

    # Stage 6 — crop from original image
    crops = crop_rows(original, rows, rows_dir)
    save_row_crops(
        crops,
        out_dir / "06_row_crops.png",
    )

    result = PageResult(
        image_name=image_path.name,
        width=w,
        height=h,
        rows=rows,
    )

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    write_alto(result, out_dir / "alto.xml")

    write_image_report(out_dir, image_path.name, len(rows))

    return result
