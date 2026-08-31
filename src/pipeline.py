from pathlib import Path
import json
import shutil

import cv2

from src.config import PipelineConfig
from src.image_utils import read_image, to_gray
from src.models import PageResult
from src.stages.binarization import create_binary_mask, create_front_ink_mask
from src.stages.row_detection import horizontal_projection
from src.stages.row_refinement import refine_rows
from src.stages.row_crop import crop_rows
from src.stages.candidate_validation import validate_candidates
from src.stages.curved_segmentation import trace_curved_segments
from src.stages.row_annotations import apply_row_exclusions
from src.report import write_image_report
from src.visualization import (
    save_original,
    save_grayscale,
    save_binarization,
    save_projection,
    save_candidates,
    save_candidate_validation,
    save_final_rows,
    save_curved_segments,
    save_row_crops,
)


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

    save_original(original, out_dir / "00_original.png")

    # Stage 1 — grayscale
    gray = to_gray(original)
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
        original,
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
        original,
        validation_decisions,
        out_dir / "04_candidate_validation.png",
    )

    # Stage 5 — final rows
    rows = refine_rows(
        candidates,
        image_width=w,
        image_height=h,
        config=config.row_detection,
    )

    # Optional page-level review annotations remove confirmed verso/bleed-
    # through candidates. Y ranges are stable even when row indices shift.
    annotation_path = image_path.parent.parent / "annotations" / f"{image_path.stem}.json"
    if not annotation_path.exists():
        annotation_path = Path("data/annotations") / f"{image_path.stem}.json"
    rows, excluded_rows = apply_row_exclusions(rows, annotation_path)
    if excluded_rows:
        print(
            f"{image_path.name}: excluded {len(excluded_rows)} "
            "reviewed bleed-through row(s)"
        )
    save_final_rows(
        original,
        rows,
        out_dir / "05_rows_final.png",
    )

    # Stage 5.5 — trace a non-horizontal path and polygon inside each row.
    rows = trace_curved_segments(
        front_ink,
        rows,
        config.row_detection,
        ink_confidence=front_confidence,
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

    write_image_report(out_dir, image_path.name, len(rows))

    return result
