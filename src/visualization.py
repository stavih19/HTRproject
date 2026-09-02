from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.image_utils import bgr_to_rgb
from src.models import Row
from src.stages.row_detection import ProjectionAnalysis
from src.stages.candidate_validation import CandidateDecision
from src.stages.column_segmentation import ColumnAnalysis


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_original(image: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(bgr_to_rgb(image))
    ax.set_title("Stage 0 — Original image")
    ax.axis("off")
    _save(fig, path)


def save_grayscale(gray: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(gray, cmap="gray")
    ax.set_title("Stage 1 — Grayscale")
    ax.axis("off")
    _save(fig, path)


def save_binarization(
    original: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    front_ink: np.ndarray,
    front_confidence: np.ndarray,
    path: Path,
):
    fig, axes = plt.subplots(1, 5, figsize=(25, 7))

    axes[0].imshow(bgr_to_rgb(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title("Analysis-only binary mask")
    axes[2].axis("off")

    axes[3].imshow(front_confidence, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Relative front-ink confidence")
    axes[3].axis("off")

    axes[4].imshow(front_ink, cmap="gray")
    axes[4].set_title("Front-ink mask (local + hysteresis)")
    axes[4].axis("off")

    fig.suptitle(
        "Stage 2 — Permissive row discovery + relative local front-ink separation"
    )
    _save(fig, path)


def save_projection(
    binary: np.ndarray,
    analysis: ProjectionAnalysis,
    path: Path,
):
    h = binary.shape[0]
    y = np.arange(h)

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    axes[0].imshow(binary, cmap="gray")
    axes[0].set_title("Binary analysis mask")
    axes[0].axis("off")

    axes[1].plot(analysis.raw_projection, y, alpha=0.35, label="global raw")
    axes[1].plot(
        analysis.smoothed_projection,
        y,
        label="global smoothed",
    )
    axes[1].axvline(
        analysis.threshold,
        linestyle="--",
        label=f"global threshold = {analysis.threshold:.1f}",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Foreground pixels")
    axes[1].set_ylabel("Y coordinate")
    axes[1].set_title("Global horizontal projection")
    axes[1].legend()

    axes[2].plot(
        analysis.local_score,
        y,
        label="best local-strip score",
    )
    axes[2].axvline(
        1.0,
        linestyle="--",
        label="local activation = 1.0",
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Local evidence / local threshold")
    axes[2].set_ylabel("Y coordinate")
    axes[2].set_title("Local projection — short/sparse row evidence")
    axes[2].legend()

    fig.suptitle(
        "Stage 3 — Combined row evidence: global projection + local vertical strips"
    )
    _save(fig, path)


def _draw_band(ax, x1, x2, y1, y2, label=None):
    xs = [x1, x2, x2, x1, x1]
    ys = [y1, y1, y2, y2, y1]
    ax.plot(xs, ys, linewidth=1.5)
    if label is not None:
        ax.text(x1 + 4, y1 + 14, label, fontsize=8)


def save_candidates(
    original: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    path: Path,
):
    h, w = original.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(bgr_to_rgb(original))

    for i, (y1, y2) in enumerate(intervals):
        _draw_band(ax, 0, w - 1, y1, y2, f"C{i:02d}")

    ax.set_title(
        f"Stage 4 — Raw row candidates ({len(intervals)} bands)"
    )
    ax.axis("off")
    _save(fig, path)


def save_candidate_validation(
    original: np.ndarray,
    decisions: Sequence[CandidateDecision],
    path: Path,
):
    h, w = original.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(bgr_to_rgb(original))

    kept = sum(decision.keep for decision in decisions)
    for index, decision in enumerate(decisions):
        y1, y2 = decision.interval
        color = "limegreen" if decision.keep else "red"
        xs = [0, w - 1, w - 1, 0, 0]
        ys = [y1, y1, y2, y2, y1]
        ax.plot(xs, ys, linewidth=1.5, color=color)
        ax.text(
            4,
            min(h - 1, y1 + 12),
            f"C{index:02d}: {decision.reason}",
            fontsize=7,
            color=color,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )

    ax.set_title(
        f"Stage 4.5 — Candidate validation ({kept} kept, "
        f"{len(decisions) - kept} rejected)"
    )
    ax.axis("off")
    _save(fig, path)


def save_columns(
    original: np.ndarray,
    analysis: ColumnAnalysis,
    path: Path,
):
    h, _ = original.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(bgr_to_rgb(original))

    for gutter_x1, gutter_x2 in analysis.gutters:
        ax.axvspan(gutter_x1, gutter_x2, color="red", alpha=0.25)
    for index, (x1, x2) in enumerate(analysis.bounds):
        xs = [x1, x2 - 1, x2 - 1, x1, x1]
        ys = [0, 0, h - 1, h - 1, 0]
        ax.plot(xs, ys, linewidth=2)
        ax.text(
            x1 + 5,
            25,
            f"Column {index}",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    suffix = " — fallback to full width" if analysis.used_fallback else ""
    ax.set_title(f"Stage 4.7 — Semantic columns ({len(analysis.bounds)}){suffix}")
    ax.axis("off")
    _save(fig, path)


def save_final_rows(
    original: np.ndarray,
    rows: Sequence[Row],
    path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(bgr_to_rgb(original))

    for row in rows:
        b = row.bbox
        _draw_band(
            ax,
            b.x1,
            b.x2 - 1,
            b.y1,
            b.y2 - 1,
            (
                f"C{row.column_index}-R{row.column_row_index:02d}"
                if row.column_index is not None
                else f"R{row.index:02d}"
            ),
        )

    ax.set_title(
        f"Stage 5 — Final row bounding boxes ({len(rows)} rows)"
    )
    ax.axis("off")
    _save(fig, path)


def save_curved_segments(
    original: np.ndarray,
    rows: Sequence[Row],
    path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(bgr_to_rgb(original))
    colors = plt.get_cmap("tab20")

    for index, row in enumerate(rows):
        if not row.baseline or not row.polygon:
            continue

        color = colors(index % 20)
        polygon = np.asarray(row.polygon)
        baseline = np.asarray(row.baseline)
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            color=color,
            alpha=0.20,
        )
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=color,
            linewidth=1.0,
        )
        ax.plot(
            baseline[:, 0],
            baseline[:, 1],
            color=color,
            linewidth=1.8,
        )
        ax.text(
            baseline[0, 0],
            baseline[0, 1],
            (
                f"C{row.column_index}-R{row.column_row_index:02d}"
                if row.column_index is not None
                else f"R{row.index:02d}"
            ),
            fontsize=7,
            color=color,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )

    ax.set_title(f"Stage 5.5 — Curved row segments ({len(rows)} rows)")
    ax.axis("off")
    _save(fig, path)


def save_row_crops(
    crops: Sequence[np.ndarray],
    path: Path,
    max_rows: int = 20,
):
    if not crops:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No row crops detected", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return

    count = min(len(crops), max_rows)
    fig, axes = plt.subplots(count, 1, figsize=(16, max(3, 1.6 * count)))

    if count == 1:
        axes = [axes]

    for i in range(count):
        axes[i].imshow(bgr_to_rgb(crops[i]))
        axes[i].set_title(f"row_{i:03d}")
        axes[i].axis("off")

    if len(crops) > max_rows:
        fig.suptitle(
            f"Stage 6 — First {max_rows} of {len(crops)} crops from original image"
        )
    else:
        fig.suptitle(
            f"Stage 6 — {len(crops)} crops from original image"
        )

    _save(fig, path)
