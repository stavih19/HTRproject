from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BinarizationConfig:
    adaptive_block_size: int = 51
    adaptive_c: int = 15

    # Relative front/back ink separation in overlapping local windows.
    front_ink_tile_size: int = 256
    front_ink_tile_overlap: float = 0.50
    front_ink_min_samples: int = 40
    front_ink_min_cluster_separation: float = 3.0
    front_ink_strong_confidence: float = 0.65
    front_ink_weak_confidence: float = 0.20


@dataclass
class RowDetectionConfig:
    # Horizontal projection
    smoothing_window: int = 9

    # A row is active when its smoothed ink count is above:
    # max(percentile_threshold, relative_width_threshold * image_width)
    percentile_threshold: float = 35.0
    relative_width_threshold: float = 0.012

    # Local density projections over temporary, overlapping vertical strips.
    # These are analysis windows only; they are not semantic columns.
    num_vertical_strips: int = 6
    strip_overlap: float = 0.25
    local_percentile_threshold: float = 55.0
    local_min_density: float = 0.012
    local_min_band_height: int = 4
    local_max_band_height: int = 90

    # Candidates from different strips that have nearby Y centers represent
    # the same logical row. The distance scales with page height.
    local_cluster_distance_ratio: float = 0.012
    local_min_output_height: int = 8
    local_max_output_height: int = 90

    # Stage 4.5 — validate that a candidate contains text-like ink before
    # padding turns a tiny/noisy interval into an apparently valid row.
    validation_min_raw_height: int = 10
    validation_min_component_area: int = 6
    validation_min_text_ink: int = 20
    validation_min_text_components: int = 2
    validation_single_component_min_width: int = 8
    validation_max_line_aspect_ratio: float = 20.0
    validation_full_height_ratio: float = 0.90

    # Candidate filtering / refinement
    min_row_height: int = 8
    max_row_height: int = 180
    # Candidate centers have already been clustered by Stage 3. Merge only
    # truly touching/overlapping intervals here; a positive gap may separate
    # two densely written but distinct text rows.
    merge_gap: int = 0

    # Final crop padding
    padding_top: int = 4
    padding_bottom: int = 4

    # Curved row segmentation. The existing box is only the search region;
    # the resulting baseline and polygon follow the local ink trajectory.
    segment_num_samples: int = 36
    segment_window_width_ratio: float = 0.055
    segment_baseline_quantile: float = 0.65
    segment_top_quantile: float = 0.05
    segment_bottom_quantile: float = 0.95
    segment_smoothing_window: int = 5
    segment_max_step_ratio: float = 0.35
    segment_min_half_height: int = 4


@dataclass
class ColumnConfig:
    enabled: bool = False
    detection_mode: str = "auto"
    expected_columns: int = 2
    reading_order: str = "rtl"
    min_gutter_width_ratio: float = 0.025
    column_padding_ratio: float = 0.01
    min_segment_ink: int = 20
    min_segment_width_ratio: float = 0.03
    fixed_boundaries: Optional[List[Tuple[float, float]]] = None


@dataclass
class PipelineConfig:
    binarization: BinarizationConfig = None
    row_detection: RowDetectionConfig = None
    columns: ColumnConfig = None
    analysis_target_height: int = 2048

    def __post_init__(self):
        if self.binarization is None:
            self.binarization = BinarizationConfig()
        if self.row_detection is None:
            self.row_detection = RowDetectionConfig()
        if self.columns is None:
            self.columns = ColumnConfig()
