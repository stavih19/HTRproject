# Historical HTR – Row Extraction Playground

A small, modular research setup for detecting logical text rows in historical
handwritten documents, with optional semantic-column splitting before HTR.

The pipeline does not remove vertical table rules or invoke Kraken. Semantic
column splitting is optional and is disabled by default.

## Pipeline

1. Load original image
2. Normalize large pages to a 2048-pixel-high analysis copy and convert to grayscale
3. Create a permissive discovery mask and a locally relative front-ink map
4. Build global and local-strip horizontal ink projections
5. Detect candidates per strip and cluster them by Y position
6. Validate candidates using raw height and text-like connected components
7. Refine / filter / merge row bands
8. Optionally detect semantic columns and split every logical row per column
9. Draw final row bounding boxes
10. Trace curved baselines and row polygons independently inside each column
11. Crop each row from the **original image**
12. Save `result.json` and `alto.xml`

Every stage automatically saves a visual diagnostic figure into the output
folder, so changes to the algorithm can be inspected rather than guessed.

## Project structure

```text
historical_htr/
├── data/
│   ├── raw/
│   └── annotations/
├── outputs/
├── src/
│   ├── config.py
│   ├── models.py
│   ├── image_utils.py
│   ├── visualization.py
│   ├── pipeline.py
│   └── stages/
│       ├── binarization.py
│       ├── row_detection.py
│       ├── row_refinement.py
│       └── row_crop.py
├── run.py
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run one image

```bash
python run.py data/raw/SL1934_00130.jpg
```

When each image finishes processing, its final image with the detected row boxes
opens in the system's default image viewer. This also happens image-by-image
when processing a folder. For headless or unattended runs, disable this with:

```bash
python run.py data/raw/SL1934_00130.jpg --no-show
```

## Run all images in a folder

```bash
python run.py data/raw
```

## Optional column splitting

Column splitting is off by default. Enable it from the command line with:

```bash
python run.py data/raw --columns
```

Explicitly disable it (and keep one full-width logical row) with:

```bash
python run.py data/raw --no-columns
```

The same flags work for a single image. The equivalent configuration setting
is `PipelineConfig().columns.enabled`. Column order defaults to right-to-left.

## Output

For each image:

```text
outputs/<image-stem>/
├── 00_original.png
├── 01_grayscale.png
├── 02_binarization.png
├── 03_projection.png
├── 04_row_candidates.png
├── 04_candidate_validation.png
├── 04_columns.png              # only when --columns is enabled
├── 05_row_segments.png
├── 05_rows_final.png
├── 06_row_crops.png
├── rows/
│   ├── row_000.png             # full-width mode
│   ├── row_001.png
│   ├── column_00/              # column mode
│   │   └── row_000.png
│   └── column_01/
├── result.json
├── alto.xml
└── report.html
```

When a folder is processed, `outputs/index.html` is also generated as a single entry point to all visual reports.

Stage 3 preserves the original full-page projection and adds normalized density
projections over six overlapping vertical analysis strips. These strips are not
treated as semantic columns. Stage 4 consumes independently detected and
Y-clustered intervals, rather than OR-ing the strip masks and accidentally
joining nearby rows from different parts of the page.

Stage 2 also estimates front-page ink independently in overlapping local
windows. It separates the weaker and stronger local-contrast populations and
uses hysteresis to retain weak stroke pixels connected to strong seeds. Row
discovery remains permissive for recall; curved baseline and polygon tracing
use the relative front-ink confidence so faint bleed-through has less influence.

The diagnostic plots are part of the pipeline by design. Each future algorithmic
stage should also add its own visual output.

With `--columns`, `04_columns.png` shows the detected boundaries. Rows are
numbered independently per column in reading order, the crop files are grouped
under `rows/column_XX/`, and ALTO receives a separate `TextBlock` per column.