# Historical HTR – Row Extraction Playground

A small, modular research setup for detecting logical text rows in historical
handwritten documents before column splitting and HTR.

The current pipeline deliberately starts from **row detection only**.
It does **not** remove vertical table rules, detect columns, or invoke Kraken.

## Pipeline

1. Load original image
2. Convert to grayscale
3. Create a permissive discovery mask and a locally relative front-ink map
4. Build global and local-strip horizontal ink projections
5. Detect candidates per strip and cluster them by Y position
6. Validate candidates using raw height and text-like connected components
7. Refine / filter / merge row bands
8. Draw final row bounding boxes
9. Trace curved baselines and row polygons
10. Crop each row from the **original image**
11. Save `result.json`

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
├── 05_row_segments.png
├── 05_rows_final.png
├── 06_row_crops.png
├── rows/
│   ├── row_000.png
│   ├── row_001.png
│   └── ...
├── result.json
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

## Next research steps

The intended progression is:

- improve row detection on the current four representative pages
- only if needed, add table-rule detection as an earlier signal
- split detected rows into columns/cells
- keep the original image untouched until the HTR input preparation stage
- integrate Kraken only after segmentation/layout is stable
