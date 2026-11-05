# Kraken recognition experiments — isolated stage 2

This directory keeps recognition isolated from the project's custom row
segmentation. `run_recognition.py` passes the existing `alto.xml` to Kraken,
so the generated text remains attached to the detected baselines and polygons.

Nothing in this directory writes into or modifies `outputs/<image-stem>/`.
The first-stage ALTO is read-only input. All recognition artifacts, downloaded
weights, dependencies, and documentation live below `recognition/`.

## Environment

Install Kraken 7.1 or newer in a separate environment. For example:

```bash
python3 -m venv .venv-kraken
.venv-kraken/bin/pip install --upgrade pip
.venv-kraken/bin/pip install --upgrade -r recognition/requirements.txt
```

Party currently pins Kraken 7.0 while PP-OCRv6 requires Kraken 7.1. Install
Party in its own environment so the two model families cannot downgrade or
invalidate one another:

```bash
python3 -m venv .venv-party
.venv-party/bin/pip install --upgrade pip
.venv-party/bin/pip install -r recognition/requirements-party.txt
```

BiblIA is registered as a manual model because its cited public record contains
the dataset but no unambiguous public recognition weight file.

## Download

```bash
python recognition/download_models.py --model ppocrv6_medium
python recognition/download_models.py --all
```

`--all` skips models without a registered public download (currently BiblIA)
and downloads the remaining configured models.

## Commands for each model

The commands below assume the first-stage image has already been processed and
that `.venv-kraken` was created as shown above.

```bash
IMAGE=data/raw/SL_Ledger_4_10_1940_00910.jpg
KRAKEN=.venv-kraken/bin/kraken
```

PP-OCRv6 Medium Multilingual (recommended general model):

```bash
python recognition/download_models.py --model ppocrv6_medium
python recognition/run_recognition.py "$IMAGE" --model ppocrv6_medium --kraken "$KRAKEN"
```

Vilne Yiddish Hand 1 (interwar Ashkenazi cursive, non-commercial weights):

```bash
python recognition/download_models.py --model vilne_yiddish
python recognition/run_recognition.py "$IMAGE" --model vilne_yiddish --kraken "$KRAKEN"
```

PP-OCRv6 Small Multilingual:

```bash
python recognition/download_models.py --model ppocrv6_small
python recognition/run_recognition.py "$IMAGE" --model ppocrv6_small --kraken "$KRAKEN"
```

PP-OCRv6 Tiny Multilingual:

```bash
python recognition/download_models.py --model ppocrv6_tiny
python recognition/run_recognition.py "$IMAGE" --model ppocrv6_tiny --kraken "$KRAKEN"
```

Party Multilingual Base (requires `requirements-party.txt`):

```bash
python recognition/download_models.py --model party_multilingual
python recognition/run_recognition.py "$IMAGE" --model party_multilingual \
  --kraken .venv-party/bin/kraken
```

BiblIA (manual weights required):

```bash
# Place the obtained model at recognition/models/biblia/biblia.mlmodel,
# then set "usable": true in recognition/models.json.
python recognition/run_recognition.py "$IMAGE" --model biblia --kraken "$KRAKEN"
```

PP-OCRv6 Samaritan Hebrew (comparison only):

```bash
python recognition/download_models.py --model samaritan_hebrew
python recognition/run_recognition.py "$IMAGE" --model samaritan_hebrew --kraken "$KRAKEN"
```

Bohemica Registers (Latin-script control experiment only):

```bash
python recognition/download_models.py --model bohemica
python recognition/run_recognition.py "$IMAGE" --model bohemica --kraken "$KRAKEN"
```

Download every publicly available model and run every downloaded model:

```bash
python recognition/download_models.py --all
python recognition/run_recognition.py "$IMAGE" --all --kraken "$KRAKEN"
```

## Run

Run the segmentation pipeline first:

```bash
python run.py data/raw/SL_Ledger_4_10_1940_00910.jpg --no-show
```

Then run one recognition model:

```bash
python recognition/run_recognition.py \
  data/raw/SL_Ledger_4_10_1940_00910.jpg \
  --model ppocrv6_medium \
  --kraken .venv-kraken/bin/kraken
```

Or run every downloaded/usable model:

```bash
python recognition/run_recognition.py \
  data/raw/SL_Ledger_4_10_1940_00910.jpg \
  --all \
  --kraken .venv-kraken/bin/kraken
```

Outputs are written under:

```text
recognition/outputs/<image-stem>/<model-id>/
├── row_text/
│   ├── row_000.txt
│   └── ...
├── transcription.txt
├── recognized_rows.txt
├── recognized_rows.json
├── recognition_report.html
└── run.json
```

`recognized_rows.txt` contains repeating two-line records: the crop filename,
then its recognized Hebrew text. Empty predictions are retained so the output
stays aligned with the segmentation. `transcription.txt` contains text only,
one prediction per line. The JSON file contains the same Unicode text plus the
ALTO line ID and source crop. Hebrew characters are stored directly rather than
as escaped Unicode sequences.

Open `recognition_report.html` in a browser to compare each row crop directly
with its prediction. The report and predictions use right-to-left Hebrew
layout.

Recognition is run independently on every crop in
`outputs/<image-stem>/rows/`, using Kraken's whole-line mode. This avoids
running recognition again against the complete page and generally gives much
better Hebrew output for the provided models. The text is normalized as
Unicode Hebrew; Latin characters produced by an OCR error are retained rather
than being replaced with invented Hebrew letters.

Each model has a dedicated directory under `recognition/models/<model-id>/`.
Weights are deliberately excluded from Git because several are large and each
has its own license.
