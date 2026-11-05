from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("models.json")
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


def load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def version_tuple(value: str) -> tuple[int, ...]:
    match = __import__("re").search(r"(\d+(?:\.\d+)+)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def check_kraken_version(kraken_command: str, entry: dict) -> bool:
    minimum = entry.get("min_kraken")
    if not minimum:
        return True
    completed = subprocess.run(
        [kraken_command, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    installed_text = (completed.stdout or completed.stderr).strip()
    installed = version_tuple(installed_text)
    required = version_tuple(minimum)
    if completed.returncode != 0 or not installed or installed < required:
        print(
            f"This model requires Kraken >= {minimum}; found "
            f"{installed_text or 'an unknown version'}. Upgrade with:\n"
            f"  {Path(kraken_command).with_name('pip')} install --upgrade "
            f"'kraken>={minimum},<8'",
            file=sys.stderr,
        )
        return False
    return True


def prepare_alto(source_alto: Path, image: Path, destination: Path) -> None:
    """Write a Kraken input ALTO with an unambiguous source-image path."""

    tree = ET.parse(source_alto)
    filename = tree.find(f".//{{{ALTO_NS}}}fileName")
    if filename is None:
        raise ValueError(f"ALTO has no sourceImageInformation/fileName: {source_alto}")
    filename.text = str(image.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _number(value: str | None) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def extract_rows(alto_path: Path) -> list[dict]:
    """Extract normalized Unicode text and OCR metadata from ALTO lines."""

    tree = ET.parse(alto_path)
    rows = []
    for index, line in enumerate(
        tree.findall(f".//{{{ALTO_NS}}}TextLine"), start=1
    ):
        words = []
        for element in line.findall(f".//{{{ALTO_NS}}}String"):
            content = unicodedata.normalize("NFC", element.get("CONTENT", ""))
            content = " ".join(content.split())
            if not content:
                continue
            word = {
                "text": content,
                "confidence": _number(element.get("WC")),
            }
            bbox = {
                "x": _number(element.get("HPOS")),
                "y": _number(element.get("VPOS")),
                "width": _number(element.get("WIDTH")),
                "height": _number(element.get("HEIGHT")),
            }
            if any(value is not None for value in bbox.values()):
                word["bbox"] = bbox
            words.append(word)

        text = " ".join(word["text"] for word in words)
        weighted_confidences = [
            (word["confidence"], len(word["text"].replace(" ", "")))
            for word in words
            if word["confidence"] is not None
        ]
        confidence = None
        total_weight = sum(weight for _, weight in weighted_confidences)
        if total_weight:
            confidence = round(
                sum(value * weight for value, weight in weighted_confidences)
                / total_weight,
                4,
            )
        rows.append(
            {
                "index": index,
                "line_id": line.get("ID") or f"line_{index:04d}",
                "text": text,
                "confidence": confidence,
                "words": words,
            }
        )
    return rows


def write_recognized_rows(
    alto_path: Path, output_directory: Path, metadata: dict
) -> None:
    rows = extract_rows(alto_path)
    plain_text = "\n".join(row["text"] for row in rows) + "\n"

    # Keep the original name for backwards compatibility and provide an
    # explicit filename for consumers of the recognition stage.
    (output_directory / "transcription.txt").write_text(plain_text, encoding="utf-8")
    (output_directory / "recognized_rows.txt").write_text(
        plain_text, encoding="utf-8"
    )
    payload = {
        "format_version": 1,
        **metadata,
        "row_count": len(rows),
        "rows": rows,
    }
    (output_directory / "recognized_rows.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_row_crop_results(
    text_paths: list[Path],
    crop_paths: list[Path],
    line_ids: list[str],
    output_directory: Path,
    metadata: dict,
) -> None:
    """Combine Kraken whole-line results into the public TXT/JSON outputs."""

    rows = []
    for index, (text_path, crop_path) in enumerate(
        zip(text_paths, crop_paths), start=1
    ):
        raw_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        normalized = unicodedata.normalize("NFC", raw_text)
        normalized = " ".join(normalized.split())
        rows.append(
            {
                "index": index,
                "line_id": (
                    line_ids[index - 1]
                    if index <= len(line_ids)
                    else f"line_{index:04d}"
                ),
                "text": normalized,
                "confidence": None,
                "source_crop": str(crop_path.resolve()),
                "words": [],
            }
        )

    plain_text = "\n".join(row["text"] for row in rows) + "\n"
    crop_and_text = "\n".join(
        item
        for row in rows
        for item in (Path(row["source_crop"]).name, row["text"])
    ) + "\n"
    (output_directory / "transcription.txt").write_text(plain_text, encoding="utf-8")
    (output_directory / "recognized_rows.txt").write_text(
        crop_and_text, encoding="utf-8"
    )
    payload = {
        "format_version": 1,
        **metadata,
        "recognition_mode": "individual_row_crops",
        "row_count": len(rows),
        "rows": rows,
    }
    (output_directory / "recognized_rows.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_recognition_report(rows, output_directory, metadata)


def write_recognition_report(
    rows: list[dict], output_directory: Path, metadata: dict
) -> None:
    """Create an RTL visual report pairing every crop with its OCR text."""

    cards = []
    for row in rows:
        crop_path = Path(row["source_crop"])
        relative_crop = os.path.relpath(crop_path, output_directory)
        prediction = row["text"] or "— לא זוהה טקסט —"
        cards.append(
            f"""
      <section class="row-card">
        <header><strong>שורה {row['index']:02d}</strong><code>{html.escape(crop_path.name)}</code></header>
        <img src="{html.escape(relative_crop)}" alt="חיתוך {row['index']:02d}" loading="lazy">
        <div class="prediction" dir="rtl" lang="he">{html.escape(prediction)}</div>
      </section>"""
        )

    model_name = html.escape(str(metadata.get("model_name", "OCR")))
    document = f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>בדיקת זיהוי כתב – {model_name}</title>
  <style>
    body {{ margin: 0; background: #f3f4f6; color: #171717; font-family: Arial, sans-serif; }}
    main {{ width: min(1500px, 96vw); margin: 24px auto; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ color: #555; margin: 0 0 24px; }}
    .row-card {{ background: white; border: 1px solid #d8d8d8; border-radius: 10px; margin: 14px 0; padding: 14px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; margin-bottom: 10px; }}
    code {{ direction: ltr; unicode-bidi: embed; color: #555; }}
    img {{ display: block; width: 100%; height: auto; background: white; border: 1px solid #eee; }}
    .prediction {{ margin-top: 10px; padding: 12px; background: #fff8dc; border-radius: 6px; font-size: 1.45rem; line-height: 1.6; text-align: right; unicode-bidi: plaintext; }}
  </style>
</head>
<body>
  <main>
    <h1>בדיקת תוצאות זיהוי כתב</h1>
    <p class="summary">מודל: {model_name} · מספר שורות: {len(rows)}</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (output_directory / "recognition_report.html").write_text(
        document, encoding="utf-8"
    )


def run_model(
    model_id: str,
    entry: dict,
    image: Path,
    source_alto: Path,
    rows_directory: Path,
    output_root: Path,
    kraken_command: str,
) -> bool:
    model_path = PROJECT_ROOT / entry["model_path"]
    if not entry.get("usable", True):
        print(f"{model_id}: skipped — {entry['notes']}")
        return False
    if not model_path.is_file():
        print(
            f"{model_id}: skipped — missing {model_path}. "
            f"Run: python recognition/download_models.py --model {model_id}"
        )
        return False
    if not check_kraken_version(kraken_command, entry):
        return False

    model_output = output_root / image.stem / model_id
    model_output.mkdir(parents=True, exist_ok=True)
    row_text_directory = model_output / "row_text"
    row_text_directory.mkdir(parents=True, exist_ok=True)
    crop_paths = sorted(rows_directory.glob("row_*.png"))
    if not crop_paths:
        print(f"{model_id}: no row crops found in {rows_directory}", file=sys.stderr)
        return False
    text_paths = [row_text_directory / f"{crop.stem}.txt" for crop in crop_paths]

    command = [
        kraken_command,
    ]
    for crop_path, text_path in zip(crop_paths, text_paths):
        command.extend(["-i", str(crop_path), str(text_path)])
    command.extend(
        [
            "ocr",
            "--no-segmentation",
            "--base-dir",
            entry.get("base_dir", "R"),
            "-m",
            str(model_path),
        ]
    )
    print(f"{model_id}: running {entry['name']}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(f"{model_id}: Kraken exited with {completed.returncode}", file=sys.stderr)
        return False

    line_tree = ET.parse(source_alto)
    line_ids = [
        line.get("ID") or f"line_{index:04d}"
        for index, line in enumerate(
            line_tree.findall(f".//{{{ALTO_NS}}}TextLine"), start=1
        )
    ]
    metadata = {
        "model_id": model_id,
        "model_name": entry["name"],
        "model_path": str(model_path),
        "image": str(image.resolve()),
        "source_alto": str(source_alto.resolve()),
        "rows_directory": str(rows_directory.resolve()),
    }
    write_row_crop_results(
        text_paths, crop_paths, line_ids, model_output, metadata
    )
    (model_output / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{model_id}: wrote {model_output / 'recognized_rows.txt'}")
    return True


def main() -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(
        description="Run Kraken recognition on this project's ALTO segmentation"
    )
    parser.add_argument("image", type=Path, help="Original image used by run.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=registry.keys())
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--alto",
        type=Path,
        help="Segmentation ALTO; defaults to outputs/<image-stem>/alto.xml",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        help="Row crops; defaults to outputs/<image-stem>/rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "recognition" / "outputs",
        help="Recognition output root",
    )
    parser.add_argument(
        "--kraken",
        default="kraken",
        help="Kraken executable name or absolute path",
    )
    args = parser.parse_args()

    image = args.image.resolve()
    if not image.is_file():
        parser.error(f"image does not exist: {image}")
    source_alto = (
        args.alto.resolve()
        if args.alto
        else PROJECT_ROOT / "outputs" / image.stem / "alto.xml"
    )
    if not source_alto.is_file():
        parser.error(f"segmentation ALTO does not exist: {source_alto}")
    rows_directory = (
        args.rows.resolve()
        if args.rows
        else PROJECT_ROOT / "outputs" / image.stem / "rows"
    )
    if not rows_directory.is_dir():
        parser.error(f"row crops directory does not exist: {rows_directory}")
    if shutil.which(args.kraken) is None:
        parser.error(
            f"Kraken executable not found: {args.kraken}. "
            "Install Kraken in a separate virtual environment or pass --kraken."
        )

    selected = registry if args.all else {args.model: registry[args.model]}
    successes = 0
    attempted = 0
    skipped = 0
    for model_id, entry in selected.items():
        model_path = PROJECT_ROOT / entry["model_path"]
        if not entry.get("usable", True) or not model_path.is_file():
            run_model(
                model_id,
                entry,
                image,
                source_alto,
                rows_directory,
                args.output,
                args.kraken,
            )
            skipped += 1
            continue
        attempted += 1
        successes += int(
            run_model(
                model_id,
                entry,
                image,
                source_alto,
                rows_directory,
                args.output,
                args.kraken,
            )
        )

    print(
        f"Recognition completed: {successes}/{attempted} available model(s) "
        f"succeeded; {skipped} skipped"
    )
    return 0 if successes == attempted and attempted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
