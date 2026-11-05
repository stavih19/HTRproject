"""Render ALTO line polygons and baselines on their source image."""

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def _name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_points(value: str) -> np.ndarray:
    values = [int(float(value)) for value in value.split()]
    if len(values) % 2:
        raise ValueError(f"Invalid ALTO points: {value!r}")
    return np.asarray(values, dtype=np.int32).reshape((-1, 2))


def find_source_image(alto_path: Path, filename: str) -> Path:
    candidates = [alto_path.parent / filename, alto_path.parent.parent / filename, Path(filename)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = [path for path in Path.cwd().rglob(Path(filename).name) if path.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        choices = "\n  ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Multiple images named {filename!r} were found. Use --image:\n  {choices}"
        )
    raise FileNotFoundError(
        f"Could not find source image {filename!r}. Pass its path with --image."
    )


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def render_alto(
    alto_path: Path, image_path: Path | None = None
) -> tuple[np.ndarray, Path, int]:
    root = ET.parse(alto_path).getroot()
    if image_path is None:
        filename = next((item for item in root.iter() if _name(item.tag) == "fileName"), None)
        if filename is None or not filename.text:
            raise ValueError("ALTO XML does not contain sourceImageInformation/fileName")
        image_path = find_source_image(alto_path, filename.text.strip())

    image_path = image_path.resolve()
    canvas = load_image(image_path)
    scale = max(canvas.shape[:2])
    thickness = max(1, round(scale / 1200))
    radius = max(2, round(scale / 500))
    line_count = 0

    for line in (item for item in root.iter() if _name(item.tag) == "TextLine"):
        line_count += 1
        polygon_node = next(
            (item for item in line.iter() if _name(item.tag) == "Polygon"), None
        )
        if polygon_node is not None and polygon_node.get("POINTS"):
            polygon = parse_points(polygon_node.get("POINTS", ""))
            if len(polygon) >= 2:
                cv2.polylines(canvas, [polygon], True, (0, 200, 0), thickness)
            for point in polygon:
                cv2.circle(canvas, tuple(point), radius, (0, 255, 0), -1)

        if line.get("BASELINE"):
            baseline = parse_points(line.get("BASELINE", ""))
            if len(baseline) >= 2:
                cv2.polylines(canvas, [baseline], False, (0, 0, 255), thickness)
            for point in baseline:
                cv2.circle(canvas, tuple(point), radius, (0, 0, 255), -1)

    return canvas, image_path, line_count


def open_image(path: Path) -> bool:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
            return True
        command = "open" if sys.platform == "darwin" else "xdg-open"
        if not shutil.which(command):
            return False
        subprocess.Popen(
            [command, str(path)], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw ALTO polygons (green) and baselines (red) on the image."
    )
    parser.add_argument("alto", type=Path, help="Path to alto.xml")
    parser.add_argument("--image", type=Path, help="Source image (auto-detected by default)")
    parser.add_argument("--output", "-o", type=Path, help="Output PNG path")
    parser.add_argument("--open", action="store_true", help="Open the rendered image")
    args = parser.parse_args()

    alto_path = args.alto.resolve()
    if not alto_path.is_file():
        parser.error(f"ALTO XML does not exist: {alto_path}")
    try:
        canvas, image_path, line_count = render_alto(alto_path, args.image)
    except (ET.ParseError, OSError, ValueError) as error:
        parser.error(str(error))

    output = (args.output or alto_path.with_name(f"{alto_path.stem}_overlay.png")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise SystemExit(f"Could not write output image: {output}")

    print(f"Source image: {image_path}")
    print(f"Text lines: {line_count}")
    print(f"Overlay: {output}")
    print("Green = polygon; red = baseline")
    if args.open and not open_image(output):
        print("Could not launch an image viewer; open the overlay manually.")


if __name__ == "__main__":
    main()
