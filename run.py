from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

from src.config import PipelineConfig
from src.pipeline import run_pipeline
from src.report import write_dataset_index


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def open_image(path: Path) -> bool:
    """Open an image with the operating system's default viewer."""
    path = path.resolve()

    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
            return True

        command = "open" if sys.platform == "darwin" else "xdg-open"
        if shutil.which(command) is None:
            return False

        subprocess.Popen(
            [command, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


def iter_images(path: Path):
    if path.is_file():
        yield path
        return

    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def main():
    parser = argparse.ArgumentParser(
        description="Historical HTR row-detection research pipeline"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input image or folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output root directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the final row-marked image after processing",
    )

    args = parser.parse_args()
    config = PipelineConfig()

    images = list(iter_images(args.input))
    if not images:
        raise SystemExit(f"No supported images found under: {args.input}")

    print(f"Running on {len(images)} image(s)")

    for image_path in images:
        result = run_pipeline(
            image_path=image_path,
            output_root=args.output,
            config=config,
        )
        print(
            f"{image_path.name}: "
            f"{len(result.rows)} candidate/final row bands"
        )

        # Show each result as soon as that image has finished, including when
        # the input is a folder containing several images.
        if not args.no_show:
            final_image = args.output / image_path.stem / "05_row_segments.png"
            if open_image(final_image):
                print(f"Opened row-marked image: {final_image}")
            else:
                print(
                    "Could not open the image viewer automatically. "
                    f"Open manually: {final_image}"
                )

    write_dataset_index(args.output)
    print(f"Visual report index: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
