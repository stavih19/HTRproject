from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("models.json")


def load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def download(model_id: str, entry: dict) -> bool:
    url = entry.get("download_url")
    if not url:
        print(f"{model_id}: no public model download is registered; {entry['notes']}")
        return False

    destination = PROJECT_ROOT / entry["model_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"{model_id}: already present at {destination}")
        return True

    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"{model_id}: downloading {entry['name']} -> {destination}")
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "HTRproject-model-downloader/1.0"}
        )
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
        return True
    except Exception as exc:
        partial.unlink(missing_ok=True)
        print(f"{model_id}: download failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(description="Download configured Kraken models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=registry.keys())
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    selected = registry if args.all else {args.model: registry[args.model]}
    results = []
    skipped = 0
    for model_id, entry in selected.items():
        if not entry.get("download_url"):
            print(f"{model_id}: skipped — {entry['notes']}")
            skipped += 1
            continue
        results.append(download(model_id, entry))
    print(
        f"Model download completed: {sum(results)}/{len(results)} downloaded or "
        f"already present; {skipped} skipped"
    )
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
