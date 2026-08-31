from pathlib import Path
import html


STAGES = [
    ("00_original.png", "Stage 0 — Original"),
    ("01_grayscale.png", "Stage 1 — Grayscale"),
    ("02_binarization.png", "Stage 2 — Binarization diagnostic"),
    ("03_projection.png", "Stage 3 — Horizontal projection"),
    ("04_row_candidates.png", "Stage 4 — Raw row candidates"),
    ("05_rows_final.png", "Stage 5 — Final row bounding boxes"),
    ("06_row_crops.png", "Stage 6 — Row crops from original"),
]


def write_image_report(output_dir: Path, image_name: str, row_count: int):
    cards = []
    for filename, title in STAGES:
        if (output_dir / filename).exists():
            cards.append(
                f'''
                <section class="card">
                  <h2>{html.escape(title)}</h2>
                  <img src="{html.escape(filename)}" alt="{html.escape(title)}">
                </section>
                '''
            )

    body = "\n".join(cards)
    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Row detection report — {html.escape(image_name)}</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  max-width: 1500px;
  margin: 0 auto;
  padding: 24px;
  line-height: 1.45;
}}
header {{
  margin-bottom: 28px;
}}
.card {{
  margin: 30px 0;
  padding: 18px;
  border: 1px solid #bbb;
  border-radius: 10px;
}}
img {{
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}}
code {{
  background: #eee;
  padding: 2px 5px;
  border-radius: 4px;
}}
</style>
</head>
<body>
<header>
  <h1>Row detection report</h1>
  <p><strong>Image:</strong> {html.escape(image_name)}</p>
  <p><strong>Detected row bands:</strong> {row_count}</p>
  <p>
    This report is intentionally stage-by-stage. Each algorithmic stage leaves
    a visual diagnostic artifact so later changes can be compared.
  </p>
</header>
{body}
</body>
</html>
'''
    (output_dir / "report.html").write_text(page, encoding="utf-8")


def write_dataset_index(output_root: Path):
    reports = sorted(output_root.glob("*/report.html"))
    items = []
    for report in reports:
        rel = report.relative_to(output_root)
        items.append(
            f'<li><a href="{html.escape(str(rel))}">'
            f'{html.escape(report.parent.name)}</a></li>'
        )

    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Historical HTR row detection reports</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
}}
li {{ margin: 10px 0; }}
</style>
</head>
<body>
<h1>Historical HTR row detection reports</h1>
<p>Open an image report to inspect all stages visually.</p>
<ul>
{''.join(items)}
</ul>
</body>
</html>
'''
    (output_root / "index.html").write_text(page, encoding="utf-8")
