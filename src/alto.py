from pathlib import Path
import re
import xml.etree.ElementTree as ET

from src.models import PageResult, Row


ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ALTO_SCHEMA = "http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"

ET.register_namespace("", ALTO_NS)
ET.register_namespace("xsi", XSI_NS)


def _tag(name: str) -> str:
    return f"{{{ALTO_NS}}}{name}"


def _xml_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return f"htr_{safe}"


def _points(points) -> str:
    return " ".join(f"{int(x)} {int(y)}" for x, y in points)


def _row_geometry(row: Row):
    box = row.bbox
    polygon = row.polygon or [
        [box.x1, box.y1],
        [box.x2 - 1, box.y1],
        [box.x2 - 1, box.y2 - 1],
        [box.x1, box.y2 - 1],
    ]
    baseline = row.baseline or [
        [box.x1, box.y2 - 1],
        [box.x2 - 1, box.y2 - 1],
    ]

    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return polygon, baseline, x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)


def _append_text_block(parent, rows, image_stem: str, block_suffix: str):
    geometries = [_row_geometry(row) for row in rows]
    block_x1 = min(geometry[2] for geometry in geometries)
    block_y1 = min(geometry[3] for geometry in geometries)
    block_x2 = max(geometry[2] + geometry[4] for geometry in geometries)
    block_y2 = max(geometry[3] + geometry[5] for geometry in geometries)
    block = ET.SubElement(
        parent,
        _tag("TextBlock"),
        {
            "HPOS": str(block_x1),
            "VPOS": str(block_y1),
            "WIDTH": str(max(1, block_x2 - block_x1)),
            "HEIGHT": str(max(1, block_y2 - block_y1)),
            "ID": _xml_id(f"block_{image_stem}_{block_suffix}"),
            "TAGREFS": "BT_text",
        },
    )
    block_shape = ET.SubElement(block, _tag("Shape"))
    ET.SubElement(
        block_shape,
        _tag("Polygon"),
        {
            "POINTS": _points(
                [
                    [block_x1, block_y1],
                    [block_x2 - 1, block_y1],
                    [block_x2 - 1, block_y2 - 1],
                    [block_x1, block_y2 - 1],
                ]
            )
        },
    )

    for row, geometry in zip(rows, geometries):
        polygon, baseline, x1, y1, width, height = geometry
        line = ET.SubElement(
            block,
            _tag("TextLine"),
            {
                "ID": _xml_id(f"line_{image_stem}_{row.index:03d}"),
                "TAGREFS": "LT_default",
                "BASELINE": _points(baseline),
                "HPOS": str(x1),
                "VPOS": str(y1),
                "WIDTH": str(width),
                "HEIGHT": str(height),
            },
        )
        shape = ET.SubElement(line, _tag("Shape"))
        ET.SubElement(shape, _tag("Polygon"), {"POINTS": _points(polygon)})
        ET.SubElement(
            line,
            _tag("String"),
            {
                "CONTENT": "",
                "HPOS": str(x1),
                "VPOS": str(y1),
                "WIDTH": str(width),
                "HEIGHT": str(height),
            },
        )


def write_alto(result: PageResult, path: Path) -> None:
    """Write detected row geometry as ALTO XML 4.2."""

    root = ET.Element(
        _tag("alto"),
        {
            f"{{{XSI_NS}}}schemaLocation": f"{ALTO_NS} {ALTO_SCHEMA}",
        },
    )

    description = ET.SubElement(root, _tag("Description"))
    ET.SubElement(description, _tag("MeasurementUnit")).text = "pixel"
    source = ET.SubElement(description, _tag("sourceImageInformation"))
    ET.SubElement(source, _tag("fileName")).text = result.image_name

    tags = ET.SubElement(root, _tag("Tags"))
    ET.SubElement(
        tags,
        _tag("OtherTag"),
        {
            "ID": "BT_text",
            "LABEL": "text",
            "DESCRIPTION": "block type text",
        },
    )
    ET.SubElement(
        tags,
        _tag("OtherTag"),
        {
            "ID": "LT_default",
            "LABEL": "default",
            "DESCRIPTION": "line type default",
        },
    )

    layout = ET.SubElement(root, _tag("Layout"))
    page_id = _xml_id(f"page_{Path(result.image_name).stem}")
    page = ET.SubElement(
        layout,
        _tag("Page"),
        {
            "WIDTH": str(result.width),
            "HEIGHT": str(result.height),
            "PHYSICAL_IMG_NR": "1",
            "ID": page_id,
        },
    )
    print_space = ET.SubElement(
        page,
        _tag("PrintSpace"),
        {
            "HPOS": "0",
            "VPOS": "0",
            "WIDTH": str(result.width),
            "HEIGHT": str(result.height),
        },
    )

    if result.rows:
        image_stem = Path(result.image_name).stem
        column_ids = sorted(
            {row.column_index for row in result.rows if row.column_index is not None}
        )
        if column_ids:
            for column_index in column_ids:
                column_rows = [
                    row for row in result.rows if row.column_index == column_index
                ]
                if column_rows:
                    _append_text_block(
                        print_space,
                        column_rows,
                        image_stem,
                        f"column_{column_index:02d}",
                    )
        else:
            _append_text_block(print_space, result.rows, image_stem, "full_width")

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
