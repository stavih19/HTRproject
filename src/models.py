from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_list(self):
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Row:
    index: int
    bbox: BoundingBox
    score: float = 1.0
    baseline: Optional[List[List[int]]] = None
    polygon: Optional[List[List[int]]] = None
    column_index: Optional[int] = None
    column_row_index: Optional[int] = None
    source_row_index: Optional[int] = None

    def to_dict(self):
        result = {
            "index": self.index,
            "bbox": self.bbox.to_list(),
            "score": float(self.score),
        }
        if self.baseline is not None:
            result["baseline"] = self.baseline
        if self.polygon is not None:
            result["polygon"] = self.polygon
        if self.column_index is not None:
            result["column_index"] = self.column_index
            result["column_row_index"] = self.column_row_index
            result["source_row_index"] = self.source_row_index
        return result


@dataclass
class PageResult:
    image_name: str
    width: int
    height: int
    rows: List[Row]

    def to_dict(self):
        return {
            "image": self.image_name,
            "width": self.width,
            "height": self.height,
            "rows": [row.to_dict() for row in self.rows],
        }
