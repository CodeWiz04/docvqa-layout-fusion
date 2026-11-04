"""Small filesystem / geometry helpers shared by the pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# paths & json
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_posix(path: str | os.PathLike[str]) -> str:
    """Store paths with forward slashes so the dataset is portable."""
    return Path(path).as_posix()


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], obj: Any, indent: int | None = None) -> None:
    """Atomically write JSON (no half-written files if the job is interrupted)."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
                n += 1
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return n


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

BBox = Sequence[float]  # (x, y, w, h)


def norm_bbox(
    x: float, y: float, w: float, h: float, page_w: float | None, page_h: float | None
) -> list[float] | None:
    """Convert an absolute box into ``[x, y, w, h]`` in ``[0, 1]`` page space."""
    if not page_w or not page_h or page_w <= 0 or page_h <= 0:
        return None
    return [round(x / page_w, 6), round(y / page_h, 6), round(w / page_w, 6), round(h / page_h, 6)]


def round_bbox(x: float, y: float, w: float, h: float) -> list[float]:
    return [round(float(x), 2), round(float(y), 2), round(float(w), 2), round(float(h), 2)]


def polygon_to_xywh(poly: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert a flat polygon ``[x0,y0,x1,y1,...]`` (or 4-corner list) to x,y,w,h."""
    if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
        # already x, y, w, h (DUE / Textract style)
        return float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3])
    xs = list(poly[0::2])
    ys = list(poly[1::2])
    if not xs or not ys:
        raise ValueError(f"invalid polygon: {poly!r}")
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def bbox_from_mapping(box: dict) -> tuple[float, float, float, float]:
    """Accept the several box dict flavours found in OCR dumps."""
    low = {str(k).lower(): v for k, v in box.items()}
    if {"left", "top", "width", "height"} <= low.keys():
        return float(low["left"]), float(low["top"]), float(low["width"]), float(low["height"])
    if {"x", "y", "w", "h"} <= low.keys():
        return float(low["x"]), float(low["y"]), float(low["w"]), float(low["h"])
    if {"x0", "y0", "x1", "y1"} <= low.keys():
        x0, y0 = float(low["x0"]), float(low["y0"])
        return x0, y0, float(low["x1"]) - x0, float(low["y1"]) - y0
    if {"left", "top", "right", "bottom"} <= low.keys():
        x0, y0 = float(low["left"]), float(low["top"])
        return x0, y0, float(low["right"]) - x0, float(low["bottom"]) - y0
    raise ValueError(f"unrecognised bbox mapping: {sorted(low)}")


def bbox_is_inside(bbox: Sequence[float], tol: float = 0.02) -> bool:
    x, y, w, h = bbox
    return (
        x >= -tol
        and y >= -tol
        and w >= 0
        and h >= 0
        and x + w <= 1 + tol
        and y + h <= 1 + tol
    )


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def in_shard(key: str, shard: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return stable_hash(key) % num_shards == shard


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
