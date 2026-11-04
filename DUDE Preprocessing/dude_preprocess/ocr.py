"""OCR adapters.

The official DUDE release provides pre-computed OCR from three engines, both in
the engine's native dump (``*_original.json``) and in the unified *DUE* format
(``*_due.json``).  This module turns any of them into one normalized structure::

    OCRDocument(engine="due", pages=[PageOCR(page=0, width=1654, height=2339,
                 tokens=[Token(text="invoice", bbox=[93, 120, 84, 18],
                               bbox_norm=[0.056, 0.051, 0.051, 0.0077], page=0)]), ...])

Every token keeps its page index, its box in the OCR reference space and the
same box normalized to ``[0, 1]`` so it stays valid for any render resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .utils import (
    bbox_from_mapping,
    norm_bbox,
    polygon_to_xywh,
    read_json,
    round_bbox,
)

OCR_ENGINES = ("auto", "due", "azure", "textract", "tesseract", "generic")


@dataclass
class Token:
    text: str
    page: int
    bbox: list[float] | None = None  # x, y, w, h in OCR reference space (pixels)
    bbox_norm: list[float] | None = None  # x, y, w, h in [0, 1]
    confidence: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class PageOCR:
    page: int
    width: float | None = None
    height: float | None = None
    tokens: list[Token] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)


@dataclass
class OCRDocument:
    engine: str
    pages: list[PageOCR] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def num_tokens(self) -> int:
        return sum(len(p.tokens) for p in self.pages)

    def page_dims(self) -> dict[int, tuple[float, float]]:
        return {
            p.page: (p.width, p.height)
            for p in self.pages
            if p.width and p.height and p.width > 0 and p.height > 0
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_TEXT_KEYS = ("text", "Text", "value", "Value", "content", "word", "Word", "token")
_BOX_KEYS = ("bbox", "BBox", "boundingBox", "BoundingBox", "box", "Box", "polygon", "Polygon")
_PAGE_KEYS = ("page", "Page", "page_index", "pageNumber", "PageNumber", "page_id", "id")
_CONF_KEYS = ("confidence", "Confidence", "score")


def _text_of(node: dict) -> str | None:
    for k in _TEXT_KEYS:
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _page_of(node: dict) -> int | None:
    for k in ("page", "page_index", "pageNumber", "PageNumber", "Page"):
        v = node.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _box_of(node: dict) -> tuple[list[float], bool] | None:
    """Return ``(x, y, w, h, ...)`` plus a flag telling whether the box is already normalized."""
    for k in _BOX_KEYS:
        if k not in node:
            continue
        v = node[k]
        if isinstance(v, dict):
            try:
                x, y, w, h = bbox_from_mapping(v)
            except ValueError:
                continue
            # A box is normalized only if every value already fits in [0, 1].
            normalized = all(0.0 <= c <= 1.0 for c in (x, y, w, h))
            return round_bbox(x, y, w, h), normalized
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            if all(isinstance(x, (int, float)) for x in v):
                if len(v) >= 8:
                    return round_bbox(*polygon_to_xywh(list(v))), False
                if len(v) == 4:
                    return round_bbox(*[float(x) for x in v]), False
            elif all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in v):
                flat = [float(c) for pt in v for c in pt[:2]]
                return round_bbox(*polygon_to_xywh(flat)), False
    # Textract style: {"Geometry": {"BoundingBox": {...}}}
    geom = node.get("Geometry") or node.get("geometry")
    if isinstance(geom, dict):
        bb = geom.get("BoundingBox") or geom.get("bbox")
        if isinstance(bb, dict):
            try:
                x, y, w, h = bbox_from_mapping(bb)
                return round_bbox(x, y, w, h), True  # Textract boxes are 0..1
            except ValueError:
                return None
    return None


def _confidence_of(node: dict) -> float | None:
    for k in _CONF_KEYS:
        v = node.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _add_token(
    tokens: list[Token],
    text: str,
    page: int,
    box: tuple[list[float], bool] | None,
    conf: float | None = None,
) -> None:
    text = text.strip()
    if not text:
        return
    if box is None:
        tokens.append(Token(text=text, page=page, confidence=conf))
        return
    bbox, normalized = box
    if normalized:
        tokens.append(Token(text=text, page=page, bbox_norm=bbox, confidence=conf))
    else:
        tokens.append(Token(text=text, page=page, bbox=bbox, confidence=conf))


# --------------------------------------------------------------------------- #
# engine parsers -> (dims, tokens)
# --------------------------------------------------------------------------- #


def _is_due(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    pages = obj.get("pages")
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        return "tokens" in pages[0] or "dimensions" in pages[0]
    return "tokens" in obj and ("pages" in obj or "metadata" in obj)


def _parse_due(obj: dict) -> tuple[dict[int, tuple[float, float]], list[Token]]:
    dims: dict[int, tuple[float, float]] = {}
    tokens: list[Token] = []

    pages = obj.get("pages")
    if isinstance(pages, list) and pages:
        for i, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            idx = page.get("id", page.get("page", i))
            idx = int(idx) if isinstance(idx, (int, float)) else i
            d = page.get("dimensions") or {}
            w = d.get("width") or page.get("width")
            h = d.get("height") or page.get("height")
            if w and h:
                dims[idx] = (float(w), float(h))
            for tok in page.get("tokens") or page.get("text_blocks") or page.get("words") or []:
                if not isinstance(tok, dict):
                    continue
                text = _text_of(tok)
                if not text:
                    continue
                t_page = tok.get("page", idx)
                t_page = int(t_page) if isinstance(t_page, (int, float)) else idx
                _add_token(tokens, text, t_page, _box_of(tok), _confidence_of(tok))
    else:
        meta = obj.get("metadata") or {}
        d = obj.get("dimensions") or meta.get("dimensions") or {}
        if d.get("width") and d.get("height"):
            dims[0] = (float(d["width"]), float(d["height"]))
        for tok in obj.get("tokens") or []:
            if not isinstance(tok, dict) or not _text_of(tok):
                continue
            _add_token(tokens, _text_of(tok) or "", int(tok.get("page", 0)), _box_of(tok), _confidence_of(tok))

    return dims, tokens


def _parse_azure(obj: dict) -> tuple[dict[int, tuple[float, float]], list[Token], list[str]]:
    warnings: list[str] = []
    root = obj.get("analyzeResult") or obj
    pages = root.get("readResults") or root.get("pages") or []
    dims: dict[int, tuple[float, float]] = {}
    tokens: list[Token] = []

    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        idx = int(page.get("page", i + 1)) - 1
        w, h = page.get("width"), page.get("height")
        unit = str(page.get("unit", "pixel")).lower()
        if unit not in {"pixel", "pixels"}:
            warnings.append(f"azure page {idx}: unit={unit!r} treated as pixels")
        if w and h:
            dims[idx] = (float(w), float(h))

        words: list[dict] = []
        for line in page.get("lines") or []:
            line_words = line.get("words") if isinstance(line, dict) else None
            if line_words:
                words.extend(line_words)
            elif isinstance(line, dict) and line.get("text"):
                words.append(
                    {"text": line["text"], "polygon": line.get("boundingBox") or line.get("polygon")}
                )
        words.extend(page.get("words") or [])
        for word in words:
            if not isinstance(word, dict):
                continue
            text = _text_of(word)
            if not text:
                continue
            box = _box_of(word)
            if box is None and word.get("boundingBox"):
                box = (round_bbox(*polygon_to_xywh(word["boundingBox"])), False)
            _add_token(tokens, text, idx, box, _confidence_of(word))

    return dims, tokens, warnings


def _parse_textract(obj: Any) -> tuple[dict[int, tuple[float, float]], list[Token], list[str]]:
    warnings: list[str] = []
    items = obj if isinstance(obj, list) else [obj]
    dims: dict[int, tuple[float, float]] = {}
    tokens: list[Token] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        blocks = item.get("Blocks") or item.get("blocks") or []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("BlockType", "WORD")).upper() not in {"WORD", "LINE"}:
                continue
            text = _text_of(block)
            if not text:
                continue
            page = _page_of(block)
            page = (page - 1) if page and page >= 1 else 0
            _add_token(tokens, text, page, _box_of(block), _confidence_of(block))
    if not dims:
        warnings.append("textract output has no page dimensions; boxes stay normalized")
    return dims, tokens, warnings


def _walk_generic(node: Any, page_hint: int = 0, depth: int = 0) -> Iterator[tuple[str, int, Any, float | None]]:
    if depth > 40:
        return
    if isinstance(node, dict):
        page = _page_of(node)
        page = page if page is not None else page_hint
        text = _text_of(node)
        box = _box_of(node)
        if text and box is not None:
            yield text, page, box, _confidence_of(node)
        for key, value in node.items():
            if key in _TEXT_KEYS:
                continue
            yield from _walk_generic(value, page, depth + 1)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_generic(value, page_hint, depth + 1)


def _parse_generic(obj: Any) -> tuple[dict[int, tuple[float, float]], list[Token], list[str]]:
    dims: dict[int, tuple[float, float]] = {}

    def collect_dims(node: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(node, dict):
            w, h = node.get("width"), node.get("height")
            page = _page_of(node)
            if w and h and isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 1 and h > 1:
                dims[page if page is not None else 0] = (float(w), float(h))
            else:
                d = node.get("dimensions")
                if isinstance(d, dict) and d.get("width") and d.get("height"):
                    dims[page if page is not None else 0] = (float(d["width"]), float(d["height"]))
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    collect_dims(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                collect_dims(value, depth + 1)

    collect_dims(obj)
    tokens: list[Token] = []
    for text, page, box, conf in _walk_generic(obj):
        _add_token(tokens, text, page, box, conf)
    return dims, tokens, ["generic adapter used; token count may be approximate"]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def detect_engine(obj: Any) -> str:
    if _is_due(obj):
        return "due"
    if isinstance(obj, dict):
        if "analyzeResult" in obj or "readResults" in obj:
            return "azure"
        if "Blocks" in obj or "DocumentMetadata" in obj:
            return "textract"
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and ("Blocks" in first or "DocumentMetadata" in first):
            return "textract"
    return "generic"


def parse_ocr(obj: Any, engine: str = "auto") -> OCRDocument:
    engine = (engine or "auto").lower()
    if engine == "auto":
        engine = detect_engine(obj)

    warnings: list[str] = []
    if engine == "due":
        dims, tokens = _parse_due(obj)
    elif engine in {"azure", "microsoft"}:
        dims, tokens, warnings = _parse_azure(obj)
        engine = "azure"
    elif engine in {"textract", "amazon"}:
        dims, tokens, warnings = _parse_textract(obj)
        engine = "textract"
    else:
        dims, tokens, warnings = _parse_generic(obj)
        engine = "generic"

    return _assemble(engine, dims, tokens, warnings)


def _assemble(
    engine: str,
    dims: dict[int, tuple[float, float]],
    tokens: list[Token],
    warnings: list[str],
) -> OCRDocument:
    # 1-normalize page numbering to 0-based
    used_pages = {t.page for t in tokens} | set(dims)
    if used_pages and 0 not in used_pages:
        tokens = [Token(t.text, t.page - 1, t.bbox, t.bbox_norm, t.confidence) for t in tokens]
        dims = {max(0, p - 1): v for p, v in dims.items()}

    pages: dict[int, PageOCR] = {}
    for page, (w, h) in dims.items():
        pages[page] = PageOCR(page=page, width=w, height=h)
    for t in tokens:
        page = pages.setdefault(t.page, PageOCR(page=t.page))
        if t.bbox is None and t.bbox_norm is not None and page.width and page.height:
            x, y, w, h = t.bbox_norm
            t.bbox = round_bbox(x * page.width, y * page.height, w * page.width, h * page.height)
        if t.bbox_norm is None and t.bbox is not None and page.width and page.height:
            t.bbox_norm = norm_bbox(*t.bbox, page.width, page.height)
        if not t.is_empty:
            page.tokens.append(t)

    ordered = [pages[k] for k in sorted(pages)]
    missing = [p.page for p in ordered if not p.tokens]
    if missing:
        warnings.append(f"{len(missing)} page(s) without OCR tokens")
    return OCRDocument(engine=engine, pages=ordered, warnings=warnings)


def load_ocr(
    path: str | Path,
    engine: str = "auto",
    lowercase: bool = False,
    page_dims: dict[int, tuple[float, float]] | None = None,
) -> OCRDocument:
    """Read one OCR dump and normalize it.

    ``page_dims`` (from the rendered PDF) is used to derive pixel boxes for
    engines that only emit normalized coordinates, and to normalize boxes when
    the OCR dump itself carries no page dimensions.
    """
    obj = read_json(path)
    doc = parse_ocr(obj, engine)
    if lowercase:
        for page in doc.pages:
            for tok in page.tokens:
                tok.text = tok.text.lower()

    if page_dims:
        for page in doc.pages:
            if not page.width or not page.height:
                if page.page in page_dims:
                    page.width, page.height = page_dims[page.page]
        for page in doc.pages:
            for tok in page.tokens:
                if tok.bbox_norm is None and tok.bbox is not None and page.width and page.height:
                    tok.bbox_norm = norm_bbox(*tok.bbox, page.width, page.height)
                if tok.bbox is None and tok.bbox_norm is not None and page.width and page.height:
                    x, y, w, h = tok.bbox_norm
                    tok.bbox = round_bbox(x * page.width, y * page.height, w * page.width, h * page.height)
    return doc


def empty_ocr(num_pages: int = 0) -> OCRDocument:
    return OCRDocument(engine="none", pages=[PageOCR(page=i) for i in range(num_pages)])
