"""PDF page handling: page count, page size and page rendering.

PyMuPDF is preferred (no external binaries needed on Windows); pdf2image/poppler
is used as a fallback when PyMuPDF is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir, to_posix


@dataclass
class PageImage:
    page: int  # 0-based
    width: int  # pixels of the stored image (or of the OCR reference space in probe mode)
    height: int
    path: str | None = None
    width_pt: float | None = None
    height_pt: float | None = None


def _import_pymupdf():
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # type: ignore

        return fitz
    except ImportError:
        return None


def render_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
    image_format: str = "jpeg",
    mode: str = "render",
    max_side: int | None = None,
    jpeg_quality: int = 85,
    overwrite: bool = False,
) -> list[PageImage]:
    """Render (or just probe) every page of a PDF.

    Parameters
    ----------
    mode
        ``render``  -> write page images and report their pixel size
        ``probe``   -> only read page count/size (cheap; useful for OCR-text-only runs)
        ``none``    -> do not touch the PDF at all
    max_side
        Downscale pages whose longest side exceeds this many pixels.  Normalized
        boxes remain valid, only the stored image resolution changes.
    """
    mode = (mode or "render").lower()
    if mode == "none":
        return []

    pdf_path = Path(pdf_path)
    fmt = image_format.lower().lstrip(".")
    fmt = "jpg" if fmt == "jpeg" else fmt

    pymupdf = _import_pymupdf()
    if pymupdf is not None:
        return _render_with_pymupdf(
            pymupdf, pdf_path, out_dir, dpi, fmt, mode, max_side, jpeg_quality, overwrite
        )

    try:
        import pdf2image  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PDF rendering needs `pymupdf` (recommended) or `pdf2image` + poppler. "
            "Install with `pip install pymupdf`, or run with --documents none/probe."
        ) from exc

    if mode == "probe":
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        pages: list[PageImage] = []
        for i, page in enumerate(reader.pages):
            box = page.mediabox
            w_pt, h_pt = float(box.width), float(box.height)
            scale = dpi / 72.0
            if max_side:
                longest = max(w_pt, h_pt) * scale
                if longest > max_side:
                    scale *= max_side / longest
            pages.append(
                PageImage(i, int(round(w_pt * scale)), int(round(h_pt * scale)), None, w_pt, h_pt)
            )
        return pages

    images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
    target_dir = ensure_dir(out_dir)
    pages = []
    for i, img in enumerate(images):
        scale = 1.0
        if max_side and max(img.width, img.height) > max_side:
            scale = max_side / max(img.width, img.height)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        path = Path(target_dir) / f"page_{i:04d}.{fmt}"
        if overwrite or not path.exists():
            img.convert("RGB").save(path, quality=jpeg_quality) if fmt in {"jpg", "jpeg"} else img.save(path)
        pages.append(PageImage(i, img.width, img.height, to_posix(path), None, None))
    return pages


def _render_with_pymupdf(
    pymupdf,
    pdf_path: Path,
    out_dir,
    dpi: int,
    fmt: str,
    mode: str,
    max_side: int | None,
    jpeg_quality: int,
    overwrite: bool,
) -> list[PageImage]:
    doc = pymupdf.open(str(pdf_path))
    try:
        if doc.needs_pass:
            raise RuntimeError(f"encrypted PDF: {pdf_path}")
        target_dir = ensure_dir(out_dir) if mode == "render" else None
        pages: list[PageImage] = []
        for i, page in enumerate(doc):
            rect = page.rect
            w_pt, h_pt = float(rect.width), float(rect.height)
            zoom = dpi / 72.0
            if max_side:
                longest = max(w_pt, h_pt) * zoom
                if longest > max_side:
                    zoom *= max_side / longest

            if mode == "probe":
                pages.append(
                    PageImage(i, int(round(w_pt * zoom)), int(round(h_pt * zoom)), None, w_pt, h_pt)
                )
                continue

            path = Path(target_dir) / f"page_{i:04d}.{fmt}"
            if overwrite or not path.exists():
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                if fmt in {"jpg", "jpeg"}:
                    try:
                        from PIL import Image  # type: ignore

                        mode_ = "RGB" if pix.n >= 3 else "L"
                        img = Image.frombytes(mode_, (pix.width, pix.height), pix.samples)
                        img.convert("RGB").save(path, quality=jpeg_quality, optimize=True)
                    except ImportError:
                        pix.save(str(path))
                else:
                    pix.save(str(path))
                width, height = pix.width, pix.height
            else:
                with pymupdf.open(str(path)) as im:
                    rect_img = im[0].rect
                width, height = int(round(rect_img.width)), int(round(rect_img.height))
            pages.append(PageImage(i, width, height, to_posix(path), w_pt, h_pt))
        return pages
    finally:
        doc.close()
