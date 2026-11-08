"""Create a tiny DUDE-shaped fixture so the pipeline can be tested offline.

Usage::

    python tools/make_fixture.py --out data/mini_dude

The fixture mirrors the real release layout::

    data/mini_dude/
      annotations.json
      PDF/{train,val,test}/*.pdf
      OCR/Amazon/*_due.json
      OCR/Azure/*_original.json
      OCR/Tesseract/*_original.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGE_W_PT, PAGE_H_PT = 612.0, 792.0
DPI = 200
PAGE_W = round(PAGE_W_PT / 72 * DPI)  # 1700 px
PAGE_H = round(PAGE_H_PT / 72 * DPI)  # 2200 px

WORDS = ["invoice", "number", "total", "amount", "date", "12/03/2021", "ACME", "Ltd", "page", "signature"]


def _page_words(page: int, offset: int = 0) -> list[tuple[str, list[float]]]:
    out = []
    for i, word in enumerate(WORDS):
        x = 100 + (i % 2) * 700
        y = 150 + (i // 2) * 120 + offset
        out.append((f"{word}", [float(x), float(y), 180.0, 32.0]))
    return out


def due_doc(doc_id: str, num_pages: int) -> dict:
    pages = []
    for p in range(num_pages):
        tokens = [
            {"id": i, "text": text, "bbox": bbox, "page": p}
            for i, (text, bbox) in enumerate(_page_words(p, offset=p * 40))
        ]
        pages.append(
            {
                "id": p,
                "dimensions": {"width": PAGE_W, "height": PAGE_H},
                "tokens": tokens,
            }
        )
    return {"metadata": {"doc_id": doc_id, "engine": "due"}, "pages": pages}


def azure_doc(doc_id: str, num_pages: int) -> dict:
    read_results = []
    for p in range(num_pages):
        words = []
        for text, (x, y, w, h) in _page_words(p):
            poly = [x, y, x + w, y, x + w, y + h, x, y + h]
            words.append({"boundingBox": poly, "text": text, "confidence": 0.99})
        read_results.append(
            {
                "page": p + 1,
                "width": PAGE_W,
                "height": PAGE_H,
                "unit": "pixel",
                "lines": [{"words": words}],
            }
        )
    return {"status": "succeeded", "analyzeResult": {"version": "3.2.0", "readResults": read_results}}


def textract_doc(doc_id: str, num_pages: int) -> list:
    blocks = []
    for p in range(num_pages):
        for text, (x, y, w, h) in _page_words(p):
            blocks.append(
                {
                    "BlockType": "WORD",
                    "Page": p + 1,
                    "Text": text,
                    "Confidence": 99.0,
                    "Geometry": {
                        "BoundingBox": {
                            "Left": x / PAGE_W,
                            "Top": y / PAGE_H,
                            "Width": w / PAGE_W,
                            "Height": h / PAGE_H,
                        }
                    },
                }
            )
    return [{"DocumentMetadata": {"Pages": num_pages}, "Blocks": blocks}]


def _box(page: int, x: float, y: float, w: float, h: float) -> dict:
    return {"left": x, "top": y, "width": w, "height": h, "page": page}


def annotations() -> dict:
    data = [
        # --- train ---------------------------------------------------------
        {
            "questionId": "train_doc_a_q1",
            "docId": "train_doc_a",
            "question": "What is the invoice number?",
            "answers": ["12345"],
            "answers_variants": ["#12345"],
            "answers_page_bounding_boxes": [[_box(0, 800, 150, 400, 32)]],
            "answer_type": "extractive",
            "data_split": "train",
        },
        {
            "questionId": "train_doc_a_q2",
            "docId": "train_doc_a",
            "question": "List the total amount and the date.",
            "answers": ["$1,200.00", "12/03/2021"],
            "answers_variants": [],
            "answers_page_bounding_boxes": [
                [_box(0, 100, 390, 300, 32)],
                [_box(1, 800, 270, 300, 32)],
            ],
            "answer_type": "list/extractive",
            "data_split": "train",
        },
        {
            "questionId": "train_doc_a_q3",
            "docId": "train_doc_a",
            "question": "Who signed the document in red ink?",
            "answers": [],
            "answers_variants": [],
            "answers_page_bounding_boxes": [[]],
            "answer_type": "not-answerable",
            "data_split": "train",
        },
        {
            "questionId": "train_doc_a_q4",
            "docId": "train_doc_a",
            "question": "Summarize the payment conditions.",
            "answers": ["Payment due within 30 days of the invoice date."],
            "answers_variants": [],
            "answers_page_bounding_boxes": [[_box(1, 100, 510, 900, 40)]],
            "answer_type": "abstractive",
            "data_split": "train",
        },
        {
            "questionId": "train_doc_b_q1",
            "docId": "train_doc_b",
            "question": "Which company issued the invoice?",
            "answers": ["ACME Ltd"],
            "answers_variants": ["Acme Limited"],
            "answers_page_bounding_boxes": [[_box(0, 100, 270, 400, 32)]],
            "answer_type": "extractive",
            "data_split": "train",
        },
        # --- val -----------------------------------------------------------
        {
            "questionId": "val_doc_a_q1",
            "docId": "val_doc_a",
            "question": "Describe the purpose of the document.",
            "answers": ["It is a purchase invoice for office supplies."],
            "answers_variants": [],
            "answers_page_bounding_boxes": [],
            "answer_type": "list/abstractive",
            "data_split": "val",
        },
        {
            "questionId": "val_doc_a_q2",
            "docId": "val_doc_a",
            "question": "What is the total amount?",
            "answers": ["$1,200.00"],
            "answers_variants": [],
            "answers_page_bounding_boxes": [[_box(0, 100, 390, 300, 32)]],
            "answer_type": "extractive",
            "data_split": "val",
        },
        # --- test (no ground truth, like the public release) ---------------
        {
            "questionId": "test_doc_a_q1",
            "docId": "test_doc_a",
            "question": "What is the invoice date?",
            "data_split": "test",
        },
    ]
    return {"dataset_name": "DUDE mini fixture", "dataset_version": "1.0.0", "data": data}


def make_pdf(path: Path, doc_id: str, num_pages: int, with_pdf: bool = True) -> None:
    if not with_pdf:
        return
    import pymupdf  # type: ignore

    doc = pymupdf.open()
    for p in range(num_pages):
        page = doc.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
        page.insert_text((72, 72), f"{doc_id} - page {p + 1}", fontsize=18)
        for i, word in enumerate(WORDS):
            page.insert_text((40 + (i % 2) * 250, 60 + (i // 2) * 40), word, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def make_fixture(root: str | Path, with_pdf: bool = True) -> Path:
    root = Path(root)
    (root / "PDF" / "train").mkdir(parents=True, exist_ok=True)
    (root / "PDF" / "val").mkdir(parents=True, exist_ok=True)
    (root / "PDF" / "test").mkdir(parents=True, exist_ok=True)

    specs = {
        "train_doc_a": ("train", 2),
        "train_doc_b": ("train", 1),
        "val_doc_a": ("val", 2),
        "test_doc_a": ("test", 1),
    }

    for doc_id, (split, num_pages) in specs.items():
        make_pdf(root / "PDF" / split / f"{doc_id}.pdf", doc_id, num_pages, with_pdf)
        # like the real release: every document exists for every OCR engine
        variants = {
            ("Amazon", "due"): due_doc(doc_id, num_pages),
            ("Amazon", "original"): due_doc(doc_id, num_pages),
            ("Azure", "original"): azure_doc(doc_id, num_pages),
            ("Tesseract", "original"): textract_doc(doc_id, num_pages),
        }
        for (engine, fmt), payload in variants.items():
            out_dir = root / "OCR" / engine
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{doc_id}_{fmt}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f)

    with open(root / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations(), f, indent=1)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="build a mini DUDE-like fixture")
    parser.add_argument("--out", default="data/mini_dude")
    parser.add_argument("--no-pdf", action="store_true", help="skip PDF generation")
    args = parser.parse_args()
    root = make_fixture(args.out, with_pdf=not args.no_pdf)
    print(f"fixture written to {root}")


if __name__ == "__main__":
    main()
