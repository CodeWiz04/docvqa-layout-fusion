"""Inspect processed DUDE records and (optionally) draw their boxes on the page images.

Examples::

    # print a few records as a compact summary
    python tools/check_results.py --out-dir data/dude_processed --split train --n 3

    # draw answer boxes (red) and OCR boxes (green) on the pages, save PNGs
    python tools/check_results.py --out-dir data/dude_processed --split train --n 3 --save-overlays checks

    # random sample instead of the first N records
    python tools/check_results.py --out-dir data/dude_processed --split val --n 5 --random

This is a *visual* check that complements `python -m dude_preprocess validate`
(which is the automated, pass/fail check).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dude_preprocess.utils import read_json, read_jsonl  # noqa: E402


def load_questions(out: Path, split: str) -> list[dict]:
    path = out / "questions" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} (run the build stage first)")
    return list(read_jsonl(path))


def print_record(q: dict, width: str = "full") -> None:
    answers = q.get("answers") or []
    boxes = [b for group in (q.get("answer_boxes") or []) for b in group]
    print("-" * 78)
    print(f"question_id : {q['question_id']}")
    print(f"split/doc   : {q['split']} / {q['doc_id']}   ({q['num_pages']} pages)")
    print(f"question    : {q['question']}")
    print(f"answers     : {answers if len(str(answers)) < 160 else str(answers)[:157] + '...'}")
    print(f"answer_type : {q.get('answer_type')} -> {q.get('answer_type_normalized')}  (answerable={q.get('answerable')})")
    print(f"answer_pages: {q.get('answer_pages')}   boxes={len(boxes)}")
    for i, b in enumerate(boxes[:3]):
        print(f"  box[{i}] page={b.get('page')} bbox={b.get('bbox')} bbox_norm={b.get('bbox_norm')}")
    print(f"ocr         : {q.get('ocr_path')}  (engine={q.get('ocr_engine')})")
    if q.get("page_images"):
        print(f"page_images : {len(q['page_images'])} e.g. {q['page_images'][0]}")


def _open_page(out: Path, q: dict, page: int):
    from PIL import Image  # imported lazily so the tool works without pillow for --no-overlay

    images = q.get("page_images") or []
    if page >= len(images):
        return None
    path = out / images[page]
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def draw_overlay(out: Path, q: dict, save_dir: Path) -> list[Path]:
    """Draw OCR boxes (green) and answer boxes (red) on every page that has them."""
    from PIL import ImageDraw

    ocr = read_json(out / q["ocr_path"]) if (out / q["ocr_path"]).exists() else {"pages": []}
    answer_boxes = [b for group in (q.get("answer_boxes") or []) for b in group]
    pages = {b.get("page", 0) for b in answer_boxes} or {0}
    written: list[Path] = []

    for page in sorted(pages):
        image = _open_page(out, q, page)
        if image is None:
            continue
        draw = ImageDraw.Draw(image)
        w, h = image.size

        page_ocr = next((p for p in ocr.get("pages") or [] if int(p.get("page", -1)) == page), None)
        for tok in (page_ocr or {}).get("tokens") or []:
            norm = tok.get("bbox_norm")
            if not norm:
                continue
            x, y, bw, bh = norm
            draw.rectangle([x * w, y * h, (x + bw) * w, (y + bh) * h], outline=(0, 160, 0), width=1)

        for box in answer_boxes:
            if int(box.get("page", 0)) != page:
                continue
            norm = box.get("bbox_norm")
            if not norm:
                continue
            x, y, bw, bh = norm
            draw.rectangle([x * w, y * h, (x + bw) * w, (y + bh) * h], outline=(220, 0, 0), width=4)

        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if len(pages) == 1 else f"_p{page}"
        path = save_dir / f"{q['question_id'][:24]}{suffix}.png"
        image.save(path)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="inspect / visualize processed DUDE records")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=3, help="how many records to show")
    parser.add_argument("--random", action="store_true", help="pick a random sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-overlays", default=None, help="folder for annotated PNG pages")
    args = parser.parse_args()

    out = Path(args.out_dir)
    rows = load_questions(out, args.split)
    if not rows:
        print(f"no questions in split '{args.split}'")
        return
    if args.random:
        random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]

    print("=" * 78)
    print(f"{args.split}: {len(rows)} questions, showing {len(sample)}")
    print("=" * 78)

    for q in sample:
        print_record(q)
        if args.save_overlays:
            written = draw_overlay(out, q, Path(args.save_overlays))
            if written:
                for p in written:
                    print(f"  overlay -> {p}")
            else:
                print("  (no page images to draw on; run build with --documents render)")

    types: dict[str, int] = {}
    for q in rows:
        key = str(q.get("answer_type_normalized"))
        types[key] = types.get(key, 0) + 1
    print("-" * 78)
    print(f"answer types in this split: {dict(sorted(types.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
