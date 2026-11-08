"""Build stage: documents -> pages + OCR, then annotations -> questions + splits."""

from __future__ import annotations

import collections
import datetime as _dt
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from . import __version__
from .annotations import Question, dedupe_questions, group_by_document, load_annotations
from .documents import PageImage, render_pdf
from .ocr import OCRDocument, empty_ocr, load_ocr
from .utils import (
    ensure_dir,
    in_shard,
    norm_bbox,
    read_json,
    read_jsonl,
    to_posix,
    write_json,
    write_jsonl,
)

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore
        return it


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass
class BuildConfig:
    data_dir: str
    out_dir: str
    annotations: str | None = None
    splits: tuple[str, ...] = ("train", "val", "test")
    documents: str = "render"  # render | probe | none
    ocr_engine: str = "Amazon"
    ocr_format: str = "due"  # due | original
    ocr_adapter: str = "auto"  # auto | due | azure | textract | tesseract | generic | none
    dpi: int = 200
    image_format: str = "jpeg"
    max_side: int | None = None
    jpeg_quality: int = 85
    lowercase_ocr: bool = False
    inline_ocr: bool = False
    workers: int = 0
    limit_docs: int = 0
    shard: int = 0
    num_shards: int = 1
    overwrite: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# index of raw files
# --------------------------------------------------------------------------- #


@dataclass
class RawIndex:
    pdfs: dict[str, str] = field(default_factory=dict)
    ocr: dict[str, str] = field(default_factory=dict)


def build_raw_index(data_dir: Path, ocr_engine: str, ocr_format: str) -> RawIndex:
    index = RawIndex()
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")

    for pdf in sorted(data_dir.rglob("*.pdf")):
        index.pdfs.setdefault(pdf.stem, str(pdf))

    suffix = f"_{ocr_format.lower()}.json"
    preferred = data_dir / "OCR" / ocr_engine
    candidates: Iterable[Path] = preferred.rglob(f"*{suffix}") if preferred.exists() else []
    found = list(candidates)
    if not found:  # be forgiving about the directory layout
        found = [
            p
            for p in data_dir.rglob(f"*{suffix}")
            if p.parent.name.lower() in {ocr_engine.lower(), "ocr", ocr_format.lower()}
            or p.parent.parent.name.lower() == ocr_engine.lower()
        ]
    for path in sorted(found):
        doc_id = path.name[: -len(suffix)]
        index.ocr.setdefault(doc_id, str(path))
    return index


# --------------------------------------------------------------------------- #
# per-document worker
# --------------------------------------------------------------------------- #


@dataclass
class DocTask:
    doc_id: str
    split: str
    pdf_path: str | None
    ocr_path: str | None
    pages_dir: str
    ocr_out: str
    doc_out: str
    #: question_id -> [(page, left, top, width, height), ...]
    questions: list[tuple[str, list[tuple[int, float, float, float, float]]]]


def _rel_path(path: str | Path, root: str | Path) -> str:
    """Store dataset-internal paths relative to the output folder."""
    try:
        return to_posix(Path(os.path.relpath(str(path), str(root))))
    except ValueError:  # different drives on Windows
        return to_posix(path)


def process_document(task: DocTask, cfg: BuildConfig) -> dict:
    """Render one document, normalize its OCR and answer boxes; write doc-level JSON."""
    warnings: list[str] = []

    rendered: list[PageImage] = []
    if task.pdf_path and cfg.documents != "none":
        try:
            rendered = render_pdf(
                task.pdf_path,
                task.pages_dir,
                dpi=cfg.dpi,
                image_format=cfg.image_format,
                mode=cfg.documents,
                max_side=cfg.max_side,
                jpeg_quality=cfg.jpeg_quality,
                overwrite=cfg.overwrite,
            )
        except Exception as exc:  # a broken PDF must not kill the run
            warnings.append(f"pdf_error: {exc}")

    render_dims = {p.page: (float(p.width), float(p.height)) for p in rendered}

    ocr: OCRDocument
    if task.ocr_path and cfg.ocr_adapter != "none":
        try:
            ocr = load_ocr(
                task.ocr_path,
                engine=cfg.ocr_adapter,
                lowercase=cfg.lowercase_ocr,
                page_dims=render_dims,
            )
            warnings.extend(ocr.warnings)
        except Exception as exc:
            warnings.append(f"ocr_error: {exc}")
            ocr = empty_ocr(len(rendered))
    else:
        ocr = empty_ocr(len(rendered))

    if rendered and ocr.num_pages and len(rendered) != ocr.num_pages:
        warnings.append(f"page_count_mismatch: pdf={len(rendered)} ocr={ocr.num_pages}")

    # When PDFs/OCR are not used, the annotations themselves tell us how many
    # pages a document must at least have.
    max_answer_page = max(
        (int(b[0]) for _, boxes in task.questions for b in boxes if int(b[0]) >= 0), default=-1
    )
    num_pages = max(len(rendered), ocr.num_pages, max_answer_page + 1, 1 if rendered else 0)

    # Answer boxes live in the OCR reference space (same renderer that produced
    # the OCR).  Fall back to the rendered pixel size when the OCR dump has none.
    ocr_dims = ocr.page_dims()
    ref_dims: dict[int, tuple[float, float]] = {}
    for page in range(num_pages):
        ref_dims[page] = ocr_dims.get(page) or render_dims.get(page) or (0.0, 0.0)

    page_images: list[str | None] = [None] * num_pages
    page_points: list[list[float] | None] = [None] * num_pages
    image_sizes: dict[int, list[int]] = {}
    for p in rendered:
        if p.page < num_pages:
            page_images[p.page] = _rel_path(p.path, cfg.out_dir) if p.path else None
            image_sizes[p.page] = [p.width, p.height]
            if p.width_pt and p.height_pt:
                page_points[p.page] = [round(p.width_pt, 2), round(p.height_pt, 2)]

    out_ocr_pages = []
    n_tokens = 0
    n_oob = 0
    ocr_by_page = {p.page: p for p in ocr.pages}
    for page in range(num_pages):
        page_ocr = ocr_by_page.get(page)
        width, height = ref_dims[page]
        tokens_out = []
        for tok in page_ocr.tokens if page_ocr else []:
            bbox = tok.bbox
            bbox_norm = tok.bbox_norm
            if bbox_norm is None and bbox and width and height:
                bbox_norm = norm_bbox(*bbox, width, height)
            if bbox is None and bbox_norm and width and height:
                x, y, w, h = bbox_norm
                bbox = [round(x * width, 2), round(y * height, 2), round(w * width, 2), round(h * height, 2)]
            if bbox_norm and (
                bbox_norm[0] < -0.01
                or bbox_norm[1] < -0.01
                or bbox_norm[0] + bbox_norm[2] > 1.01
                or bbox_norm[1] + bbox_norm[3] > 1.01
            ):
                n_oob += 1
            tokens_out.append(
                {
                    "text": tok.text,
                    "bbox": bbox,
                    "bbox_norm": bbox_norm,
                }
            )
        n_tokens += len(tokens_out)
        img = page_images[page]
        out_ocr_pages.append(
            {
                "page": page,
                "width": width or None,
                "height": height or None,
                "image": img,
                "image_path": to_posix(img) if img else None,
                "text": " ".join(t["text"] for t in tokens_out),
                "tokens": tokens_out,
            }
        )

    for p in out_ocr_pages:
        p.pop("image", None)

    if n_oob:
        warnings.append(f"ocr_boxes_out_of_bounds: {n_oob}")

    ocr_payload = {
        "doc_id": task.doc_id,
        "split": task.split,
        "ocr_engine": ocr.engine,
        "num_pages": num_pages,
        "num_tokens": n_tokens,
        "pages": out_ocr_pages,
    }
    write_json(task.ocr_out, ocr_payload)
    # --- answer boxes per question -----------------------------------------
    answer_boxes: dict[str, dict] = {}
    n_answer_oob = 0
    for qid, boxes in task.questions:
        groups: list[list[dict]] = []
        pages: set[int] = set()
        for page, left, top, width, height in boxes:
            pw, ph = ref_dims.get(page, (0.0, 0.0))
            norm = norm_bbox(left, top, width, height, pw, ph)
            if norm and (
                norm[0] < -0.01
                or norm[1] < -0.01
                or norm[0] + norm[2] > 1.01
                or norm[1] + norm[3] > 1.01
            ):
                n_answer_oob += 1
            entry = {
                "page": page,
                "bbox": [round(float(left), 2), round(float(top), 2), round(float(width), 2), round(float(height), 2)],
                "bbox_norm": norm,
            }
            groups.append([entry])
            pages.add(page)
        answer_boxes[qid] = {"answer_pages": sorted(pages), "boxes": groups}

    if n_answer_oob:
        warnings.append(f"answer_boxes_out_of_bounds: {n_answer_oob}")

    doc_payload = {
        "doc_id": task.doc_id,
        "split": task.split,
        "num_pages": num_pages,
        "page_images": [img for img in page_images if img],
        "page_sizes": {
            # coordinate space the OCR tokens and answer boxes refer to
            "ocr_px": [
                ([round(w, 2), round(h, 2)] if w and h else None)
                for w, h in (ref_dims[p] for p in range(num_pages))
            ],
            # actual size of the stored page images (differs when --max-side/--dpi scale them)
            "image_px": [image_sizes.get(p) for p in range(num_pages)],
            "pdf_pt": page_points,
        },
        "ocr_path": _rel_path(str(task.ocr_out), cfg.out_dir),
        "ocr_engine": ocr.engine,        "num_tokens": n_tokens,
        "num_questions": len(task.questions),
        "has_pdf": bool(task.pdf_path),
        "has_ocr": bool(task.ocr_path),
        "warnings": sorted(set(warnings)),
        "answer_boxes": answer_boxes,
    }
    write_json(task.doc_out, doc_payload)

    return {
        "doc_id": task.doc_id,
        "split": task.split,
        "num_pages": num_pages,
        "num_tokens": n_tokens,
        "warnings": len(warnings),
    }


# --------------------------------------------------------------------------- #
# stage 1: process documents
# --------------------------------------------------------------------------- #


def _doc_paths(cfg: BuildConfig, split: str, doc_id: str) -> tuple[Path, Path, Path]:
    out = Path(cfg.out_dir)
    return (
        out / "pages" / split / doc_id,
        out / "ocr" / split / f"{doc_id}.json",
        out / "docs" / split / f"{doc_id}.json",
    )


def _make_tasks(
    cfg: BuildConfig,
    split: str,
    questions: list[Question],
    index: RawIndex,
) -> list[DocTask]:
    grouped = group_by_document(questions)
    tasks: list[DocTask] = []
    for (q_split, doc_id), qs in sorted(grouped.items()):
        if q_split != split:
            continue
        if not in_shard(doc_id, cfg.shard, cfg.num_shards):
            continue
        pages_dir, ocr_out, doc_out = _doc_paths(cfg, split, doc_id)
        if (
            not cfg.overwrite
            and doc_out.exists()
            and ocr_out.exists()
            and (cfg.documents != "render" or pages_dir.exists())
        ):
            continue
        q_boxes = [
            (
                q.question_id,
                [(b.page, b.left, b.top, b.width, b.height) for g in q.answer_boxes for b in g],
            )
            for q in qs
        ]
        tasks.append(
            DocTask(
                doc_id=doc_id,
                split=split,
                pdf_path=index.pdfs.get(doc_id),
                ocr_path=index.ocr.get(doc_id),
                pages_dir=str(pages_dir),
                ocr_out=str(ocr_out),
                doc_out=str(doc_out),
                questions=q_boxes,
            )
        )
    return tasks


def process_split(cfg: BuildConfig, split: str, questions: list[Question], index: RawIndex) -> dict:
    tasks = _make_tasks(cfg, split, questions, index)
    if cfg.limit_docs and cfg.limit_docs > 0:
        tasks = tasks[: cfg.limit_docs]
    if not tasks:
        return {"documents": 0, "cached": True}

    if cfg.workers and cfg.workers > 1 and len(tasks) > 1:
        results = []
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(process_document, t, cfg): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split}"):
                task = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append(
                        {"doc_id": task.doc_id, "split": split, "error": f"{exc}", "trace": traceback.format_exc(limit=2)}
                    )
    else:
        results = []
        for task in tqdm(tasks, desc=split):
            try:
                results.append(process_document(task, cfg))
            except Exception as exc:
                results.append(
                    {"doc_id": task.doc_id, "split": split, "error": str(exc), "trace": traceback.format_exc(limit=2)}
                )

    summary = summarize(results)
    failed = summary.get("failed") or []
    if failed:
        write_jsonl(Path(cfg.out_dir) / "failed" / f"{split}.jsonl", failed)
    return summary


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    return {
        "documents": len(ok),
        "pages": sum(r.get("num_pages", 0) for r in ok),
        "tokens": sum(r.get("num_tokens", 0) for r in ok),
        "documents_with_warnings": sum(1 for r in ok if r.get("warnings")),
        "failed": [r for r in results if "error" in r],
    }


# --------------------------------------------------------------------------- #
# stage 2: emit questions / documents / stats
# --------------------------------------------------------------------------- #


def _load_doc_records(cfg: BuildConfig, split: str) -> dict[str, dict]:
    docs_dir = Path(cfg.out_dir) / "docs" / split
    records: dict[str, dict] = {}
    if not docs_dir.exists():
        return records
    for path in sorted(docs_dir.glob("*.json")):
        try:
            rec = read_json(path)
        except Exception:
            continue
        records[rec["doc_id"]] = rec
    return records


def emit(cfg: BuildConfig, questions: list[Question], duplicates: list[str] | None = None) -> dict:
    out = Path(cfg.out_dir)
    ensure_dir(out / "questions")

    grouped = group_by_document(questions)
    stats: dict = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": __version__,
        "config": cfg.as_dict(),
        "splits": {},
        "totals": {},
        "missing_documents": {},
        "duplicate_question_ids": {
            "count": len(duplicates or []),
            "examples": list(duplicates or [])[:20],
        },
    }

    doc_rows: list[dict] = []
    questions_rows: list[dict] = []

    for split in cfg.splits:
        docs = _load_doc_records(cfg, split)
        per_split_questions = {k: v for k, v in grouped.items() if k[0] == split}
        referenced = {doc_id for (_, doc_id) in per_split_questions}
        missing = sorted(referenced - set(docs))
        stats["missing_documents"][split] = missing[:200]

        counters = collections.Counter()
        answer_types: collections.Counter = collections.Counter()
        n_answers = 0
        n_answer_boxes = 0
        q_rows: list[dict] = []

        for (_, doc_id), qs in sorted(per_split_questions.items()):
            doc = docs.get(doc_id)
            for q in qs:
                counters["questions"] += 1
                answer_types[q.answer_type or "none"] += 1
                n_answers += len(q.answers)
                if q.answerable and not q.answers:
                    counters["answerable_without_answer_text"] += 1
                if doc is None:
                    counters["questions_without_document"] += 1
                    continue
                boxes_info = (doc.get("answer_boxes") or {}).get(q.question_id, {})
                boxes = boxes_info.get("boxes", [])
                n_answer_boxes += sum(len(g) for g in boxes)
                rec = {
                    "question_id": q.question_id,
                    "split": split,
                    "doc_id": doc_id,
                    "question": q.question,
                    "answers": q.answers,
                    "answer_variants": q.answer_variants,
                    "answer_type": q.answer_type_raw,
                    "answer_type_normalized": q.answer_type,
                    "answerable": q.answerable,
                    "has_ground_truth": q.has_ground_truth,
                    "answer_pages": boxes_info.get("answer_pages", q.answer_pages),
                    "answer_boxes": boxes,
                    "num_pages": doc["num_pages"],
                    "page_images": doc["page_images"],
                    "ocr_path": doc["ocr_path"],
                    "ocr_engine": doc.get("ocr_engine"),
                }
                if cfg.inline_ocr:
                    try:
                        rec["ocr_pages"] = read_json(out / doc["ocr_path"])["pages"]
                    except Exception:
                        rec["ocr_pages"] = []
                q_rows.append(rec)

        write_jsonl(out / "questions" / f"{split}.jsonl", q_rows)
        questions_rows.extend(q_rows)

        for doc in docs.values():
            doc_rows.append({k: v for k, v in doc.items() if k != "answer_boxes"})

        pages = sum(int(d.get("num_pages") or 0) for d in docs.values())
        tokens = sum(int(d.get("num_tokens") or 0) for d in docs.values())
        docs_with_ocr = sum(1 for d in docs.values() if d.get("has_ocr"))
        docs_with_pdf = sum(1 for d in docs.values() if d.get("has_pdf"))
        failed_file = out / "failed" / f"{split}.jsonl"
        n_failed = sum(1 for _ in read_jsonl(failed_file)) if failed_file.exists() else 0
        split_stats = {
            "documents": len(docs),
            "documents_with_pdf": docs_with_pdf,
            "documents_with_ocr": docs_with_ocr,
            "pages": pages,
            "ocr_tokens": tokens,
            "questions": counters["questions"],
            "questions_without_document": counters["questions_without_document"],
            "questions_with_answer_boxes": sum(1 for r in q_rows if any(r["answer_boxes"])),
            "answerable_without_answer_text": counters["answerable_without_answer_text"],
            "answer_spans": n_answer_boxes,
            "avg_answers_per_question": round(n_answers / counters["questions"], 2) if counters["questions"] else 0,
            "answer_types": dict(answer_types),
            "pages_per_document": round(pages / len(docs), 2) if docs else 0,
            "tokens_per_page": round(tokens / pages, 2) if pages else 0,
            "missing_documents": len(missing),
            "failed_documents": n_failed,
        }
        stats["splits"][split] = split_stats

    write_jsonl(out / "documents.jsonl", sorted(doc_rows, key=lambda r: (r["split"], r["doc_id"])))

    totals = collections.Counter()
    for s in stats["splits"].values():
        for key in ("documents", "pages", "ocr_tokens", "questions", "questions_with_answer_boxes", "answer_spans"):
            totals[key] += int(s.get(key, 0))
    stats["totals"] = dict(totals)

    # sanity: splits must be disjoint at document level
    docs_per_split = collections.defaultdict(set)
    for row in doc_rows:
        docs_per_split[row["split"]].add(row["doc_id"])
    overlap = {}
    names = list(docs_per_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = docs_per_split[a] & docs_per_split[b]
            if inter:
                overlap[f"{a}|{b}"] = len(inter)
    stats["split_document_overlap"] = overlap

    write_json(out / "stats.json", stats, indent=2)
    write_json(
        out / "manifest.json",
        {
            "pipeline_version": __version__,
            "created_at": stats["generated_at"],
            "config": cfg.as_dict(),
            "files": {
                "documents": "documents.jsonl",
                "questions": {s: f"questions/{s}.jsonl" for s in cfg.splits},
                "ocr_dir": "ocr/",
                "pages_dir": "pages/",
                "stats": "stats.json",
            },
        },
        indent=2,
    )
    return stats


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #


def run_build(cfg: BuildConfig, stage: str = "all") -> dict:
    data_dir = Path(cfg.data_dir)
    out = ensure_dir(cfg.out_dir)
    for sub in ("docs", "ocr", "pages", "questions"):
        ensure_dir(out / sub)

    annotations = cfg.annotations or _discover_annotations(data_dir)
    if not annotations:
        raise FileNotFoundError(
            "No annotation file found. Pass --annotations <path to "
            "2023-03-23_DUDE_gt_test_PUBLIC.json> or put it in --data-dir."
        )
    print(f"[dude] annotations: {annotations}", file=sys.stderr)
    questions = load_annotations(annotations, cfg.splits)
    questions, duplicated = dedupe_questions(questions)
    if duplicated:
        print(
            f"[dude] warning: dropped {len(duplicated)} duplicated question ids "
            f"(e.g. {duplicated[0]})",
            file=sys.stderr,
        )
    print(f"[dude] loaded {len(questions)} questions", file=sys.stderr)

    index = RawIndex()
    if stage in {"all", "process"}:
        index = build_raw_index(data_dir, cfg.ocr_engine, cfg.ocr_format)
        print(
            f"[dude] indexed {len(index.pdfs)} PDFs and {len(index.ocr)} OCR files",
            file=sys.stderr,
        )

    started = time.time()
    if stage in {"all", "process"}:
        for split in cfg.splits:
            split_questions = [q for q in questions if q.split == split]
            if not split_questions:
                continue
            print(f"[dude] processing split '{split}'", file=sys.stderr)
            summary = process_split(cfg, split, split_questions, index)
            failed = summary.get("failed") or []
            print(
                f"[dude] {split}: {summary.get('documents', 0)} documents, "
                f"{summary.get('pages', 0)} pages, {summary.get('tokens', 0)} tokens"
                + (f", {len(failed)} failed" if failed else ""),
                file=sys.stderr,
            )

    stats = {}
    if stage in {"all", "emit"}:
        stats = emit(cfg, questions, duplicated)
        print(f"[dude] wrote outputs to {out} ({time.time() - started:.1f}s)", file=sys.stderr)
        for split, s in stats["splits"].items():
            print(
                f"[dude]   {split:<5} docs={s['documents']:<6} pages={s['pages']:<7} "
                f"questions={s['questions']:<7} missing_docs={s['missing_documents']}",
                file=sys.stderr,
            )
    return stats


def _discover_annotations(data_dir: Path) -> str | None:
    if not data_dir.exists():
        return None
    patterns = ("*DUDE*.json", "*annot*.json", "*question*.json", "*gt*.json")
    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return None
