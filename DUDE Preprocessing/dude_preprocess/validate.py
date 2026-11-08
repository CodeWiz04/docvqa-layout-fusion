"""Quality checks for the preprocessed dataset.

Run after ``build`` to make sure every requirement of the preprocessing brief is
actually satisfied by the produced files (documents, questions, answers, OCR,
bounding boxes, page information, answer types, splits).
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Iterable

from .annotations import ANSWER_TYPES
from .utils import bbox_is_inside, read_json, read_jsonl

CHECK_NAMES = (
    "files_present",
    "question_ids_unique",
    "documents_referenced",
    "answers_match_answer_type",
    "answer_types_valid",
    "answer_pages_in_range",
    "answer_boxes_in_range",
    "ocr_pages_in_range",
    "ocr_boxes_normalized",
    "page_images_complete",
    "splits_disjoint",
)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: dict = {}

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok(),
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def _limited(seq: Iterable[str], n: int = 5) -> str:
    seq = list(seq)
    head = seq[:n]
    more = "" if len(seq) <= n else f" (+{len(seq) - n} more)"
    return ", ".join(head) + more


def validate(
    out_dir: str | Path,
    check_files: bool = False,
    ocr_sample: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Report:
    out = Path(out_dir)
    report = Report()

    docs_path = out / "documents.jsonl"
    if not docs_path.exists():
        report.error(f"missing {docs_path}")
        return report
    if not (out / "stats.json").exists():
        report.warn("stats.json missing (run the emit stage)")

    documents: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(docs_path):
        key = (row["split"], row["doc_id"])
        if key in documents:
            report.error(f"duplicate document entry {key}")
        documents[key] = row

    docs_per_split: dict[str, set[str]] = collections.defaultdict(set)
    for split, doc_id in documents:
        docs_per_split[split].add(doc_id)
    names = sorted(docs_per_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = docs_per_split[a] & docs_per_split[b]
            if overlap:
                report.error(
                    f"document leakage between '{a}' and '{b}': {len(overlap)} docs ({_limited(sorted(overlap))})"
                )

    seen_qids: set[str] = set()
    type_counter: collections.Counter = collections.Counter()
    per_split: dict[str, dict] = {}

    for split in splits:
        q_path = out / "questions" / f"{split}.jsonl"
        if not q_path.exists():
            if split in docs_per_split:
                report.error(f"missing questions file for split '{split}': {q_path}")
            continue

        counters = collections.Counter()
        missing_docs: list[str] = []
        bad_type: list[str] = []
        bad_answer_count: list[str] = []
        bad_pages: list[str] = []
        bad_boxes: list[str] = []
        empty_answer_text = 0

        for q in read_jsonl(q_path):
            counters["questions"] += 1
            qid = q["question_id"]
            if qid in seen_qids:
                report.error(f"duplicate question_id {qid}")
            seen_qids.add(qid)

            doc = documents.get((split, q["doc_id"]))
            if doc is None:
                missing_docs.append(qid)
                continue

            atype = q.get("answer_type_normalized")
            type_counter[atype or "none"] += 1
            if atype is not None and atype not in ANSWER_TYPES:
                bad_type.append(f"{qid}:{atype}")

            answers = q.get("answers") or []
            boxes = [b for group in (q.get("answer_boxes") or []) for b in group]
            if atype == "unanswerable" and (answers or boxes):
                bad_answer_count.append(f"{qid}:unanswerable-with-answers")
            if atype in {"extractive", "abstractive", "list"} and not answers and not boxes:
                bad_answer_count.append(f"{qid}:answerable-without-answers")
            if q.get("answerable") and not answers and not boxes:
                bad_answer_count.append(f"{qid}:answerable-flag-without-answers")
            if q.get("answerable") and not answers and boxes:
                empty_answer_text += 1

            num_pages = int(doc.get("num_pages") or 0)
            for page in q.get("answer_pages") or []:
                if not (0 <= int(page) < max(num_pages, 1)):
                    bad_pages.append(f"{qid}:page={page}/num_pages={num_pages}")
            for group in q.get("answer_boxes") or []:
                for box in group:
                    page = int(box.get("page", 0))
                    if not (0 <= page < max(num_pages, 1)):
                        bad_boxes.append(f"{qid}:page={page}")
                    norm = box.get("bbox_norm")
                    if norm is not None and not bbox_is_inside(norm):
                        bad_boxes.append(f"{qid}:norm={norm}")

        for label, items in (
            ("questions with a missing document", missing_docs),
            ("invalid answer types", bad_type),
            ("answer/answer_type mismatches", bad_answer_count),
            ("answer pages out of range", bad_pages),
            ("answer boxes out of range", bad_boxes),
        ):
            if items:
                report.error(f"[{split}] {len(items)} {label}: {_limited(items)}")

        # --- document level checks -----------------------------------------
        bad_pages_img: list[str] = []
        bad_ocr_pages: list[str] = []
        bad_ocr_boxes: list[str] = []
        missing_files: list[str] = []
        n_docs = 0
        for doc_id in sorted(docs_per_split.get(split, set())):
            doc = documents[(split, doc_id)]
            n_docs += 1
            num_pages = int(doc.get("num_pages") or 0)
            images = doc.get("page_images") or []
            if images and len(images) != num_pages:
                bad_pages_img.append(f"{doc_id}:{len(images)}!={num_pages}")
            if check_files:
                for rel in images:
                    if not (out / rel).exists():
                        missing_files.append(rel)
                        break

            ocr_rel = doc.get("ocr_path")
            if not ocr_rel:
                bad_ocr_pages.append(f"{doc_id}:no-ocr-path")
                continue
            ocr_file = out / ocr_rel
            if not ocr_file.exists():
                bad_ocr_pages.append(f"{doc_id}:missing-{ocr_rel}")
                continue
            if check_files and not images:
                missing_files.append(f"{doc_id}:no-page-images")
            if ocr_sample and n_docs > ocr_sample:
                continue
            try:
                ocr = read_json(ocr_file)
            except Exception as exc:
                bad_ocr_pages.append(f"{doc_id}:unreadable({exc})")
                continue
            for page in ocr.get("pages") or []:
                idx = int(page.get("page", -1))
                if not (0 <= idx < max(num_pages, 1)):
                    bad_ocr_pages.append(f"{doc_id}:page={idx}/{num_pages}")
                for tok in page.get("tokens") or []:
                    norm = tok.get("bbox_norm")
                    if norm is not None and not bbox_is_inside(norm):
                        bad_ocr_boxes.append(f"{doc_id}:{tok.get('text', '')[:20]}:{norm}")

        for label, items in (
            ("documents with more page images than pages", bad_pages_img),
            ("documents with OCR problems", bad_ocr_pages),
            ("OCR tokens outside the page", bad_ocr_boxes),
            ("missing files on disk", missing_files),
        ):
            if items:
                report.error(f"[{split}] {len(items)} {label}: {_limited(items)}")

        per_split[split] = {
            "documents": n_docs,
            "questions": counters["questions"],
            "questions_without_document": len(missing_docs),
        }
        if empty_answer_text:
            report.warn(
                f"[{split}] {empty_answer_text} answerable question(s) have a bounding box but an empty answer string"
            )

    report.info = {
        "splits": per_split,
        "answer_types": dict(type_counter),
        "documents": len(documents),
        "questions": len(seen_qids),
    }
    return report


def print_report(report: Report) -> None:
    info = report.info
    print("=" * 72)
    print("DUDE preprocessing validation")
    print("=" * 72)
    for split, s in (info.get("splits") or {}).items():
        print(
            f"  {split:<5} documents={s['documents']:<6} questions={s['questions']:<7} "
            f"missing_doc_refs={s['questions_without_document']}"
        )
    print(f"  answer types: {info.get('answer_types')}")
    print(f"  totals: documents={info.get('documents')} questions={info.get('questions')}")
    print("-" * 72)
    if report.errors:
        print(f"ERRORS ({len(report.errors)}):")
        for e in report.errors:
            print(f"  [x] {e}")
    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  [!] {w}")
    if report.ok():
        print("RESULT: OK - all checks passed")
    else:
        print("RESULT: FAILED")
    print("=" * 72)
