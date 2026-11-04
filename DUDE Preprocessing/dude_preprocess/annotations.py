"""Loading and normalizing DUDE annotations (questions, answers, answer types).

The official DUDE release ships a single JSON file
(``2023-03-23_DUDE_gt_test_PUBLIC.json``) with the shape::

    {"dataset_name": "DUDE Dataset 1.0.8",
     "dataset_version": "1.0.8",
     "data": [ {questionId, question, answers, answers_variants,
                answers_page_bounding_boxes, answer_type, docId, data_split}, ... ]}

The RRC platform download instead ships one ``questions.json`` per split folder.
Both layouts (and plain lists) are accepted here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .utils import read_json

SPLITS = ("train", "val", "test")

#: Canonical answer types required by the benchmark description.
ANSWER_TYPES = ("extractive", "abstractive", "list", "unanswerable")


def normalize_answer_type(raw: str | None) -> str | None:
    """Map the many DUDE spellings onto the four canonical answer types.

    ``extractive``      -> ``extractive``
    ``abstractive``     -> ``abstractive``
    ``list/extractive`` -> ``list``   (a list of extracted spans)
    ``list/abstractive``-> ``list``   (a list of abstracted answers)
    ``not-answerable``  -> ``unanswerable``
    """
    if raw is None:
        return None
    key = re.sub(r"[\s_]+", "-", str(raw).strip().lower())
    if not key:
        return None
    if "list" in key:
        return "list"
    if "extract" in key:
        return "extractive"
    if "abstract" in key:
        return "abstractive"
    if "answerable" in key or key in {"unanswerable", "no-answer", "none"}:
        return "unanswerable"
    return None


@dataclass
class RawBBox:
    page: int
    left: float
    top: float
    width: float
    height: float


@dataclass
class Question:
    question_id: str
    doc_id: str
    split: str
    question: str
    answers: list[str] = field(default_factory=list)
    answer_variants: list[str] = field(default_factory=list)
    answer_type_raw: str | None = None
    answer_type: str | None = None
    #: one group of boxes per evidence span, like the source annotations
    answer_boxes: list[list[RawBBox]] = field(default_factory=list)
    has_ground_truth: bool = False

    @property
    def answerable(self) -> bool:
        if not self.has_ground_truth or self.answer_type in {None, "unanswerable"}:
            return False
        # A handful of DUDE questions carry boxes but an empty answer string;
        # they are still answerable (there is evidence on the page).
        return bool(self.answers) or bool(self.answer_boxes)

    @property
    def answer_pages(self) -> list[int]:
        pages = {b.page for group in self.answer_boxes for b in group}
        return sorted(p for p in pages if p >= 0)


def _first(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _parse_boxes(raw: Any) -> list[list[RawBBox]]:
    """``answers_page_bounding_boxes`` is ``[[{left,top,width,height,page}, ...], ...]``.

    Older exports wrap everything in one extra list level; both are handled.
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    groups: list[list[RawBBox]] = []
    for item in raw:
        if isinstance(item, dict):
            groups.append([_box(item)])
        elif isinstance(item, list):
            if item and isinstance(item[0], dict):
                groups.append([_box(b) for b in item])
            elif item and isinstance(item[0], list):
                for sub in item:
                    groups.append([_box(b) for b in sub if isinstance(b, dict)])
    return [g for g in groups if g]


def _box(b: dict) -> RawBBox:
    return RawBBox(
        page=int(_first(b, "page", "page_index", "page_idx", default=0)),
        left=float(_first(b, "left", "x", "x0", default=0.0)),
        top=float(_first(b, "top", "y", "y0", default=0.0)),
        width=float(_first(b, "width", "w", default=0.0)),
        height=float(_first(b, "height", "h", default=0.0)),
    )


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v is not None]
    out: list[str] = []
    for v in value:
        if v is None:
            continue
        if isinstance(v, dict):
            v = _first(v, "text", "answer", "value", default="")
        text = str(v)
        if text.strip():
            out.append(text)
    return out


def normalize_record(raw: dict, default_split: str) -> Question:
    qid = str(_first(raw, "questionId", "question_id", "id", default=""))
    doc_id = str(_first(raw, "docId", "doc_id", "document_id", "file_id", default=""))
    if not doc_id and "_" in qid:
        doc_id = qid.split("_")[0]
    if not qid:
        qid = f"{doc_id}_{abs(hash((doc_id, raw.get('question', '')))) % 10**16:016x}"

    split = str(_first(raw, "data_split", "split", "set_name", default=default_split))
    split = "val" if split.lower() in {"valid", "validation", "dev"} else split.lower()

    answers = _as_str_list(_first(raw, "answers", "answer", default=[]))
    boxes = _parse_boxes(_first(raw, "answers_page_bounding_boxes", "answers_bboxes", default=[]))
    type_raw = _first(raw, "answer_type", "answerType", "qtype", default=None)
    type_raw = None if type_raw is None else str(type_raw)

    # The public test release carries no answers / answer types at all.
    has_gt = bool(answers) or bool(boxes) or type_raw is not None

    return Question(
        question_id=qid,
        doc_id=doc_id,
        split=split,
        question=str(_first(raw, "question", "text", "query", default="")).strip(),
        answers=answers,
        answer_variants=_as_str_list(_first(raw, "answers_variants", "answer_variants", default=[])),
        answer_type_raw=type_raw,
        answer_type=normalize_answer_type(type_raw),
        answer_boxes=boxes,
        has_ground_truth=has_gt,
    )


# --------------------------------------------------------------------------- #
# discovery / loading
# --------------------------------------------------------------------------- #

_ANNOTATION_NAME_RE = re.compile(r"(annot|question|gt|ground)", re.IGNORECASE)


def _load_file(path: Path, default_split: str) -> list[Question]:
    obj = read_json(path)
    if isinstance(obj, dict):
        # official single-file release
        records = obj.get("data") or obj.get("annotations") or obj.get("questions")
        if records is None:
            raise ValueError(
                f"{path}: expected a JSON list of annotations or an object with a 'data' list"
            )
    else:
        records = obj
    if isinstance(records, dict):
        records = [records]

    split_hint = default_split or path.parent.name.lower()
    split_hint = split_hint if split_hint in SPLITS else "train"
    return [normalize_record(r, split_hint) for r in records if isinstance(r, dict)]


def load_annotations(path: str | Path, splits: Iterable[str] = SPLITS) -> list[Question]:
    """Load questions from a single annotation file or a dataset directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotations not found: {path}")

    questions: list[Question] = []
    if path.is_file():
        questions = _load_file(path, "")
    else:
        files = sorted(
            p
            for p in path.rglob("*.json")
            if _ANNOTATION_NAME_RE.search(p.name) and "OCR" not in p.parts
        )
        if not files:
            raise FileNotFoundError(
                f"no annotation json found under {path} (looked for names containing "
                "'annot', 'question', 'gt' or 'ground')"
            )
        seen: set[str] = set()
        for f in files:
            split = f.parent.name.lower() if f.parent.name.lower() in SPLITS else ""
            for q in _load_file(f, split):
                if q.question_id in seen:
                    continue
                seen.add(q.question_id)
                questions.append(q)

    wanted = set(splits)
    questions = [q for q in questions if q.split in wanted]
    questions.sort(key=lambda q: (q.split, q.doc_id, q.question_id))
    return questions


def group_by_document(questions: Iterable[Question]) -> dict[tuple[str, str], list[Question]]:
    grouped: dict[tuple[str, str], list[Question]] = {}
    for q in questions:
        grouped.setdefault((q.split, q.doc_id), []).append(q)
    return grouped


def dedupe_questions(questions: Iterable[Question]) -> tuple[list[Question], list[str]]:
    """Drop repeated ``questionId``s (the public release contains a few).

    Returns the kept questions and the ids that were dropped.
    """
    kept: list[Question] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for q in questions:
        if q.question_id in seen:
            dropped.append(q.question_id)
            continue
        seen.add(q.question_id)
        kept.append(q)
    return kept, dropped
