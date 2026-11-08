"""End-to-end tests for the DUDE preprocessing pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dude_preprocess.annotations import load_annotations, normalize_answer_type
from dude_preprocess.build import BuildConfig, run_build
from dude_preprocess.ocr import parse_ocr
from dude_preprocess.validate import validate
from tools.make_fixture import PAGE_H, PAGE_W, make_fixture


def _has_pymupdf() -> bool:
    try:
        import pymupdf  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# unit: answer type normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("extractive", "extractive"),
        ("ABSTRACTIVE", "abstractive"),
        ("list/extractive", "list"),
        ("list/abstractive", "list"),
        ("not-answerable", "unanswerable"),
        ("Not Answerable", "unanswerable"),
        (None, None),
        ("", None),
    ],
)
def test_answer_type_normalization(raw, expected):
    assert normalize_answer_type(raw) == expected


# --------------------------------------------------------------------------- #
# unit: OCR adapters
# --------------------------------------------------------------------------- #


def test_due_adapter_normalizes_boxes():
    obj = {
        "pages": [
            {
                "id": 0,
                "dimensions": {"width": 1000, "height": 2000},
                "tokens": [
                    {"text": "hello", "bbox": [100, 200, 50, 20], "page": 0},
                    {"text": "world", "bbox": [300, 400, 80, 20], "page": 0},
                ],
            }
        ]
    }
    doc = parse_ocr(obj)
    assert doc.engine == "due"
    assert doc.num_tokens == 2
    tok = doc.pages[0].tokens[0]
    assert tok.page == 0
    assert tok.bbox == [100, 200, 50, 20]
    assert tok.bbox_norm == pytest.approx([0.1, 0.1, 0.05, 0.01])


def test_azure_adapter_reads_polygons():
    obj = {
        "analyzeResult": {
            "readResults": [
                {
                    "page": 1,
                    "width": 1000,
                    "height": 2000,
                    "unit": "pixel",
                    "lines": [
                        {
                            "words": [
                                {
                                    "boundingBox": [100, 200, 300, 200, 300, 240, 100, 240],
                                    "text": "Total",
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    }
    doc = parse_ocr(obj)
    assert doc.engine == "azure"
    assert doc.num_tokens == 1
    tok = doc.pages[0].tokens[0]
    assert tok.page == 0
    assert tok.bbox == [100.0, 200.0, 200.0, 40.0]
    assert tok.bbox_norm == pytest.approx([0.1, 0.1, 0.2, 0.02])


def test_textract_adapter_keeps_normalized_boxes():
    obj = [
        {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                {
                    "BlockType": "WORD",
                    "Page": 1,
                    "Text": "Invoice",
                    "Geometry": {
                        "BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.05}
                    },
                }
            ],
        }
    ]
    doc = parse_ocr(obj)
    assert doc.engine == "textract"
    tok = doc.pages[0].tokens[0]
    assert tok.page == 0
    assert tok.bbox_norm == pytest.approx([0.1, 0.2, 0.3, 0.05])


def test_page_numbering_is_zero_based():
    obj = {"pages": [{"id": 1, "dimensions": {"width": 100, "height": 100}, "tokens": [
        {"text": "a", "bbox": [1, 1, 5, 5], "page": 1}]}]}
    doc = parse_ocr(obj)
    assert doc.pages[0].page == 0
    assert doc.pages[0].tokens[0].page == 0


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


@pytest.fixture()
def built(tmp_path):
    data = make_fixture(tmp_path / "raw", with_pdf=_has_pymupdf())
    out = tmp_path / "out"
    cfg = BuildConfig(
        data_dir=str(data),
        out_dir=str(out),
        annotations=str(data / "annotations.json"),
        documents="render" if _has_pymupdf() else "none",
        ocr_engine="Amazon",
        ocr_format="due",
        workers=1,
    )
    stats = run_build(cfg, stage="all")
    return data, out, stats


def test_questions_and_splits(built):
    _, out, _ = built
    for split, expected in (("train", 5), ("val", 2), ("test", 1)):
        path = out / "questions" / f"{split}.jsonl"
        assert path.exists()
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == expected
        assert all(r["split"] == split for r in rows)


def test_question_schema_has_every_required_field(built):
    _, out, _ = built
    rows = [json.loads(l) for l in (out / "questions" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    q = next(r for r in rows if r["question_id"] == "train_doc_a_q2")
    assert q["question"] == "List the total amount and the date."
    assert q["answers"] == ["$1,200.00", "12/03/2021"]
    assert q["answer_type"] == "list/extractive"
    assert q["answer_type_normalized"] == "list"
    assert q["answerable"] is True
    assert sorted(q["answer_pages"]) == [0, 1]
    assert len(q["answer_boxes"]) == 2
    assert q["num_pages"] == 2
    assert q["ocr_path"].startswith("ocr/train/")

    norm = q["answer_boxes"][0][0]["bbox_norm"]
    assert norm == pytest.approx([100 / PAGE_W, 390 / PAGE_H, 300 / PAGE_W, 32 / PAGE_H], rel=1e-4)

    unanswerable = next(r for r in rows if r["question_id"] == "train_doc_a_q3")
    assert unanswerable["answer_type_normalized"] == "unanswerable"
    assert unanswerable["answers"] == []
    assert unanswerable["answerable"] is False


def test_ocr_output_structure(built):
    _, out, _ = built
    doc = json.loads((out / "ocr" / "train" / "train_doc_a.json").read_text(encoding="utf-8"))
    assert doc["num_pages"] == 2
    assert len(doc["pages"]) == 2
    for page in doc["pages"]:
        assert page["page"] in (0, 1)
        assert page["tokens"], "OCR tokens must be present"
        for tok in page["tokens"]:
            assert tok["text"]
            assert tok["bbox_norm"] is None or (0 <= tok["bbox_norm"][0] <= 1)
            assert tok["bbox_norm"] is not None, "normalized box expected for pixel-based OCR"


def test_document_manifest_and_page_images(built):
    _, out, _ = built
    docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {d["doc_id"]: d for d in docs}
    assert set(by_id) == {"train_doc_a", "train_doc_b", "val_doc_a", "test_doc_a"}
    doc = by_id["train_doc_a"]
    assert doc["num_pages"] == 2
    assert doc["ocr_engine"] == "due"
    if doc["page_images"]:
        assert len(doc["page_images"]) == 2
        assert (out / doc["page_images"][0]).exists()
        assert doc["page_sizes"]["ocr_px"][0] == [PAGE_W, PAGE_H]
        assert doc["page_sizes"]["image_px"][0] is not None


@pytest.mark.parametrize(
    "engine,fmt,expected",
    [("Amazon", "due", "due"), ("Azure", "original", "azure"), ("Tesseract", "original", "textract")],
)
def test_ocr_engine_configurations(tmp_path, engine, fmt, expected):
    data = make_fixture(tmp_path / "raw", with_pdf=False)
    out = tmp_path / "out"
    cfg = BuildConfig(
        data_dir=str(data),
        out_dir=str(out),
        annotations=str(data / "annotations.json"),
        documents="none",
        ocr_engine=engine,
        ocr_format=fmt,
        workers=1,
    )
    run_build(cfg)
    docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert docs and all(d["ocr_engine"] == expected for d in docs)
    ocr = json.loads((out / "ocr" / "train" / "train_doc_a.json").read_text(encoding="utf-8"))
    assert ocr["num_tokens"] == 20
    assert all(t["bbox_norm"] for p in ocr["pages"] for t in p["tokens"])
    report = validate(out)
    assert report.ok(), report.errors


def test_validation_passes(built):
    _, out, _ = built
    report = validate(out, check_files=bool(_has_pymupdf()))
    assert report.ok(), report.errors


def test_rebuild_is_resumable(built, tmp_path):
    data, out, _ = built
    before = len((out / "documents.jsonl").read_text(encoding="utf-8").splitlines())
    cfg = BuildConfig(
        data_dir=str(data),
        out_dir=str(out),
        annotations=str(data / "annotations.json"),
        documents="render" if _has_pymupdf() else "none",
        workers=1,
    )
    run_build(cfg, stage="all")
    after = len((out / "documents.jsonl").read_text(encoding="utf-8").splitlines())
    assert before == after
    assert (out / "stats.json").exists()
    assert (out / "manifest.json").exists()


def test_emit_only_rebuilds_from_cached_documents(built):
    data, out, _ = built
    cfg = BuildConfig(
        data_dir=str(data / "does-not-exist"),
        out_dir=str(out),
        annotations=str(data / "annotations.json"),
    )
    stats = run_build(cfg, stage="emit")
    assert stats["totals"]["questions"] == 8
    assert stats["totals"]["documents"] == 4


def test_probe_mode_is_resumable(tmp_path):
    if not _has_pymupdf():
        pytest.skip("PDF probing needs pymupdf")
    from dude_preprocess.annotations import load_annotations
    from dude_preprocess.build import build_raw_index, process_split

    data = make_fixture(tmp_path / "raw", with_pdf=True)
    out = tmp_path / "out"
    cfg = BuildConfig(
        data_dir=str(data),
        out_dir=str(out),
        annotations=str(data / "annotations.json"),
        documents="probe",
        workers=1,
    )
    run_build(cfg, stage="process")
    assert not (out / "pages" / "train").exists()  # probe must not render images
    questions = load_annotations(data / "annotations.json")
    index = build_raw_index(data, cfg.ocr_engine, cfg.ocr_format)
    summary = process_split(cfg, "train", [q for q in questions if q.split == "train"], index)
    assert summary.get("cached") is True


def test_validation_catches_broken_boxes(built):
    _, out, _ = built
    path = out / "questions" / "train.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["answer_boxes"] = [[{"page": 0, "bbox": [0, 0, 10, 10], "bbox_norm": [0.9, 0.9, 0.5, 0.5]}]]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report = validate(out)
    assert not report.ok()
    assert any("out of range" in e for e in report.errors)


def test_cli_build_validate_inspect(tmp_path):
    from dude_preprocess.cli import main

    data = make_fixture(tmp_path / "raw", with_pdf=False)
    out = tmp_path / "out"
    assert (
        main(
            [
                "build",
                "--data-dir",
                str(data),
                "--out-dir",
                str(out),
                "--documents",
                "none",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    assert (out / "questions" / "train.jsonl").exists()
    assert (out / "manifest.json").exists()
    assert main(["validate", "--out-dir", str(out)]) == 0
    assert main(["inspect", "--out-dir", str(out), "--split", "train", "-n", "1"]) == 0


def test_cli_config_file(tmp_path):
    from dude_preprocess.cli import main

    data = make_fixture(tmp_path / "raw", with_pdf=False)
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data-dir: {data.as_posix()}\nout-dir: {out.as_posix()}\n"
        "documents: none\nocr-engine: Azure\nocr-format: original\nworkers: 1\n",
        encoding="utf-8",
    )
    assert main(["build", "--config", str(cfg)]) == 0
    rows = [json.loads(l) for l in (out / "questions" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert main(["validate", "--out-dir", str(out), "--json"]) == 0
