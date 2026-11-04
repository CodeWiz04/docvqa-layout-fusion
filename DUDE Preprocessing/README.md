# DUDE dataset preprocessing

A preprocessing pipeline that turns the raw **DUDE** (Document UnderstanDing of Everything)
release — PDFs, per-engine OCR dumps and the annotation file — into one clean, model-agnostic
dataset that contains **everything the preprocessing brief asks for**:

| Requirement | Where it lives in the output |
|---|---|
| **Documents** – original document pages/images | `pages/<split>/<doc_id>/page_0000.jpg` (rendered from the PDFs) + `documents.jsonl` |
| **Questions** | `questions/{train,val,test}.jsonl` → `question` |
| **Answers** (ground truth) | `questions/*.jsonl` → `answers`, `answer_variants` |
| **OCR** – text extracted from pages | `ocr/<split>/<doc_id>.json` → `pages[].text`, `pages[].tokens[].text` |
| **Bounding boxes** – OCR box coordinates | `pages[].tokens[].bbox` (OCR space) and `.bbox_norm` (`0..1`) |
| **Page information** – which page holds each element | every OCR token has `page` (0-based); documents have `num_pages`, `page_sizes`; questions have `answer_pages` |
| **Answer types** – extractive / abstractive / list / unanswerable | `answer_type` (raw, e.g. `list/extractive`) and `answer_type_normalized` (`list`) |
| **Dataset splits** – train / val / test | one file per split, split stored on every record, no document appears in two splits |

It also adds what any training/eval loop needs: answer *evidence* boxes (`answer_boxes`),
per-document page images and sizes, resumable/sharded processing, and a validation + statistics report.

---

## 1. Install

Requires **Python 3.10+**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Core dependency is **PyMuPDF** (renders pages, no external binaries needed on Windows).
`pdf2image` + poppler works as a fallback if you cannot install PyMuPDF.

## 2. Get the data

You need two things:

1. **The DUDE binaries** (PDFs + OCR), downloaded from
   <https://rrc.cvc.uab.es/?ch=23&com=downloads> (free registration) or from the
   [HuggingFace mirror](https://huggingface.co/datasets/jordyvl/DUDE_loader) (~21 GB).
   Unzip somewhere, e.g. `D:\data\DUDE_train-val-test_binaries`.
2. **The annotations** (14 MB) — this pipeline can fetch them for you:

```powershell
python -m dude_preprocess download-annotations --out data/2023-03-23_DUDE_gt_test_PUBLIC.json
```

Expected raw layout (the loader is also tolerant about the exact folder names):

```
DUDE_train-val-test_binaries/
  PDF/train/<docId>.pdf            # ~5,000 multi-page PDFs
  PDF/val/<docId>.pdf
  PDF/test/<docId>.pdf
  OCR/Amazon/<docId>_due.json      # unified DUE format
  OCR/Amazon/<docId>_original.json # native Amazon Textract dump
  OCR/Azure/<docId>_original.json
  OCR/Tesseract/<docId>_original.json
2023-03-23_DUDE_gt_test_PUBLIC.json
```

## 3. Run

```powershell
python -m dude_preprocess build `
  --data-dir D:\data\DUDE_train-val-test_binaries `
  --annotations data\2023-03-23_DUDE_gt_test_PUBLIC.json `
  --out-dir data\dude_processed `
  --ocr-engine Amazon --ocr-format due `
  --dpi 200 --max-side 3000 `
  --workers 8
```

Then check the result:

```powershell
python -m dude_preprocess validate --out-dir data\dude_processed --check-files
python -m dude_preprocess inspect  --out-dir data\dude_processed --split train -n 1
```

### Common variants

```powershell
# quick smoke test on 20 documents per split, no OCR box validation
python -m dude_preprocess build --data-dir ... --out-dir ... --limit-docs 20

# annotations + OCR only (no page images, much faster / smaller)
python -m dude_preprocess build --data-dir ... --out-dir ... --documents probe

# annotations only (e.g. you already have page images elsewhere)
python -m dude_preprocess build --data-dir ... --out-dir ... --documents none --ocr-adapter none

# different OCR engine
python -m dude_preprocess build ... --ocr-engine Azure --ocr-format original
python -m dude_preprocess build ... --ocr-engine Tesseract --ocr-format original

# re-emit the jsonl/stats after tweaking something (uses the cached per-document files)
python -m dude_preprocess build ... --stage emit

# distributed / split across machines
python -m dude_preprocess build ... --shard 0 --num-shards 4
python -m dude_preprocess build ... --shard 1 --num-shards 4 --stage process
# ... then merge the docs/ folders and run --stage emit once

# everything from a YAML file
python -m dude_preprocess build --config config.example.yaml
```

`build` is **resumable**: an interrupted run continues where it stopped, and rerunning is a no-op
for documents that are already processed (`--overwrite` forces a redo).

All CLI options: `python -m dude_preprocess build --help`.

---

## 4. Output layout

```
data/dude_processed/
  documents.jsonl                   # one line per document (manifest)
  questions/train.jsonl             # one line per question
  questions/val.jsonl
  questions/test.jsonl
  ocr/train/<docId>.json            # normalized OCR: text + boxes + page indices
  pages/train/<docId>/page_0000.jpg # rendered page images
  docs/train/<docId>.json           # per-document intermediate record (cached work)
  stats.json                        # counts, distributions, warnings, QC numbers
  manifest.json                     # run config + file map
```

### `questions/<split>.jsonl`

```json
{
  "question_id": "a49e6c..._2f48d8...",
  "split": "train",
  "doc_id": "a49e6c...",
  "question": "What are the equipment used for the ARSENIC TRIOXIDE?",
  "answers": ["1.Cellulose ester membrane filter", "2.Personal sampling pump"],
  "answer_variants": [],
  "answer_type": "list/extractive",
  "answer_type_normalized": "list",
  "answerable": true,
  "has_ground_truth": true,
  "answer_pages": [0],
  "answer_boxes": [
    [{"page": 0, "bbox": [967.0, 223.0, 540.0, 32.0], "bbox_norm": [0.43, 0.09, 0.24, 0.014]}]
  ],
  "num_pages": 4,
  "page_images": ["pages/train/a49e6c.../page_0000.jpg", "..."],
  "ocr_path": "ocr/train/a49e6c....json",
  "ocr_engine": "due"
}
```

* `answer_boxes` keeps the annotation structure: **one inner list per evidence span**
  (two answers on two pages ⇒ two groups). `answer_pages` is the deduplicated, sorted page list.
* `answer_type` is the raw DUDE label, `answer_type_normalized` is one of
  `extractive | abstractive | list | unanswerable` (`null` for the public test split, which has no ground truth).
* `answers`/`answer_variants` are `[]`; `answerable` is `false` for unanswerable questions and for test-split questions.
* Add `--inline-ocr` to also embed `ocr_pages` in every question record (self-contained, much larger).

### `ocr/<split>/<docId>.json`

```json
{
  "doc_id": "...", "split": "train", "ocr_engine": "due",
  "num_pages": 4, "num_tokens": 812,
  "pages": [
    {
      "page": 0,
      "width": 1654, "height": 2339,
      "image_path": "pages/train/.../page_0000.jpg",
      "text": "INVOICE number 12345 ...",
      "tokens": [
        {"text": "INVOICE", "bbox": [93.0, 120.0, 84.0, 18.0], "bbox_norm": [0.056, 0.051, 0.051, 0.0077]}
      ]
    }
  ]
}
```

### `documents.jsonl`

`doc_id`, `split`, `num_pages`, `page_images`, `page_sizes` (`ocr_px` / `image_px` / `pdf_pt`),
`ocr_path`, `ocr_engine`, `num_tokens`, `num_questions`, `has_pdf`, `has_ocr`, `warnings`.

### Coordinate spaces

* `bbox` is in the **OCR reference space** (whatever pixel size the OCR engine used, usually
  the page rendered at 200 dpi). Use `page_sizes.ocr_px` if you need it.
* `bbox_norm` is the same box in `[0, 1]` page space — **resolution independent**. To draw it on a
  stored page image: `x_px = bbox_norm[0] * image_width`, `y_px = bbox_norm[1] * image_height`.
  This stays correct when you change `--dpi` or `--max-side`.
* Answer boxes reuse the OCR space (that is how DUDE annotated them), so annotation boxes and OCR
  tokens line up with each other. `stats.json` reports any box that fell outside the page.

---

## 5. Answer types

| DUDE label | Normalized | Notes |
|---|---|---|
| `extractive` | `extractive` | answer copied verbatim from the page; has evidence boxes |
| `abstractive` | `abstractive` | free-form answer |
| `list/extractive` | `list` | list of extracted spans |
| `list/abstractive` | `list` | list of free-form answers |
| `not-answerable` | `unanswerable` | no answer, no evidence page |
| *(missing)* | `null` | public test split (no ground truth released) |

Counts in the public annotations: 13,275 extractive, 11,513 abstractive, 2,091 list,
3,172 unanswerable, 11,402 test (unlabelled) — 41,453 unique questions.

## 6. Splits

The split comes straight from `data_split`. Documents are split disjointly:
2,974 train / 744 val / 1,299 test documents. `validate` fails if a document id ever appears in
two splits, and `stats.json` stores the check result (`split_document_overlap`).

## 7. Validation

`validate` checks: files present, unique question ids, every question's document exists,
answers consistent with the answer type, valid answer types, answer pages/boxes in range,
OCR page indices in range, OCR boxes inside `[0, 1]`, page-image completeness, split leakage.
Exit code is non-zero on failure, so it can gate a training job.

## 8. Tests

```powershell
python tools/make_fixture.py --out data/mini_dude   # tiny synthetic DUDE-like dataset
python -m pytest tests -q
```

The fixture covers all three OCR dumps (DUE, Azure, Textract), all four answer types,
multi-page documents and the unlabelled test split.

## 9. Notes & troubleshooting

* **~21 GB** of input, ~5,000 PDFs, ~15,000 pages. Rendering at 200 dpi with 8 workers takes
  roughly 30–90 min on a laptop; `--max-side 3000` keeps big posters/drawings manageable.
* Page images are JPEG (q=85) by default; use `--image-format png` for lossless.
* If page boxes look shifted, the OCR was produced with a different render DPI. Normalized boxes
  are still internally consistent; check `stats.json` → `ocr_boxes_out_of_bounds`.
* The public annotations contain 3 duplicated `questionId`s (one with conflicting answer types);
  the pipeline keeps the first occurrence and reports the dropped ids in `stats.json`.
* A few questions have a bounding box but an empty answer string — they are kept and counted in
  `stats.json` (`answerable_without_answer_text`) and reported as a validation warning.
* Encrypted/corrupt PDFs do not abort the run: the document is kept with a `pdf_error` warning.
* Documents referenced by questions but absent from `--data-dir` are listed in
  `stats.json` → `missing_documents`.
* In `--documents none` mode (no PDFs) `num_pages` is a **lower bound** derived from the answer
  pages, and the public test split reports `0` because it has no answers. Run with
  `--documents render/probe` to get real page counts.
