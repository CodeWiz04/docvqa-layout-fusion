# docvqa-layout-fusion

**Layout-Guided OCR-Vision Fusion (LG-OVF) for document question answering.**
A controlled study of *how* vision, OCR text and layout are combined: five model variants (V1-V5) share one backbone and differ **only** in the fusion module, so any score difference comes from the fusion design and not from extra inputs or capacity.

[![Open the full pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CodeWiz04/docvqa-layout-fusion/blob/main/LGOVF_full_pipeline_and_results.ipynb)

---

## The idea

Most layout-aware document models concatenate or add text, position and image embeddings and let a standard self-attention stack sort it out. This project makes the fusion step the *independent variable*:

| Variant | Inputs | Fusion |
|---|---|---|
| **V1** Vision-only | page image (+ question) | none |
| **V2** OCR-only | OCR text | none |
| **V3** Vision + OCR | image + OCR text | static concatenation |
| **V4** Vision + OCR + Layout | image + OCR text + boxes | static concatenation |
| **V5** LG-OVF (proposed) | image + OCR text + boxes | query-conditioned cross-attention, kept at inference |

**Shared backbone:** ViT-Base/16 (`google/vit-base-patch16-224-in21k`) for the page, BERT-Base (`bert-base-uncased`) for question + OCR tokens, learned embedding tables over the normalised `x_min, y_min, x_max, y_max` of each OCR word (0-1000 grid), and a 2-layer Transformer decoder that generates the answer.

**V5 fusion**, per attention head: `F = softmax(((T + L) W_Q (V W_K)^T) / sqrt(d)) (V W_V)`, where `T` are the OCR text tokens, `L` their layout embeddings and `V` the vision patches. V4 and V5 receive identical inputs, so the **V5 vs V4** comparison isolates the fusion mechanism.

The notebook asserts the ablation is clean: unit tests (section 25) check that V1 cannot see OCR text, V2 cannot see the image, V2 and V3 cannot see layout, and that V4 and V5 can.

## Research questions

1. **RQ1** Does explicit inference-time fusion (V5) beat vision-only, OCR-only and concatenation baselines on DocVQA?
2. **RQ2** Does the benefit change with document complexity (DocVQA -> InfographicVQA -> ChartQA)?
3. **RQ3** Does it persist under distribution shift (multi-domain, multi-page DUDE, and a change of OCR engine)?

## Results

Trained on DocVQA only, on a **reduced compute budget** (single Colab T4; see `PRESETS` in the notebook). Scores are ANLS (%) unless noted; the test set is 2,000 questions from random whole documents of the official validation split (official test answers are not public). Brackets are 95% bootstrap CIs.

**DocVQA test ablation (E1)**

| Variant | Test ANLS | vs V4 |
|---|---|---|
| V1 Vision-only | 4.80 [3.86, 5.74] | -20.6 |
| V2 OCR-only | 24.60 [22.71, 26.49] | -0.8 |
| V3 Vision + OCR | 23.80 [21.93, 25.67] | -1.6 |
| V4 Vision + OCR + Layout | 25.40 [23.49, 27.31] | 0.0 |
| **V5 LG-OVF** | **26.80 [24.86, 28.74]** | **+1.4** |

**Transfer without retraining (E3, E4)**

| Set (metric) | V1 | V2 | V3 | V4 | V5 | V5 - V4 [95% CI] |
|---|---|---|---|---|---|---|
| DocVQA (ANLS) | 4.8 | 24.6 | 23.8 | 25.4 | **26.8** | +1.4 [-1.3, +4.1] |
| DocVQA, Tesseract OCR (ANLS) | 4.6 | 18.8 | 18.1 | 19.4 | **20.4** | +1.0 [-10.1, +12.1] |
| InfographicVQA (ANLS) | 1.2 | 5.1 | 4.6 | 5.5 | **5.9** | +0.4 [-3.3, +4.1] |
| ChartQA (relaxed acc.) | 1.8 | 2.9 | 3.2 | 3.6 | **3.9** | +0.3 [-2.1, +2.7] |
| DUDE (ANLS) | 2.1 | 7.8 | 7.4 | 8.3 | **8.6** | +0.3 [-6.0, +6.6] |

**Cost (E6)**: V5 has 241.7 M parameters vs 238.2 M for V4 (+1.5%) and about 8% lower CPU latency per question (1,049 vs 1,145 ms), because its cross-attention output is 512 tokens long instead of the 1,088 of the concatenation variants.

**Supplementary in-domain runs (Part 6, Kaggle 2x T4, 3 epochs)**: trained *on* the target dataset, V5 vs V2 gives ChartQA test ANLS 13.68 vs 12.52 and RA 13.90 vs 13.57; InfographicVQA validation ANLS 6.34 vs 6.25. These differ from the zero-shot protocol above and are reported separately.

### Read these numbers carefully

* V5 is the best variant in every experiment, but **no V5 - V4 gap is statistically significant** at 95% with these sample sizes (every CI includes zero). The consistent direction is encouraging; it is not proof.
* Absolute scores are far below published full-scale numbers because of the reduced training budget. Only comparisons *between variants trained under the same budget* are meaningful.
* The InfographicVQA copy used in Part 6 (`Ahren09/InfoVQA`) has no public test answers, so only validation ANLS is reported there.
* Some intervals in Part 7 use a conservative variance bound because per-question scores were not kept for every run (stated in the notebook).

## Repository layout

```
LGOVF_full_pipeline_and_results.ipynb   MAIN NOTEBOOK: data -> metrics -> models -> training -> results (Parts 1-7)
docvqa.ipynb                            earlier DocVQA notebook (kept for history)
info-chart.ipynb                        earlier Kaggle notebook: ChartQA / InfographicVQA in-domain runs
src/
  config.py                             shared settings (image size, tokenizer, max length, seed)
  dataset.py                            preprocessing P1-P6 + Dataset + DataLoader for the official DocVQA files
  viz.py                                sample visualisation and dataset statistics
DUDE Preprocessing/                     standalone pipeline that turns raw DUDE into a clean dataset (own README)
requirements.txt
```

### What is in the main notebook

| Part | Contents |
|---|---|
| 1 | Data loading from Drive, document-level split, page + OCR cache, loaders, statistics |
| 2 | Metrics: ANLS, relaxed accuracy, ANLSL, BLEU-4, bootstrap CIs, paired V4-vs-V5 test, `evaluate()` |
| 3 | Backbone, baselines V1-V4, proposed fusion V5, unit tests, training, computational cost |
| 4 | Main results, significance, charts, error analysis, attention maps, success/failure gallery |
| 5 | Complexity scaling (InfographicVQA, ChartQA), robustness (DUDE), OCR-shift control |
| 6 | In-domain runs on ChartQA and InfographicVQA (executed on Kaggle; outputs preserved) |
| 7 | Report compilation on CPU: every table and figure from one results file, exact parameter counts, CPU latency |

## Quick start

### Reproduce everything (Colab)

1. Click the **Open in Colab** badge and choose a **GPU runtime** (T4 is enough for the `t4` preset).
2. Put the SP-DocVQA files on your Google Drive (the notebook locates them in `CFG.src_dir` or anywhere in My Drive).
3. Run top to bottom. Pick the budget with `PRESET` (`dry` = smoke test, `t4` = reported setup, `full` = largest). Run the overfit check (section 27) and the speed probe (27b) before the long training runs.

### Use the preprocessing code locally

```bash
git clone https://github.com/CodeWiz04/docvqa-layout-fusion && cd docvqa-layout-fusion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```python
import sys; sys.path.append("src")
from transformers import AutoTokenizer
from config import CFG
from dataset import DocVQADataset
from viz import show_sample

tok = AutoTokenizer.from_pretrained(CFG.tokenizer_name)
ds = DocVQADataset("train", tok, max_samples=20)
show_sample(ds[0], tok)
```

## Data (not committed to git)

| Dataset | Used for | Where |
|---|---|---|
| **SP-DocVQA** | training and the main ablation | https://rrc.cvc.uab.es/?ch=17 (free registration) |
| **InfographicVQA** | zero-shot complexity scaling | Hugging Face `lmms-lab/DocVQA` (config `InfographicVQA`) |
| **ChartQA** | zero-shot complexity scaling | Hugging Face `HuggingFaceM4/ChartQA` |
| **DUDE** | zero-shot robustness (multi-page, multi-domain) | Hugging Face `jordyvl/DUDE_subset_100val`; full release at https://rrc.cvc.uab.es/?ch=23 |

For `src/dataset.py`, arrange DocVQA as:

```
data/docvqa/
  train/  train_v1.0_withQT.json  documents/  ocr_results/
  val/    val_v1.0_withQT.json    documents/  ocr_results/
  test/   test_v1.0.json          documents/  ocr_results/
```

## DUDE preprocessing

`DUDE Preprocessing/` is a self-contained, resumable pipeline (PyMuPDF) that turns the raw DUDE PDFs, OCR dumps and annotations into per-split JSONL files with page images, OCR tokens and boxes, answer types and evidence boxes, plus a validation report. See [`DUDE Preprocessing/README.md`](DUDE%20Preprocessing/README.md).

## Metrics

* **ANLS** (primary): Average Normalised Levenshtein Similarity with the 0.5 threshold; DocVQA, InfographicVQA, DUDE.
* **Relaxed accuracy** (5% numeric tolerance): ChartQA.
* **BLEU-4** (SacreBLEU, corpus level): secondary generation-quality diagnostic.
* **ANLSL**: list-valued answers, with Hungarian alignment of predicted and ground-truth lists.
* 95% confidence intervals by bootstrap over questions; V5 - V4 differences are paired.

## Known limitations

* Reduced training budget and subset evaluation for InfographicVQA, ChartQA and DUDE (see the notebook's Part 5 settings).
* Tesseract replaces the training-time Microsoft OCR on ChartQA and DUDE; the OCR-shift control (DocVQA re-OCR'd with Tesseract) measures how much that substitution alone changes scores.
* The vision-only variant V1 still receives the question text; otherwise it could not know what is asked.
* Every variant is trained once with a single seed (42), so seed-to-seed variance is not measured.

## References

DocVQA (Mathew et al., WACV 2021), InfographicVQA (Mathew et al., WACV 2022), ChartQA (Masry et al., ACL Findings 2022), DUDE (Van Landeghem et al., ICCV 2023), LayoutLM / LayoutLMv2 (Xu et al.), LayoutLMv3 (Huang et al.), Pix2Struct (Lee et al.), Donut (Kim et al.), Marten (Wang et al., 2025).

## License

No license file has been added yet; until one is, all rights are reserved by the authors.# docvqa-layout-fusion

Layout-Guided OCR-Vision Fusion (LG-OVF) for document VQA. Variants V1-V5 share one backbone; only the fusion module differs.

## Setup

```bash
git clone <your-repo-url> && cd docvqa-layout-fusion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Colab or Kaggle, skip the venv and run `pip install -r requirements.txt` in a cell.

## Data (not committed to git)

Register at https://rrc.cvc.uab.es/?ch=17 (DocVQA, Task 1) and download train, val and test. Arrange as:

```
data/docvqa/
  train/  train_v1.0_withQT.json  documents/  ocr_results/
  val/    val_v1.0_withQT.json    documents/  ocr_results/
  test/   test_v1.0.json          documents/  ocr_results/
```

## Layout

```
src/config.py     shared settings (image size, tokenizer, max length)
src/dataset.py    preprocessing P1-P6 + Dataset + DataLoader
src/viz.py        sample visualisation and dataset statistics
notebooks/        main notebook
report/           LaTeX + PDF
```

## Quick check

```python
import sys; sys.path.append("src")
from transformers import AutoTokenizer
from config import CFG
from dataset import DocVQADataset
from viz import show_sample

tok = AutoTokenizer.from_pretrained(CFG.tokenizer_name)
ds = DocVQADataset("train", tok, max_samples=20)
show_sample(ds[0], tok)
```


