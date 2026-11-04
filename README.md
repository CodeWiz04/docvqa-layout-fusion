# docvqa-layout-fusion

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


