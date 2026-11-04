"""Preprocessing pipeline for the DUDE (Document UnderstanDing of Everything) dataset.

Turns the raw DUDE distribution (PDFs + per-engine OCR + annotations) into a
single, model-agnostic, normalized dataset containing:

  * document pages (rendered images + page dimensions)
  * questions
  * ground-truth answers (+ answer variants)
  * OCR text with bounding boxes and page indices
  * answer page indices and answer bounding boxes
  * normalized answer types (extractive / abstractive / list / unanswerable)
  * train / val / test splits
"""

__version__ = "1.0.0"
