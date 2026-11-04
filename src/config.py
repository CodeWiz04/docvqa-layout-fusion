"""Shared settings. Every person imports this file so all loaders use identical values."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ---- paths (see README for the expected folder layout) ----
    data_root: Path = Path("data/docvqa")

    # ---- image settings (paper: longer side -> 1024, pad to square, 16x16 patches) ----
    img_size: int = 1024          # lower to 512 or 384 if GPU memory is tight (patches = (img/16)^2)
    patch_size: int = 16
    img_mean: tuple = (0.5, 0.5, 0.5)   # ViT-Base in21k normalisation
    img_std: tuple = (0.5, 0.5, 0.5)

    # ---- text settings ----
    tokenizer_name: str = "bert-base-uncased"
    max_seq_len: int = 512        # question + OCR tokens + 3 special tokens
    max_answer_len: int = 32      # target length for the generative answer head

    # ---- layout ----
    box_grid: int = 1000          # boxes normalised to a [0, 1000] integer grid

    # ---- misc ----
    seed: int = 42
    num_workers: int = 2
    batch_size: int = 16


CFG = Config()
