"""DocVQA preprocessing: official files -> model-ready tensors (steps P1-P6 of the methodology).

Expected folder layout (official RRC download, one folder per split):

data/docvqa/
    train/  train_v1.0_withQT.json   documents/*.png   ocr_results/*.json
    val/    val_v1.0_withQT.json     documents/*.png   ocr_results/*.json
    test/   test_v1.0.json           documents/*.png   ocr_results/*.json   (no answers)

OCR file for documents/abcd0001_3.png is ocr_results/abcd0001_3.json (Microsoft Read format).
"""
import json
import random
import re
import string
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config import CFG

ANN_NAMES = {
    "train": "train_v1.0_withQT.json",
    "val": "val_v1.0_withQT.json",
    "test": "test_v1.0.json",
}


# --------------------------------------------------------------------------- text helpers
def normalize_text(s: str) -> str:
    """Lower-case, strip punctuation and extra spaces. Used for answer matching."""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- OCR + layout (P2, P3)
def normalize_box(quad, page_w, page_h, grid=CFG.box_grid):
    """8-number quadrilateral [x1,y1,...,x4,y4] -> [xmin,ymin,xmax,ymax] on a 0..grid integer scale (P3)."""
    xs, ys = quad[0::2], quad[1::2]
    box = [min(xs) / page_w, min(ys) / page_h, max(xs) / page_w, max(ys) / page_h]
    return [int(min(max(v, 0.0), 1.0) * grid) for v in box]


def load_ocr(ocr_path: Path):
    """Return (words, boxes) for one page. Boxes are already normalised to 0..1000."""
    with open(ocr_path, encoding="utf-8") as f:
        res = json.load(f)["recognitionResults"][0]
    page_w, page_h = res["width"], res["height"]
    words, boxes = [], []
    for line in res["lines"]:
        for w in line["words"]:
            if not w["text"].strip():
                continue
            words.append(w["text"])
            boxes.append(normalize_box(w["boundingBox"], page_w, page_h))
    return words, boxes


# --------------------------------------------------------------------------- answer span (P4)
def find_answer_span(words, answers, max_ngram=12):
    """Find the first OCR word span whose normalised text equals any ground-truth answer.

    Returns (word_start, word_end) inclusive, or (-1, -1) when the answer is not an exact OCR span.
    """
    targets = {normalize_text(a) for a in answers if a.strip()}
    norm_words = [normalize_text(w) for w in words]
    for n in range(1, max_ngram + 1):
        for i in range(len(words) - n + 1):
            if " ".join(norm_words[i:i + n]).strip() in targets:
                return i, i + n - 1
    return -1, -1


# --------------------------------------------------------------------------- image (P1, P6)
def build_image_transform(train: bool):
    """Resize longer side to img_size, pad to a square canvas, normalise. Light augmentation only when training."""
    size = CFG.img_size

    def resize_pad(img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = size / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(img, (0, 0))  # top-left anchor keeps coordinates easy to reason about
        return canvas

    class GaussianNoise:
        def __call__(self, x):
            return (x + torch.randn_like(x) * 0.02).clamp(0, 1)

    steps = [T.Lambda(resize_pad)]
    if train:
        steps += [
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomRotation(2, fill=255),
            T.ToTensor(),
            T.RandomApply([GaussianNoise()], p=0.3),
        ]
    else:
        steps += [T.ToTensor()]
    steps += [T.Normalize(CFG.img_mean, CFG.img_std)]
    return T.Compose(steps)


# --------------------------------------------------------------------------- dataset
class DocVQADataset(Dataset):
    def __init__(self, split: str, tokenizer, root: Path = None, max_samples: int = None):
        assert split in ANN_NAMES, f"split must be one of {list(ANN_NAMES)}"
        self.split = split
        self.root = Path(root or CFG.data_root) / split
        self.tok = tokenizer
        self.train = split == "train"
        self.transform = build_image_transform(self.train)

        with open(self.root / ANN_NAMES[split], encoding="utf-8") as f:
            self.items = json.load(f)["data"]
        if max_samples:  # handy for quick debugging
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        stem = Path(item["image"]).stem
        image = Image.open(self.root / item["image"]).convert("RGB")
        words, boxes = load_ocr(self.root / "ocr_results" / f"{stem}.json")
        answers = item.get("answers", [])  # test split has none

        enc = self._encode(item["question"], words, boxes)
        w_start, w_end = find_answer_span(words, answers) if answers else (-1, -1)
        start_tok, end_tok = self._word_span_to_token_span(enc["word_ids"], w_start, w_end)

        # target sequence for the generative answer head (first answer is the training target)
        target = self.tok(answers[0] if answers else "", add_special_tokens=True,
                          truncation=True, max_length=CFG.max_answer_len,
                          padding="max_length", return_tensors="pt")["input_ids"][0]
        labels = target.clone()
        labels[labels == self.tok.pad_token_id] = -100  # ignored by cross-entropy

        return {
            "pixel_values": self.transform(image),                  # (3, S, S)
            "input_ids": enc["input_ids"],                          # (L,)
            "attention_mask": enc["attention_mask"],                # (L,)
            "bbox": enc["bbox"],                                    # (L, 4) in 0..1000
            "labels": labels,                                       # (A,)
            "start_positions": torch.tensor(start_tok),             # -1 if not extractive
            "end_positions": torch.tensor(end_tok),
            "question_id": item["questionId"],
            "question": item["question"],
            "answers": answers,
            "image_path": str(self.root / item["image"]),
        }

    # ---- tokenisation: [CLS] question [SEP] OCR words [SEP]; each sub-word inherits its word's box
    def _encode(self, question, words, boxes):
        tok, L = self.tok, CFG.max_seq_len
        q_ids = tok(question, add_special_tokens=False, truncation=True, max_length=64)["input_ids"]

        ocr = tok(words, is_split_into_words=True, add_special_tokens=False)
        ocr_ids, word_ids = ocr["input_ids"], ocr.word_ids()

        budget = L - len(q_ids) - 3                      # room left for OCR tokens
        ocr_ids, word_ids = ocr_ids[:budget], word_ids[:budget]

        input_ids = [tok.cls_token_id] + q_ids + [tok.sep_token_id] + ocr_ids + [tok.sep_token_id]
        bbox = [[0, 0, 0, 0]] * (len(q_ids) + 2) + [boxes[w] for w in word_ids] + [[1000, 1000, 1000, 1000]]
        # word index for every position (-1 for special/question tokens) - used for span mapping
        full_word_ids = [-1] * (len(q_ids) + 2) + list(word_ids) + [-1]

        pad = L - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad
        input_ids += [tok.pad_token_id] * pad
        bbox += [[0, 0, 0, 0]] * pad
        full_word_ids += [-1] * pad

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "bbox": torch.tensor(bbox),
            "word_ids": full_word_ids,
        }

    @staticmethod
    def _word_span_to_token_span(word_ids, w_start, w_end):
        """Convert a word-level answer span into token positions; (-1,-1) if absent or truncated away."""
        if w_start < 0:
            return -1, -1
        pos = [i for i, w in enumerate(word_ids) if w_start <= w <= w_end]
        return (pos[0], pos[-1]) if pos else (-1, -1)


# --------------------------------------------------------------------------- loaders
def collate_fn(batch):
    """Stack tensors; keep strings / answer lists as plain python lists (needed later for ANLS)."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def make_loader(dataset, shuffle=False, batch_size=None):
    return DataLoader(dataset, batch_size=batch_size or CFG.batch_size, shuffle=shuffle,
                      num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)


def set_seed(seed=CFG.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
