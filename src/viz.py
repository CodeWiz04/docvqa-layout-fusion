"""Helpers for checking and visualising the preprocessing output."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from config import CFG


def denormalize(pixel_values):
    """Undo the ViT normalisation so a tensor can be shown as an image."""
    mean = np.array(CFG.img_mean)[:, None, None]
    std = np.array(CFG.img_std)[:, None, None]
    img = pixel_values.numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def show_sample(sample, tokenizer, max_boxes=400):
    """Left: raw page. Right: preprocessed page with OCR boxes. Prints question, answer and extracted span."""
    raw = Image.open(sample["image_path"]).convert("RGB")
    prep = denormalize(sample["pixel_values"])
    S = prep.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    ax[0].imshow(raw); ax[0].set_title("Raw page"); ax[0].axis("off")
    ax[1].imshow(prep); ax[1].set_title(f"Preprocessed ({S}x{S}) + OCR boxes"); ax[1].axis("off")

    # boxes are in the ORIGINAL page's 0..1000 space; the page is scaled by one factor and pinned top-left
    w, h = raw.size
    scale = S / max(w, h)
    for box, m in list(zip(sample["bbox"].tolist(), sample["attention_mask"].tolist()))[:max_boxes]:
        if not m or box == [0, 0, 0, 0] or box == [1000, 1000, 1000, 1000]:
            continue
        x0, y0, x1, y1 = box[0] / 1000 * w * scale, box[1] / 1000 * h * scale, \
                         box[2] / 1000 * w * scale, box[3] / 1000 * h * scale
        ax[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=0.6))
    plt.tight_layout()
    plt.show()

    print("Question  :", sample["question"])
    print("Answers   :", sample["answers"])
    s, e = int(sample["start_positions"]), int(sample["end_positions"])
    if s >= 0:
        print("Span found:", tokenizer.decode(sample["input_ids"][s:e + 1]))
    else:
        print("Span found: none (answer is not an exact OCR span - generative head handles this)")


def dataset_stats(dataset, n=500):
    """Quick numbers for the report's dataset table (estimated from the first n samples)."""
    n = min(n, len(dataset))
    ocr_tokens, extractive = [], 0
    for i in range(n):
        s = dataset[i]
        ocr_tokens.append(int(s["attention_mask"].sum()))
        extractive += int(s["start_positions"]) >= 0
    return {
        "questions": len(dataset),
        "avg_tokens_per_sample": float(np.mean(ocr_tokens)),
        "pct_truncated_at_max_len": 100 * float(np.mean([t >= CFG.max_seq_len for t in ocr_tokens])),
        "pct_extractive_answers": 100 * extractive / n,
    }
