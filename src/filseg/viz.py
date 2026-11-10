"""Figures for the notebook and report: overlays, IoU/Dice distributions, one-to-many / many-to-one counts."""
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .metrics import Score


def overlay(img: np.ndarray, label: np.ndarray, gt_masks=None, crop=None, alpha=0.45) -> np.ndarray:
    """RGB overlay: predicted instances in distinct colours (filled), GT outlines in white."""
    rgb = np.dstack([img] * 3).astype(np.float32)
    rng = np.random.default_rng(0)
    cols = rng.uniform(60, 255, (int(label.max()) + 1, 3))
    m = label > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * cols[label[m]]
    out = rgb.astype(np.uint8)
    if gt_masks is not None:
        for g in gt_masks:
            cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, (255, 255, 255), 1)
    if crop is not None:
        y0, x0, h, w = crop
        out = out[y0:y0 + h, x0:x0 + w]
    return out


def plot_distributions(score: Score, human: Score | None = None, path=None):
    """Histograms of matched-pair IoU (TP) and nonzero-pair Dice; optional human reference."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(score.tp_ious, bins=bins, alpha=0.7, density=True, label="model")
    ax[1].hist(score.dice_vals, bins=bins, alpha=0.7, density=True, label="model")
    if human is not None:
        ax[0].hist(human.tp_ious, bins=bins, alpha=0.5, density=True, label="annotator vs annotator")
        ax[1].hist(human.dice_vals, bins=bins, alpha=0.5, density=True, label="annotator vs annotator")
    ax[0].set(title="IoU of matched pairs (IoU>0.5)", xlabel="IoU")
    ax[1].set(title="Dice of overlapping pairs", xlabel="Dice")
    g, p = Counter(score.gt_deg), Counter(score.pred_deg)
    ks = range(0, 5)
    ax[2].bar([k - 0.2 for k in ks], [g.get(k, 0) if k < 4 else sum(v for kk, v in g.items() if kk >= 4) for k in ks],
              0.4, label="GT -> #preds (1:n)")
    ax[2].bar([k + 0.2 for k in ks], [p.get(k, 0) if k < 4 else sum(v for kk, v in p.items() if kk >= 4) for k in ks],
              0.4, label="pred -> #GT (n:1)")
    ax[2].set(title="Overlap degree (IoU>0)", xticks=list(ks), xticklabels=["0", "1", "2", "3", "4+"])
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
