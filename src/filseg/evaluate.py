"""Validation helpers: cached probabilities, GT loading, scoring post-processed predictions, human reference."""
import itertools
from pathlib import Path

import cv2
import numpy as np

from .data import CACHE, load_index
from .metrics import Score, gt_insts, label_to_insts


def prob_dir(name: str) -> Path:
    return CACHE / "probs" / name


def save_prob(path: Path, prob: np.ndarray):
    """(3, H, W) float in [0, 1] -> uint8 PNG (lossless, 1/255 quantization)."""
    img = np.round(prob.astype(np.float32) * 255).astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite(str(path), img[..., ::-1])


def load_prob(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))[..., ::-1]
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def load_gt(stems):
    idx = load_index()
    return {s: [(e.entry_id, gt_insts(e.anns)) for e in idx[s]] for s in stems}


def score_labels(labels: dict, gt: dict) -> Score:
    sc = Score()
    for s, lab in labels.items():
        pred = label_to_insts(lab)
        for eid, g in gt[s]:
            sc.add(eid, g, pred)
    return sc


def human_reference(gt: dict) -> Score:
    """Each annotator used as the 'prediction' against every other annotator of the same image."""
    sc = Score()
    for s, entries in gt.items():
        for (ea, ga), (eb, gb) in itertools.permutations(entries, 2):
            sc.add(eb, gb, ga)
    return sc


def pq_sparse(s: dict, r: float = 0.8) -> float:
    """Expected PQ if each GT filament survives in the (test) labels with prob r (public-notebook sparse-label model).
    s: Score.summary() dict with sq, tp, fp, fn."""
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    num = r * s["sq"] * tp
    den = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return num / den if den else 0.0
