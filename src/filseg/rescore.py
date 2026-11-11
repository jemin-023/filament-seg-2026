"""Instance rescoring: predict P(instance matches an annotator at IoU > 0.5) from cheap per-instance features.

PQ algebra: keeping an instance with match probability p and expected IoU q helps iff p*q > PQ/2, so a calibrated
p lets us drop faint/false candidates that the threshold alone cannot separate.
"""
import numpy as np
from scipy import ndimage as ndi

from .metrics import label_to_insts, overlap

FEATURES = ["log_area", "p_mean", "p_max", "p_q90", "s_mean", "s_max", "elong", "contrast", "r_norm", "n_px_hi"]


def disk_geometry(img: np.ndarray):
    on = img > 20
    ys, xs = np.nonzero(on[::4, ::4])
    cy, cx = ys.mean() * 4, xs.mean() * 4
    r = np.sqrt(on.sum() / np.pi)
    return cy, cx, r


def instance_features(label: np.ndarray, prob: np.ndarray, img: np.ndarray) -> np.ndarray:
    """(n_inst, len(FEATURES)) in label order 1..n."""
    P = prob[0].astype(np.float32)
    S = prob[1].astype(np.float32)
    if P.max() > 1.5:   # uint8 input
        P, S = P / 255.0, S / 255.0
    cy, cx, R = disk_geometry(img)
    rows = []
    for k, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            rows.append([0.0] * len(FEATURES))
            continue
        # pad the slice for the background ring
        y0, y1 = max(sl[0].start - 15, 0), min(sl[0].stop + 15, label.shape[0])
        x0, x1 = max(sl[1].start - 15, 0), min(sl[1].stop + 15, label.shape[1])
        lab = label[y0:y1, x0:x1]
        m = lab == k
        p, s, im = P[y0:y1, x0:x1][m], S[y0:y1, x0:x1][m], img[y0:y1, x0:x1].astype(np.float32)
        ring = ndi.binary_dilation(m, iterations=10) & (lab == 0)
        contrast = (im[ring].mean() - im[m].mean()) if ring.any() else 0.0
        yy, xx = np.nonzero(m)
        if len(yy) > 2:
            ev = np.linalg.eigvalsh(np.cov(np.stack([yy, xx])))
            elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        r = np.hypot(yy.mean() + y0 - cy, xx.mean() + x0 - cx) / R
        rows.append([np.log(m.sum()), p.mean(), p.max(), np.quantile(p, 0.9), s.mean(), s.max(), elong, contrast, r,
                     np.log1p((p > 0.7).sum())])
    return np.asarray(rows, np.float32).reshape(-1, len(FEATURES))


def match_rate(label: np.ndarray, gt_entries: list) -> np.ndarray:
    """Fraction of the image's annotators whose GT matches each predicted instance (IoU > 0.5)."""
    pred = label_to_insts(label)
    hits = np.zeros(len(pred))
    for _, g in gt_entries:
        if len(g) and len(pred):
            iou, _ = overlap(g, pred)
            hits += (iou > 0.5).any(0)
    return hits / max(len(gt_entries), 1)


def filter_label(label: np.ndarray, scores: np.ndarray, tau: float) -> np.ndarray:
    keep = np.concatenate([[False], scores >= tau])
    lut = np.zeros(len(keep), np.int32)
    lut[keep] = np.arange(1, keep.sum() + 1)
    return lut[label]
