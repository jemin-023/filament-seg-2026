"""Exact re-implementation of the organizers' self-evaluation scorer, made fast.

Official logic (self-evaluation-notebook): for every GT annotator-image entry, build the IoU matrix between its
GT filaments and the predictions for that image; hit = IoU > 0.5; TP IoUs summed, FP = predictions hit by no GT,
FN = GT hitting no prediction; PQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN), aggregated over ALL entries.
We compute intersections on bbox crops instead of dense (n, H*W) matmuls; results are identical.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Inst:
    """Sparse binary mask: bbox (y0, x0, y1, x1) + cropped boolean array."""
    y0: int
    x0: int
    crop: np.ndarray

    @property
    def area(self):
        return int(self.crop.sum())

    @staticmethod
    def from_mask(mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return Inst(int(y0), int(x0), mask[y0:y1, x0:x1].astype(bool))


def _inter(a: Inst, b: Inst) -> int:
    ay1, ax1 = a.y0 + a.crop.shape[0], a.x0 + a.crop.shape[1]
    by1, bx1 = b.y0 + b.crop.shape[0], b.x0 + b.crop.shape[1]
    y0, x0, y1, x1 = max(a.y0, b.y0), max(a.x0, b.x0), min(ay1, by1), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0
    ca = a.crop[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
    cb = b.crop[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]
    return int(np.count_nonzero(ca & cb))


def overlap(gt: list, pred: list):
    """IoU and Dice matrices (n_gt, n_pred), same definitions as the official get_overlap_matrices."""
    iou = np.zeros((len(gt), len(pred)))
    dice = np.zeros_like(iou)
    ga = [g.area for g in gt]
    pa = [p.area for p in pred]
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            inter = _inter(g, p)
            if inter:
                iou[i, j] = inter / (ga[i] + pa[j] - inter)
                dice[i, j] = 2 * inter / (ga[i] + pa[j])
    return iou, dice


@dataclass
class Score:
    tp_ious: list = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    dice_vals: list = field(default_factory=list)     # nonzero dice pairs (official distribution plot)
    gt_deg: list = field(default_factory=list)        # per GT: number of overlapping preds (IoU>0)
    pred_deg: list = field(default_factory=list)      # per pred: number of overlapping GT
    per_entry: list = field(default_factory=list)     # (entry_id, sum_iou, tp, fp, fn)

    def add(self, entry_id: str, gt: list, pred: list):
        n_gt, n_pred = len(gt), len(pred)
        if n_gt == 0:
            self.fp += n_pred
            self.pred_deg += [0] * n_pred
            self.per_entry.append((entry_id, 0.0, 0, n_pred, 0))
            return
        if n_pred == 0:
            self.fn += n_gt
            self.gt_deg += [0] * n_gt
            self.per_entry.append((entry_id, 0.0, 0, 0, n_gt))
            return
        iou, dice = overlap(gt, pred)
        hit = iou > 0.5
        tps = iou[hit].tolist()
        fp = int((hit.sum(0) == 0).sum())
        fn = int((hit.sum(1) == 0).sum())
        self.tp_ious += tps
        self.fp += fp
        self.fn += fn
        self.dice_vals += dice[dice > 0].tolist()
        self.gt_deg += (iou > 0).sum(1).tolist()
        self.pred_deg += (iou > 0).sum(0).tolist()
        self.per_entry.append((entry_id, float(sum(tps)), len(tps), fp, fn))

    def merge(self, other: "Score"):
        self.tp_ious += other.tp_ious
        self.fp += other.fp
        self.fn += other.fn
        self.dice_vals += other.dice_vals
        self.gt_deg += other.gt_deg
        self.pred_deg += other.pred_deg
        self.per_entry += other.per_entry
        return self

    @property
    def tp(self):
        return len(self.tp_ious)

    @property
    def pq(self):
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return sum(self.tp_ious) / den if den > 0 else 0.0

    @property
    def sq(self):
        return float(np.mean(self.tp_ious)) if self.tp_ious else 0.0

    @property
    def rq(self):
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / den if den > 0 else 0.0

    def summary(self) -> dict:
        g = np.array(self.gt_deg)
        p = np.array(self.pred_deg)
        return dict(
            pq=round(self.pq, 4), sq=round(self.sq, 4), rq=round(self.rq, 4),
            tp=self.tp, fp=self.fp, fn=self.fn,
            mean_dice_nonzero=round(float(np.mean(self.dice_vals)), 4) if self.dice_vals else 0.0,
            one_to_many=int((g > 1).sum()), many_to_one=int((p > 1).sum()),
        )


def masks_to_insts(masks) -> list:
    out = []
    for m in masks:
        inst = Inst.from_mask(m)
        if inst is not None:
            out.append(inst)
    return out


def label_to_insts(label: np.ndarray) -> list:
    """Integer label map (0 = background) -> list of Inst (disjoint by construction)."""
    from scipy import ndimage as ndi
    out = []
    for k, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            continue
        crop = label[sl] == k
        out.append(Inst(sl[0].start, sl[1].start, crop))
    return out


def gt_insts(anns: list) -> list:
    """GT annotations (COCO polygons) -> Inst list. Empty rasterizations are kept (they count as FN officially)."""
    from .rle import polygon_to_mask
    out = []
    for a in anns:
        inst = Inst.from_mask(polygon_to_mask(a["segmentation"]))
        out.append(inst if inst is not None else Inst(0, 0, np.zeros((1, 1), bool)))
    return out
