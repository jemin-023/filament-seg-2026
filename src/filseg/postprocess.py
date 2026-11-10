"""Probability maps -> disjoint filament instances (integer label map).

prob channels: 0 semantic filament probability, 1 spine, 2 boundary.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def remove_small(label: np.ndarray, min_area: int) -> np.ndarray:
    areas = np.bincount(label.ravel())
    keep = areas >= min_area
    keep[0] = False
    lut = np.zeros(len(areas), np.int32)
    lut[keep] = np.arange(1, keep.sum() + 1)
    return lut[label]


def simple(prob: np.ndarray, thr=0.5, min_area=100) -> np.ndarray:
    """v0: threshold semantic probability, 8-connected components, drop small ones."""
    label, _ = ndi.label(prob[0] > thr, structure=EIGHT)
    return remove_small(label, min_area)


@dataclass
class PPParams:
    t_low: float = 0.3      # connectivity threshold (groups fragments)
    t_high: float = 0.5     # a group must contain a pixel above this
    t_shape: float = 0.45   # pixels kept inside a group
    r_close: int = 3        # closing radius applied to the low mask before grouping
    dilate: int = 0         # final per-instance thickness calibration (px, negative = erode)
    fill: bool = True       # fill holes inside each instance
    min_area: int = 150     # drop instances smaller than this (px)
    min_peak: float = 0.0   # drop instances whose max probability is below this

    def dict(self):
        return asdict(self)


def _dilate_labels(label: np.ndarray, d: int) -> np.ndarray:
    if d == 0:
        return label
    if d > 0:
        grown = ndi.grey_dilation(label, footprint=disk(d))
        return np.where(label > 0, label, grown)          # grow only into background: stays disjoint
    keep = ndi.binary_erosion(label > 0, structure=disk(-d))
    # erosion of the union also separates touching instances' borders; fine (instances rarely touch)
    return np.where(keep, label, 0)


def _fill_holes(label: np.ndarray) -> np.ndarray:
    out = label.copy()
    for k, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            continue
        m = label[sl] == k
        filled = ndi.binary_fill_holes(m)
        hole = filled & ~m & (label[sl] == 0)
        if hole.any():
            out[sl][hole] = k
    return out


def full(prob: np.ndarray, p: PPParams) -> np.ndarray:
    P = prob[0]
    if P.dtype == np.uint8:
        P = P.astype(np.float32) / 255.0
    low = P > p.t_low
    if p.r_close > 0:
        low = ndi.binary_closing(low, structure=disk(p.r_close))
    groups, n = ndi.label(low, structure=EIGHT)
    if n == 0:
        return groups
    ok = np.zeros(n + 1, bool)
    ok[np.unique(groups[P > max(p.t_high, p.min_peak)])] = True   # groups containing a confident seed
    ok[0] = False
    lut = np.zeros(n + 1, np.int32)
    lut[ok] = np.arange(1, ok.sum() + 1)
    groups = lut[groups]
    label = np.where(P > p.t_shape, groups, 0)
    if p.fill:
        label = _fill_holes(label)
    label = _dilate_labels(label, p.dilate)
    return remove_small(label, p.min_area)
