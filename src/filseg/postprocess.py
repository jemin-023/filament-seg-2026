"""Probability maps -> disjoint filament instances (integer label map)."""
import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)


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
