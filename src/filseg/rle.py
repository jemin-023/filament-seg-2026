"""COCO compressed-RLE helpers matching the competition submission format (counts string only, 2048x2048)."""
import numpy as np
from pycocotools import mask as mask_util

SIZE = 2048


def encode(mask: np.ndarray) -> str:
    """Binary (H, W) mask -> compressed RLE counts string."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode(counts: str, h: int = SIZE, w: int = SIZE) -> np.ndarray:
    """Compressed RLE counts string -> uint8 (H, W) mask."""
    return mask_util.decode({"size": [h, w], "counts": counts.encode("ascii")})


def polygon_to_mask(polygon: list, h: int = SIZE, w: int = SIZE) -> np.ndarray:
    """COCO polygon list ([[x0, y0, ...]]) -> uint8 mask, rasterized exactly like pycocotools does for GT."""
    rles = mask_util.frPyObjects(polygon, h, w)
    return mask_util.decode(mask_util.merge(rles))


def polygon_to_rle(polygon: list, h: int = SIZE, w: int = SIZE) -> str:
    rles = mask_util.frPyObjects(polygon, h, w)
    return mask_util.merge(rles)["counts"].decode("ascii")
