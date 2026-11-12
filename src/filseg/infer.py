"""Full-image inference (2048^2, no tiling) with optional flip TTA and multi-model averaging."""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import normalize

AMP = torch.float16 if os.environ.get("FILSEG_AMP") == "fp16" else torch.bfloat16

TTA = {
    "id": (lambda x: x, lambda x: x),
    "h": (lambda x: x.flip(-1), lambda x: x.flip(-1)),
    "v": (lambda x: x.flip(-2), lambda x: x.flip(-2)),
    "hv": (lambda x: x.flip(-1, -2), lambda x: x.flip(-1, -2)),
}


@torch.no_grad()
def predict_logits(model, img_u8: np.ndarray, mode="crop", size=1024, tta=("id",), device="cuda") -> torch.Tensor:
    """Returns mean logits (3, 2048, 2048) on device, float32."""
    x = torch.from_numpy(normalize(img_u8))[None, None].to(device)
    if mode == "full":
        x = F.interpolate(x, size=(size, size), mode="area")
    acc = None
    for t in tta:
        fwd, inv = TTA[t]
        with torch.autocast("cuda", dtype=AMP):
            y = inv(model(fwd(x))).float()
        if mode == "full":
            y = F.interpolate(y, size=img_u8.shape, mode="bilinear", align_corners=False)
        acc = y if acc is None else acc + y
    return acc[0] / len(tta)


@torch.no_grad()
def predict_probs(models, img_u8, tta=("id",), device="cuda") -> np.ndarray:
    """Per model: mean logits over TTA -> sigmoid; then mean probability over models (matches cached-prob ensembles,
    scripts/ens_probs.py) -> (3, H, W) float16 numpy."""
    acc = None
    for m in models:
        pr = torch.sigmoid(predict_logits(m, img_u8, m.cfg.get("mode", "crop"), m.cfg.get("size", 1024), tta, device))
        acc = pr if acc is None else acc + pr
    return (acc / len(models)).half().cpu().numpy()
