"""End-to-end inference pipeline used by both the submission script and the notebook.

image (2048^2 uint8) -> U-Net ensemble + flip TTA -> probabilities -> instance post-processing
-> instance rescoring filter -> disjoint filament masks -> RLE rows.
"""
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .infer import predict_probs
from .model import load_model
from .postprocess import PPParams, full
from .rescore import filter_label, instance_features
from .rle import encode


@dataclass
class Pipeline:
    ckpts: list
    pp: PPParams
    rescorer: dict | None = None
    tta: tuple = ("id", "h")
    device: str = "cuda"
    timings: dict = field(default_factory=lambda: {"model": 0.0, "post": 0.0, "rescore": 0.0, "n": 0})

    def __post_init__(self):
        self.models = [load_model(c, self.device) for c in self.ckpts]

    @classmethod
    def from_files(cls, ckpts, pp_json=None, rescorer_pkl=None, tta=("id", "h"), device="cuda"):
        rescorer = None
        if rescorer_pkl and Path(rescorer_pkl).exists():
            with open(rescorer_pkl, "rb") as fh:
                rescorer = pickle.load(fh)
        if rescorer is not None and pp_json is None:
            pp = PPParams(**rescorer["pp"])          # rescorer was fit on instances from these exact params
        else:
            pp = PPParams(**json.loads(Path(pp_json).read_text())["params"])
        return cls(list(ckpts), pp, rescorer, tuple(tta), device)

    def probs(self, img: np.ndarray) -> np.ndarray:
        t = time.time()
        prob = predict_probs(self.models, img, tta=self.tta, device=self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.timings["model"] += time.time() - t
        return prob.astype(np.float32)

    def instances(self, img: np.ndarray, prob: np.ndarray | None = None) -> np.ndarray:
        """Returns an integer label map (0 = background, 1..n = filaments)."""
        prob = self.probs(img) if prob is None else prob
        t = time.time()
        label = full(prob, self.pp)
        self.timings["post"] += time.time() - t
        if self.rescorer is not None and label.max() > 0:
            t = time.time()
            scores = self.rescorer["model"].predict(instance_features(label, prob, img))
            label = filter_label(label, scores, self.rescorer["tau"])
            self.timings["rescore"] += time.time() - t
        self.timings["n"] += 1
        return label

    @staticmethod
    def rows(stem: str, label: np.ndarray) -> list:
        out = []
        for k in range(1, int(label.max()) + 1):
            m = label == k
            if m.any():
                out.append((f"{stem}_{len(out)}", encode(m)))
        return out

    def timing_summary(self) -> dict:
        n = max(self.timings["n"], 1)
        return {k: round(v / n, 3) for k, v in self.timings.items() if k != "n"}
