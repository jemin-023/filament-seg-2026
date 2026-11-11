"""Average cached probability maps of several models -> new cache dir.

python scripts/ens_probs.py --src effv2s_f0 r34_f0 --out ens2_f0 [--w 0.5 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.evaluate import load_prob, prob_dir, save_prob  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--w", nargs="+", type=float, default=None)
a = ap.parse_args()
w = np.array(a.w or [1.0] * len(a.src)); w = w / w.sum()
out = prob_dir(a.out); out.mkdir(parents=True, exist_ok=True)
for p in sorted(prob_dir(a.src[0]).glob("*.png")):
    acc = sum(wi * load_prob(prob_dir(s) / p.name) for wi, s in zip(w, a.src))
    save_prob(out / p.name, acc)
print("wrote", out)
