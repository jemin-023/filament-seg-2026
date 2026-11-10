"""End-to-end test inference: models -> probabilities -> instances -> submission CSV (+ timing).

python scripts/make_submission.py --ckpt runs/a/best.pt runs/b/best.pt --pp cache/probs/x/_tuning/best.json \
       --tta id h --out submissions/sub.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import TEST_IMG  # noqa: E402
from filseg.infer import predict_probs  # noqa: E402
from filseg.model import load_model  # noqa: E402
from filseg.postprocess import PPParams, full  # noqa: E402
from filseg.rle import decode, encode  # noqa: E402


def label_to_rows(stem, label):
    rows = []
    for k in range(1, int(label.max()) + 1):
        m = label == k
        if m.any():
            rows.append((f"{stem}_{len(rows)}", encode(m)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--pp", required=True, help="json with {'params': {...}}")
    ap.add_argument("--tta", nargs="+", default=["id"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", default=str(TEST_IMG))
    a = ap.parse_args()
    pp = PPParams(**json.loads(Path(a.pp).read_text())["params"])
    models = [load_model(c) for c in a.ckpt]
    stems = sorted(p.stem for p in Path(a.img_dir).glob("*.jpeg"))
    rows, t = [], {"read": 0.0, "model": 0.0, "post": 0.0, "rle": 0.0}
    t_all = time.time()
    for s in stems:
        t0 = time.time()
        img = cv2.imread(str(Path(a.img_dir) / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        t1 = time.time()
        prob = predict_probs(models, img, tta=tuple(a.tta))
        torch.cuda.synchronize()
        t2 = time.time()
        label = full(prob.astype(np.float32), pp)
        t3 = time.time()
        rows += label_to_rows(s, label)
        t4 = time.time()
        for k, v in zip(t, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            t[k] += v
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    n = len(stems)
    print(f"{n} images, {len(df)} instances ({len(df) / n:.1f}/img), images without predictions: "
          f"{n - df.filament_id.str.rsplit('_', n=1).str[0].nunique()}")
    print("seconds/image:", {k: round(v / n, 3) for k, v in t.items()}, f"total {(time.time() - t_all) / n:.3f}")
    # format self-check: every RLE decodes to a non-empty 2048x2048 mask; ids unique; instances disjoint per image
    assert df.filament_id.is_unique
    for s, g in df.groupby(df.filament_id.str.rsplit("_", n=1).str[0]):
        acc = np.zeros((2048, 2048), np.uint8)
        for r in g.segmentation_rle:
            m = decode(r)
            assert m.shape == (2048, 2048) and m.any()
            acc += m
        assert acc.max() <= 1, f"overlapping instances in {s}"
    print("format check OK ->", a.out)


if __name__ == "__main__":
    main()
