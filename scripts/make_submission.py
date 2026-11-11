"""End-to-end test inference -> submission CSV, with timing and a format self-check.

python scripts/make_submission.py --ckpt runs/a/best.pt runs/b/best.pt --rescorer weights/rescorer.pkl \
       --tta id h --out submissions/sub.csv
(or --pp <tuning best.json> without a rescorer)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import TEST_IMG  # noqa: E402
from filseg.evaluate import load_prob, prob_dir  # noqa: E402
from filseg.pipeline import Pipeline  # noqa: E402
from filseg.rle import decode  # noqa: E402


def check(df: pd.DataFrame, n_images: int):
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert df.filament_id.is_unique
    img_of = df.filament_id.str.rsplit("_", n=1).str[0]
    for s, g in df.groupby(img_of):
        acc = np.zeros((2048, 2048), np.uint8)
        for r in g.segmentation_rle:
            m = decode(r)
            assert m.shape == (2048, 2048) and m.any()
            acc += m
        assert acc.max() <= 1, f"overlapping instances in {s}"
    print(f"format check OK: {len(df)} rows, {img_of.nunique()}/{n_images} images have predictions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[])
    ap.add_argument("--probs", default=None, help="use cached test probs cache/probs/<name>/ instead of running models (CPU)")
    ap.add_argument("--pp", default=None)
    ap.add_argument("--rescorer", default=None)
    ap.add_argument("--tta", nargs="+", default=["id", "h"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", default=str(TEST_IMG))
    a = ap.parse_args()
    assert a.ckpt or a.probs, "need --ckpt or --probs"
    pipe = Pipeline.from_files([] if a.probs else a.ckpt, a.pp, a.rescorer, a.tta, device="cpu" if a.probs else "cuda")
    print("post-processing:", pipe.pp.dict(), "| rescorer tau:", pipe.rescorer and pipe.rescorer["tau"])
    stems = sorted(p.stem for p in Path(a.img_dir).glob("*.jpeg"))
    rows, t0 = [], time.time()
    for s in stems:
        img = cv2.imread(str(Path(a.img_dir) / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        prob = load_prob(prob_dir(a.probs) / f"{s}.png") if a.probs else None
        rows += pipe.rows(s, pipe.instances(img, prob))
    total = (time.time() - t0) / len(stems)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(stems)} images, {len(df) / len(stems):.1f} instances/img; seconds/img {pipe.timing_summary()} "
          f"end-to-end {total:.3f}")
    check(df, len(stems))
    print("->", a.out)


if __name__ == "__main__":
    main()
