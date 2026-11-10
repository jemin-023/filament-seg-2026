"""Run model(s) on validation or test images and cache uint8 probability PNGs for post-processing tuning.

python scripts/cache_probs.py --ckpt runs/effv2s_c768_f0/best.pt --name effv2s_f0 --split val --fold 0 --tta id h
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import CACHE, TEST_IMG, TRAIN_IMG  # noqa: E402
from filseg.evaluate import prob_dir, save_prob  # noqa: E402
from filseg.infer import predict_probs  # noqa: E402
from filseg.model import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", nargs="+", default=["id"])
    a = ap.parse_args()
    if a.split == "val":
        f = pd.read_csv(CACHE / "folds.csv")
        stems, img_dir = f[f.fold == a.fold].stem.tolist(), TRAIN_IMG
    else:
        stems, img_dir = sorted(p.stem for p in TEST_IMG.glob("*.jpeg")), TEST_IMG
    out = prob_dir(a.name)
    out.mkdir(parents=True, exist_ok=True)
    models = [load_model(c) for c in a.ckpt]
    t_model = 0.0
    for s in stems:
        img = cv2.imread(str(img_dir / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        torch.cuda.synchronize()
        t = time.time()
        prob = predict_probs(models, img, tta=tuple(a.tta))
        torch.cuda.synchronize()
        t_model += time.time() - t
        save_prob(out / f"{s}.png", prob)
    print(f"{len(stems)} images, model time {t_model / len(stems):.3f} s/img ({len(models)} models x {len(a.tta)} TTA)")


if __name__ == "__main__":
    main()
