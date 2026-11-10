"""Precompute training targets (one 3-channel uint8 PNG per image) + folds + crop-sampling metadata.

Channels (each = mean over the image's annotators, scaled to 0..255):
  0 soft filament mask, 1 spine heatmap (5 px polyline, Gaussian sigma 2), 2 inner boundary band (~2 px).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import CACHE, date, instrument, load_index  # noqa: E402
from filseg.rle import polygon_to_mask  # noqa: E402

OUT = CACHE / "targets"
K3 = np.ones((3, 3), np.uint8)


def entry_maps(anns):
    mask = np.zeros((2048, 2048), np.uint8)
    spine = np.zeros((2048, 2048), np.uint8)
    bnd = np.zeros((2048, 2048), np.uint8)
    for a in anns:
        m = polygon_to_mask(a["segmentation"])
        mask |= m
        bnd |= m & (1 - cv2.erode(m, K3, iterations=2))
        s = np.asarray(a["spine"], np.float32).reshape(-1, 2)
        if len(s) >= 2:
            cv2.polylines(spine, [np.round(s).astype(np.int32)], False, 1, thickness=5)
    spine = cv2.GaussianBlur(spine.astype(np.float32), (0, 0), 2.0)
    spine = np.clip(spine / 0.75, 0, 1)   # centre of a 5 px line blurred by sigma 2 ~ 0.75
    return mask.astype(np.float32), spine, bnd.astype(np.float32)


def work(args):
    stem, entries = args
    acc = [np.zeros((2048, 2048), np.float32) for _ in range(3)]
    centers = []
    for e in entries:
        for i, m in enumerate(entry_maps(e.anns)):
            acc[i] += m
        for a in e.anns:
            x, y, w, h = a["bbox"]
            centers.append([x + w / 2, y + h / 2])
    img = np.stack([np.round(255 * a / len(entries)) for a in acc], -1).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{stem}.png"), img[..., ::-1])   # store RGB order: ch0 = mask
    return stem, centers


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index()
    with Pool(6) as p:
        res = dict(p.map(work, list(idx.items()), chunksize=4))
    stems = sorted(idx)
    df = pd.DataFrame({"stem": stems, "date": [date(s) for s in stems], "inst": [instrument(s) for s in stems],
                       "n_annot": [len(idx[s]) for s in stems]})
    df["fold"] = -1
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (_, va) in enumerate(sgk.split(df, df["inst"], groups=df["date"])):
        df.loc[va, "fold"] = f
    df.to_csv(CACHE / "folds.csv", index=False)
    (CACHE / "centers.json").write_text(json.dumps(res))
    print(df.groupby("fold").agg(n=("stem", "size"), dates=("date", "nunique"), annot=("n_annot", "sum")))
    print(pd.crosstab(df.fold, df.inst))
    assert df.groupby("date").fold.nunique().max() == 1


if __name__ == "__main__":
    main()
