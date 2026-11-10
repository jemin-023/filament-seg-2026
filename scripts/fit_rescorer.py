"""Fit the instance rescorer on cached validation probabilities and tune its keep-threshold tau for PQ.

Out-of-fold scores (GroupKFold by date inside the validation fold) are used to pick tau honestly; the final
model is refit on all validation instances and saved to weights/.

python scripts/fit_rescorer.py --name effv2s_f0 --fold 0 --pp cache/probs/effv2s_f0/_tuning/best.json --min-area 50
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import CACHE, TRAIN_IMG  # noqa: E402
from filseg.evaluate import load_gt, load_prob, prob_dir, score_labels  # noqa: E402
from filseg.postprocess import PPParams, full  # noqa: E402
from filseg.rescore import FEATURES, filter_label, instance_features, match_rate  # noqa: E402

GT, ARGS = {}, {}


def work(stem):
    prob = load_prob(prob_dir(ARGS["name"]) / f"{stem}.png")
    img = cv2.imread(str(TRAIN_IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    label = full(prob, PPParams(**ARGS["pp"]))
    return stem, label, instance_features(label, prob, img), match_rate(label, GT[stem])


def make_model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pp", required=True)
    ap.add_argument("--min-area", type=int, default=None, help="override pp min_area (let the rescorer filter)")
    ap.add_argument("--t-high", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "weights" / "rescorer.pkl"))
    a = ap.parse_args()
    pp = json.loads(Path(a.pp).read_text())["params"]
    if a.min_area is not None:
        pp["min_area"] = a.min_area
    if a.t_high is not None:
        pp["t_high"] = a.t_high
    ARGS.update(name=a.name, pp=pp)
    f = pd.read_csv(CACHE / "folds.csv")
    stems = f[f.fold == a.fold].stem.tolist()
    GT.update(load_gt(stems))
    with Pool(a.jobs) as pool:
        res = pool.map(work, stems, chunksize=2)
    labels = {s: lab for s, lab, _, _ in res}
    X = np.concatenate([x for _, _, x, _ in res])
    y = np.concatenate([m for _, _, _, m in res])
    groups = np.concatenate([[s[:8]] * len(m) for s, _, _, m in res])
    print(f"pp {pp}\n{len(X)} instances, mean match rate {y.mean():.3f}")
    base = score_labels(labels, GT)
    print("no rescoring:", base.summary())

    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, groups):
        oof[va] = make_model().fit(X[tr], y[tr]).predict(X[va])
    offs = np.cumsum([0] + [len(m) for _, _, _, m in res])
    rows = []
    for tau in np.arange(0.0, 0.71, 0.05):
        filt = {s: filter_label(labels[s], oof[offs[i]:offs[i + 1]], tau) for i, s in enumerate(stems)}
        sc = score_labels(filt, GT)
        rows.append(dict(tau=round(tau, 2), **sc.summary()))
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["pq"])
    print("best:", best)
    model = make_model().fit(X, y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"model": model, "features": FEATURES, "tau": best["tau"], "pp": pp}, fh)
    tune = prob_dir(a.name) / "_tuning"
    tune.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(tune / "rescorer_tau.csv", index=False)
    (tune / "rescorer.json").write_text(json.dumps({"pp": pp, "best": best, "baseline": base.summary()}, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
