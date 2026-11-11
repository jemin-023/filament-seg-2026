"""Honest (cross-fold) evaluation: for each held-out fold k, tune post-processing (Optuna) and fit the rescorer on the
OTHER available folds' out-of-fold probabilities only, then score fold k. Pools the held-out scores.

Prob caches must be OOF: cache/probs/<tmpl.format(k)> holds fold-k val probs from a model that never saw fold k.

python scripts/honest_eval.py --tmpl ens2_f{} --folds 0 1 --trials 40 --jobs 2
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import CACHE, TRAIN_IMG  # noqa: E402
from filseg.evaluate import load_gt, load_prob, prob_dir, pq_sparse  # noqa: E402
from filseg.metrics import Score, label_to_insts  # noqa: E402
from filseg.postprocess import PPParams, full  # noqa: E402
from filseg.rescore import filter_label, instance_features, match_rate  # noqa: E402

SEM, GT, SRC = {}, {}, {}   # stem -> uint8 semantic prob; stem -> gt entries; stem -> prob cache name


def _score(args):
    s, p = args
    pred = label_to_insts(full(SEM[s], PPParams(**p)))
    sc = Score()
    for eid, g in GT[s]:
        sc.add(eid, g, pred)
    return sc


def evaluate(pool, stems, p):
    tot = Score()
    for sc in pool.imap_unordered(_score, [(s, p) for s in stems], chunksize=4):
        tot.merge(sc)
    return tot


def _inst(args):
    s, p = args
    prob = load_prob(prob_dir(SRC[s]) / f"{s}.png")
    img = cv2.imread(str(TRAIN_IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
    lab = full(prob, PPParams(**p))
    return s, label_to_insts(lab), instance_features(lab, prob, img), match_rate(lab, GT[s])


def make_model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


def tune(pool, stems, trials, seed=0):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        p = dict(t_low=t.suggest_float("t_low", 0.15, 0.5, step=0.05),
                 t_high=t.suggest_float("t_high", 0.4, 0.8, step=0.05),
                 t_shape=t.suggest_float("t_shape", 0.35, 0.65, step=0.05),
                 r_close=t.suggest_int("r_close", 0, 7), dilate=t.suggest_int("dilate", -1, 1),
                 fill=t.suggest_categorical("fill", [True, False]),
                 min_area=t.suggest_int("min_area", 100, 500, step=50))
        return evaluate(pool, stems, p).pq

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    # warm starts: the two configs tuned in-sample on fold 0 so far (only used as starting points)
    st.enqueue_trial(dict(t_low=0.15, t_high=0.4, t_shape=0.65, r_close=6, dilate=0, fill=True, min_area=350))
    st.enqueue_trial(dict(t_low=0.3, t_high=0.6, t_shape=0.5, r_close=2, dilate=0, fill=True, min_area=350))
    st.optimize(obj, n_trials=trials)
    return st.best_params, st.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmpl", required=True, help="prob cache name template, e.g. ens2_f{}")
    ap.add_argument("--folds", nargs="+", type=int, required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--resc-min-area", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    f = pd.read_csv(CACHE / "folds.csv")
    by_fold = {k: f[f.fold == k].stem.tolist() for k in a.folds}
    for k, stems in by_fold.items():
        for s in stems:
            SRC[s] = a.tmpl.format(k)
            SEM[s] = np.round(load_prob(prob_dir(SRC[s]) / f"{s}.png")[:1] * 255).astype(np.uint8)
        GT.update(load_gt(stems))
    pooled = {"pp": Score(), "resc": Score()}
    report = {}
    with Pool(a.jobs) as pool:
        for k in a.folds:
            t0 = time.time()
            tr = [s for j in a.folds if j != k for s in by_fold[j]]
            te = by_fold[k]
            pp, in_pq = tune(pool, tr, a.trials)
            sc_pp = evaluate(pool, te, pp)
            # rescorer: instances from the tuned pp but a low min_area; fit + pick tau on train folds only
            rp = dict(pp, min_area=a.resc_min_area)
            res_tr = pool.map(_inst, [(s, rp) for s in tr], chunksize=2)
            X = np.concatenate([r[2] for r in res_tr]); y = np.concatenate([r[3] for r in res_tr])
            # tau: OOF within the train folds (by fold) so the threshold is not chosen on in-sample predictions
            fold_of = np.concatenate([[f.set_index("stem").fold[r[0]]] * len(r[3]) for r in res_tr])
            oof = np.zeros(len(y))
            for j in set(fold_of):
                m = fold_of == j
                oof[m] = make_model().fit(X[~m], y[~m]).predict(X[m]) if (~m).any() and len(set(fold_of)) > 1 \
                    else make_model().fit(X, y).predict(X[m])
            offs = np.cumsum([0] + [len(r[3]) for r in res_tr])
            best_tau, best_pq = 0.0, -1
            for tau in np.arange(0.0, 0.61, 0.05):
                sc = Score()
                for i, r in enumerate(res_tr):
                    sc_i = oof[offs[i]:offs[i + 1]]
                    pred = [q for q, v in zip(r[1], sc_i) if v >= tau]
                    for eid, g in GT[r[0]]:
                        sc.add(eid, g, pred)
                if sc.pq > best_pq:
                    best_tau, best_pq = round(float(tau), 2), sc.pq
            model = make_model().fit(X, y)
            res_te = pool.map(_inst, [(s, rp) for s in te], chunksize=2)
            sc_r = Score()
            for s, insts, x, _ in res_te:
                pred = [q for q, v in zip(insts, model.predict(x)) if v >= best_tau] if len(x) else []
                for eid, g in GT[s]:
                    sc_r.add(eid, g, pred)
            pooled["pp"].merge(sc_pp); pooled["resc"].merge(sc_r)
            report[k] = dict(pp=pp, tune_pq=round(in_pq, 4), tau=best_tau, held_pp=sc_pp.summary(),
                             held_resc=sc_r.summary(), minutes=round((time.time() - t0) / 60, 1))
            print(f"fold {k}: tuned-on-others {in_pq:.4f} | held-out pp {sc_pp.pq:.4f} | held-out pp+rescorer "
                  f"{sc_r.pq:.4f} (tau {best_tau}) | {report[k]['minutes']} min | pp {pp}", flush=True)
    for key, sc in pooled.items():
        s = sc.summary()
        print(f"POOLED honest {key}: PQ {s['pq']:.4f}  SQ {s['sq']:.4f}  RQ {s['rq']:.4f}  TP {s['tp']} FP {s['fp']} "
              f"FN {s['fn']}  PQ'(r=.8) {pq_sparse(s):.4f}")
        report[f"pooled_{key}"] = s
    out = Path(a.out or ROOT / "runs" / f"honest_{a.tmpl.replace('{}', 'X')}.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print("saved", out)


if __name__ == "__main__":
    main()
