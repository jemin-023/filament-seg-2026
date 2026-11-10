"""Tune post-processing parameters for PQ on cached validation probabilities (Optuna TPE).

python scripts/tune_postproc.py --name effv2s_f0 --fold 0 --trials 150 --jobs 6
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import CACHE  # noqa: E402
from filseg.evaluate import load_gt, load_prob, prob_dir  # noqa: E402
from filseg.metrics import Score, label_to_insts  # noqa: E402
from filseg.postprocess import PPParams, full  # noqa: E402

PROBS, GT = {}, {}


def _score_one(args):
    stem, params = args
    lab = full(PROBS[stem], PPParams(**params))
    pred = label_to_insts(lab)
    sc = Score()
    for eid, g in GT[stem]:
        sc.add(eid, g, pred)
    return sc


def evaluate(pool, stems, params) -> Score:
    total = Score()
    for sc in pool.imap_unordered(_score_one, [(s, params) for s in stems], chunksize=4):
        total.merge(sc)
    return total


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    f = pd.read_csv(CACHE / "folds.csv")
    stems = f[f.fold == a.fold].stem.tolist()
    for s in stems:  # keep only the semantic channel as uint8 (shared with workers via fork)
        PROBS[s] = np.round(load_prob(prob_dir(a.name) / f"{s}.png")[:1] * 255).astype(np.uint8)
    GT.update(load_gt(stems))
    out = CACHE / "probs" / a.name / "_tuning"
    out.mkdir(exist_ok=True)
    with Pool(a.jobs) as pool:
        base = evaluate(pool, stems, PPParams().dict())
        print("default params:", base.summary(), flush=True)

        def objective(trial):
            p = dict(
                t_low=trial.suggest_float("t_low", 0.15, 0.5, step=0.05),
                t_high=trial.suggest_float("t_high", 0.4, 0.8, step=0.05),
                t_shape=trial.suggest_float("t_shape", 0.25, 0.65, step=0.05),
                r_close=trial.suggest_int("r_close", 0, 7),
                dilate=trial.suggest_int("dilate", -2, 3),
                fill=trial.suggest_categorical("fill", [True, False]),
                min_area=trial.suggest_int("min_area", 50, 600, step=25),
            )
            sc = evaluate(pool, stems, p)
            trial.set_user_attr("summary", sc.summary())
            return sc.pq

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.enqueue_trial({k: v for k, v in PPParams().dict().items() if k != "min_peak"})
        t = time.time()
        study.optimize(objective, n_trials=a.trials, show_progress_bar=False)
        best = study.best_trial
        print(f"best PQ {best.value:.4f} in {(time.time() - t) / 60:.1f} min: {best.params}")
        print("summary:", best.user_attrs["summary"])
        rows = [dict(**t.params, **t.user_attrs["summary"]) for t in study.trials if t.value is not None]
        pd.DataFrame(rows).sort_values("pq", ascending=False).to_csv(out / "trials.csv", index=False)
        (out / "best.json").write_text(json.dumps({"params": best.params, "summary": best.user_attrs["summary"]}, indent=1))


if __name__ == "__main__":
    main()
