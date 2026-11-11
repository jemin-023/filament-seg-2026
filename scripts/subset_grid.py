"""Does the PQ-optimal thickness/density differ between single-annotator and multi-annotator images?
python scripts/subset_grid.py --name ens2_f0 --fold 0
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import CACHE
from filseg.evaluate import load_gt, load_prob, prob_dir, pq_sparse
from filseg.metrics import Score, label_to_insts
from filseg.postprocess import PPParams, full

PROBS, GT = {}, {}

def one(args):
    s, p = args
    pred = label_to_insts(full(PROBS[s], PPParams(**p)))
    sc = Score()
    for eid, g in GT[s]:
        sc.add(eid, g, pred)
    return s, sc

ap = argparse.ArgumentParser(); ap.add_argument("--name"); ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--base", default="configs/pp_effv2s_f0.json"); a = ap.parse_args()
f = pd.read_csv(CACHE / "folds.csv"); stems = f[f.fold == a.fold].stem.tolist()
for s in stems:
    PROBS[s] = np.round(load_prob(prob_dir(a.name) / f"{s}.png")[:1] * 255).astype(np.uint8)
GT.update(load_gt(stems))
single = {s for s in stems if len(GT[s]) == 1}
print(f"{len(single)} single-annotator imgs, {len(stems) - len(single)} multi")
base = json.loads(Path(a.base).read_text())["params"]
rows = []
with Pool(4) as pool:
    for dil in (-1, 0, 1, 2, 3):
        for ts in (base["t_shape"] - 0.2, base["t_shape"] - 0.1, base["t_shape"]):
            for ma in (150, 350):
                p = dict(base, dilate=dil, t_shape=round(ts, 2), min_area=ma)
                S1, SM, SA = Score(), Score(), Score()
                for s, sc in pool.imap_unordered(one, [(s, p) for s in stems], chunksize=4):
                    (S1 if s in single else SM).merge(sc); SA.merge(sc)
                r = dict(dil=dil, t_shape=round(ts, 2), min_area=ma, pq_all=SA.pq, pq_single=S1.pq, pq_multi=SM.pq,
                         sparse_all=round(pq_sparse(SA.summary()), 4))
                rows.append(r); print(r, flush=True)
d = pd.DataFrame(rows)
for c in ["pq_all", "pq_single", "pq_multi", "sparse_all"]:
    print("best by", c, d.loc[d[c].idxmax()].to_dict())
d.to_csv(prob_dir(a.name) / "_tuning" / "subset_grid.csv", index=False)
