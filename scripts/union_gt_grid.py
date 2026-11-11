"""If the test GT were the union of annotators (merged into connected components), which pp would be optimal?"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import CACHE
from filseg.evaluate import load_gt, load_prob, prob_dir
from filseg.metrics import Score, label_to_insts
from filseg.postprocess import PPParams, full

NAME = sys.argv[1]
PROBS, UG, GT = {}, {}, {}

def union_insts(entries):
    m = np.zeros((2048, 2048), bool)
    for _, g in entries:
        for i in g:
            m[i.y0:i.y0 + i.crop.shape[0], i.x0:i.x0 + i.crop.shape[1]] |= i.crop
    lab, _ = ndi.label(m, structure=np.ones((3, 3)))
    return label_to_insts(lab)

def one(args):
    s, p = args
    pred = label_to_insts(full(PROBS[s], PPParams(**p)))
    a, b = Score(), Score()
    a.add(s, UG[s], pred)
    for eid, g in GT[s]:
        b.add(eid, g, pred)
    return a, b, sum(x.area for x in pred), len(pred)

f = pd.read_csv(CACHE / "folds.csv"); stems = f[f.fold == 0].stem.tolist()
GT.update(load_gt(stems))
stems = [s for s in stems if len(GT[s]) >= 2]
for s in stems:
    PROBS[s] = np.round(load_prob(prob_dir(NAME) / f"{s}.png")[:1] * 255).astype(np.uint8)
    UG[s] = union_insts(GT[s])
ua = sum(x.area for s in stems for x in UG[s]); un = sum(len(UG[s]) for s in stems)
ia = np.mean([sum(x.area for x in g) for s in stems for _, g in GT[s]]) * len(stems)
print(f"{len(stems)} multi imgs; union GT inst/img {un/len(stems):.2f}, union area / mean-annotator area {ua/ia:.3f}")
base = json.loads(Path("configs/pp_effv2s_f0.json").read_text())["params"]
rows = []
with Pool(4) as pool:
    for dil in (0, 1, 2, 3):
        for ts in (0.35, 0.45, 0.55, 0.65):
            for th in (0.3, 0.4):
                p = dict(base, dilate=dil, t_shape=ts, t_high=th, min_area=200)
                U, A, area, n = Score(), Score(), 0, 0
                for a, b, ar, k in pool.imap_unordered(one, [(s, p) for s in stems], chunksize=2):
                    U.merge(a); A.merge(b); area += ar; n += k
                r = dict(dil=dil, t_shape=ts, t_high=th, pq_union=round(U.pq, 4), pq_annot=round(A.pq, 4),
                         area_vs_union=round(area / ua, 3), inst_img=round(n / len(stems), 2))
                rows.append(r); print(r, flush=True)
d = pd.DataFrame(rows)
print("BEST union:", d.loc[d.pq_union.idxmax()].to_dict())
print("BEST annot:", d.loc[d.pq_annot.idxmax()].to_dict())
