"""Check our scorer against the organizers' notebook code (executed verbatim) on annotator-vs-annotator data."""
import json, random, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import load_index
from filseg.metrics import Score, gt_insts
from filseg.rle import polygon_to_rle


def official_namespace():
    nb = json.loads((ROOT / "ext/selfeval/self-evaluation-notebook.ipynb").read_text())
    code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {}
    for src in code[:4]:          # imports, df placeholders, helpers, get_pq_score
        exec(src, ns)
    return ns


def main(n_files=25, seed=0):
    idx = load_index()
    multi = [s for s, es in idx.items() if len(es) >= 2]
    random.Random(seed).shuffle(multi)
    gt_rows, pred_rows, ours = [], [], Score()
    for stem in multi[:n_files]:
        es = idx[stem]
        pred_entry, gt_entries = es[0], es[1:]
        for k, a in enumerate(pred_entry.anns):
            pred_rows.append((f"{stem}_{k}", polygon_to_rle(a["segmentation"])))
        pred = gt_insts(pred_entry.anns)
        for e in gt_entries:
            for k, a in enumerate(e.anns):
                gt_rows.append((f"{e.entry_id}_{k}", polygon_to_rle(a["segmentation"])))
            ours.add(e.entry_id, gt_insts(e.anns), pred)
    ns = official_namespace()
    cols = ["filament_id", "segmentation_rle"]
    odf = ns["get_overlap_df"](pd.DataFrame(gt_rows, columns=cols), pd.DataFrame(pred_rows, columns=cols))
    off = ns["get_pq_score"](odf)
    print(f"official PQ {off:.6f}  ours {ours.pq:.6f}  {ours.summary()}")
    assert abs(off - ours.pq) < 1e-9, "scorer mismatch"
    print("OK: scorer matches official")


if __name__ == "__main__":
    main()
