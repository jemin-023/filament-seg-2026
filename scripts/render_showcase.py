"""Render high-resolution showcase figures: held-out fold-0 images, 3 fold-0 models (never trained on them)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2, numpy as np, pandas as pd
from filseg.data import TRAIN_IMG, load_index
from filseg.pipeline import Pipeline
from filseg.rle import polygon_to_mask
import colorsys

W = ROOT / "weights"
pipe = Pipeline.from_files(sorted(str(p) for p in W.glob("*_f0.pt")), rescorer_pkl=W / "rescorer.pkl")
folds = pd.read_csv(ROOT / "data" / "folds.csv")
val = folds[(folds.fold == 0) & (folds.n_annot > 1)].stem.tolist()[:14]
idx = load_index()
out = ROOT / "figures"
res = []
for s in val:
    img = cv2.imread(str(TRAIN_IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
    lab = pipe.instances(img)
    pm = lab > 0
    un = [np.any(np.stack([polygon_to_mask(a['segmentation']) for a in e.anns]), 0) if e.anns else np.zeros_like(pm) for e in idx[s]]
    d = np.mean([2 * (pm & g).sum() / max(pm.sum() + g.sum(), 1) for g in un])
    res.append((d, s, img, lab))
    print(s, lab.max(), round(float(d), 3), flush=True)
res.sort(key=lambda r: -r[0])

def render(img, lab, gts, scale=1):
    h = img.shape[0]
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    # gentle contrast stretch
    lo, hi = np.percentile(img, [1, 99.7]); base = np.clip((base - lo) / (hi - lo), 0, 1) * 255
    n = int(lab.max())
    cols = np.array([[c * 255 for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.85, 1.0)][::-1] for i in range(n + 1)])
    m = lab > 0
    base[m] = 0.35 * base[m] + 0.65 * cols[lab[m]]
    up = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    for k in range(1, n + 1):
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(up, [c * scale for c in cs], -1, tuple(int(x) for x in cols[k]), 2)
    for g in gts:
        cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(up, [c * scale for c in cs], -1, (255, 255, 255), 1, cv2.LINE_AA)
    return up

for rank, (d, s, img, lab) in enumerate(res[:3]):
    gts = [polygon_to_mask(a["segmentation"]) for a in idx[s][0].anns]
    full = render(img, lab, gts, 2)  # 4096x4096
    cv2.imwrite(str(out / f"showcase_{rank}_{s}_full.png"), full)
    ys, xs = np.nonzero(lab)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = np.clip(cy - 450, 0, 2048 - 900), np.clip(cx - 450, 0, 2048 - 900)
    crop = render(img, lab, gts, 3)[y0 * 3:(y0 + 900) * 3, x0 * 3:(x0 + 900) * 3]
    cv2.imwrite(str(out / f"showcase_{rank}_{s}_crop.png"), crop)
    print("saved", s, d)
