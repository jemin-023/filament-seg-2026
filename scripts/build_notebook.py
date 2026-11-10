"""Generate notebooks/pipeline.ipynb (the competition deliverable) from code, so it stays in sync with src/."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Solar filament instance segmentation: full inference pipeline

This notebook reproduces our submission **without retraining**: it loads the committed weights from `weights/`,
runs the whole pipeline on the provided test H-alpha images, writes `submission.csv`, and reports runtime.
An optional section re-scores our held-out validation fold with an exact re-implementation of the organizers'
Panoptic Quality scorer and draws the IoU / Dice / one-to-many / many-to-one distributions.

**Pipeline:** 2048x2048 grayscale image → U-Net ensemble (EfficientNetV2-S and ResNet34 encoders, 3 heads:
filament mask, spine, boundary) with horizontal-flip TTA → hysteresis instance grouping + thickness calibration
→ learned instance rescoring → disjoint filament masks → COCO RLE.

Set `FILSEG_DATA` to the folder that contains `train/` and `test/` if the data is not at the default location."""),
    code("""import os, sys, time, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import cv2, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from filseg.data import TEST_IMG, TRAIN_IMG, CACHE, load_index
from filseg.pipeline import Pipeline
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| test images:", TEST_IMG)"""),
    md("## 1. Load the trained models and post-processing configuration"),
    code("""WEIGHTS = ROOT / "weights"
CKPTS = sorted(str(p) for p in WEIGHTS.glob("*.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipe = Pipeline.from_files(CKPTS, rescorer_pkl=WEIGHTS / "rescorer.pkl", tta=("id", "h"), device=DEVICE)
print("models:", [Path(c).name for c in CKPTS])
print("post-processing:", pipe.pp.dict())
print("rescorer threshold:", pipe.rescorer and pipe.rescorer["tau"])"""),
    md("## 2. Predict the test set and write `submission.csv`"),
    code("""stems = sorted(p.stem for p in TEST_IMG.glob("*.jpeg"))
rows, t0 = [], time.time()
for s in stems:
    img = cv2.imread(str(TEST_IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
    rows += pipe.rows(s, pipe.instances(img))
elapsed = time.time() - t0
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
sub.to_csv(ROOT / "submission.csv", index=False)
print(f"{len(stems)} images -> {len(sub)} filaments ({len(sub)/len(stems):.1f}/image)")
print(f"end-to-end {elapsed/len(stems):.2f} s/image on {DEVICE}; per stage:", pipe.timing_summary())
sub.head()"""),
    md("## 3. Validation on the held-out fold (optional, ~5 min on GPU)\n"
       "Fold 0 of our date-grouped split was never used for training the fold-0 models. Each annotator of an "
       "image is a separate ground-truth entry, exactly as in the organizers' self-evaluation notebook."),
    code("""RUN_VALIDATION = True
if RUN_VALIDATION:
    from filseg.evaluate import load_gt, human_reference, score_labels
    from filseg.viz import plot_distributions
    folds = pd.read_csv(CACHE / "folds.csv") if (CACHE / "folds.csv").exists() else pd.read_csv(ROOT / "data" / "folds.csv")
    val = folds[folds.fold == 0].stem.tolist()
    gt = load_gt(val)
    labels = {}
    for s in val:
        img = cv2.imread(str(TRAIN_IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        labels[s] = pipe.instances(img)
    model_score = score_labels(labels, gt)
    multi = [s for s in val if len(gt[s]) > 1]
    human = human_reference({s: gt[s] for s in multi})
    model_multi = score_labels({s: labels[s] for s in multi}, gt)
    print("model, all fold-0 images     :", model_score.summary())
    print("model, multi-annotated images:", model_multi.summary())
    print("annotator vs annotator (same):", human.summary())
    plot_distributions(model_score, human); plt.show()"""),
    md("## 4. Qualitative examples (prediction fill = colours, GT outlines = white)"),
    code("""if RUN_VALIDATION:
    from filseg.viz import overlay
    from filseg.rle import polygon_to_mask
    idx = load_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, s in zip(axes, multi[:3]):
        img = cv2.imread(str(TRAIN_IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        gts = [polygon_to_mask(a["segmentation"]) for a in idx[s][0].anns]
        ys, xs = np.nonzero(labels[s]) if labels[s].any() else (np.array([1024]), np.array([1024]))
        cy, cx = int(np.median(ys)), int(np.median(xs))
        crop = (max(cy - 300, 0), max(cx - 300, 0), 600, 600)
        ax.imshow(overlay(img, labels[s], gts, crop)); ax.set_title(s); ax.axis("off")
    plt.tight_layout(); plt.show()"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "pipeline.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
