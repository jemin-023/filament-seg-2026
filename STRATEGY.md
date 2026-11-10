# Winning strategy — Solar Filament Segmentation 2026

Written 2026-11-11 (Day 1). Deadline 2026-11-15 23:59 CDT. Tags: **[V]** = verified from COMPETITION_NOTES.md, **[M]** = measured today on the local data (`eda/eda.py`, `eda/agree.py`), **[A]** = assumption or prior knowledge that still needs checking.

## Executive summary
1. **The ceiling is human agreement.** [M] Two annotators of the same image agree at **PQ 0.354** (SQ 0.61 × RQ 0.55, Dice 0.62, 598 pairs). The organizers' "PQ > 0.35" bar is human-level, so we should aim to approximate the *consensus* annotator, which can beat any single annotator against another.
2. **Architecture:** a bottom-up U-Net with pretrained **EfficientNetV2-S** encoder (1-channel input), one decoder with 3 heads (soft semantic mask, spine heatmap, boundary). Train on full-res 768² crops and run inference on the whole 2048² image (no tiling).
3. **Fallback:** U-Net-ResNet34 on the whole disk downsampled to 1024², semantic head only. This is also the Day-1 baseline.
4. **Loss:** BCE + soft Dice against **soft multi-annotator targets** (mean of annotator masks), plus 0.5×(BCE+Dice) on the spine and 0.3×BCE on the boundary. AdamW, cosine schedule, AMP, EMA.
5. **Post-processing (the biggest lever):** hysteresis grouping (low threshold for connectivity, separate threshold for shape), closing, **mask-thickness calibration (dilate/erode d px)**, spine-guided fragment merging, min-area filter, and a small **instance rescoring classifier**. A candidate is kept when P(match) ≳ 0.3; the derivation is in §1.
6. **Instances rarely touch** [M]: GT connected components (5230) slightly *outnumber* GT instances (4993), and only 0.4% of annotation sets have touching instances. So connected components plus merge logic beats Mask R-CNN-style decoding here, and fragmentation matters more than merging.
7. **Validation:** 5 folds grouped by **calendar date** (same day = same Sun across the 6 GONG sites). Score against each annotator separately and average. The public LB covers ~90 images, so its noise is about ±0.02 PQ (1 SE); ignore LB deltas below 0.03.
8. **Ensemble:** 2 models × {identity, hflip} TTA, about 2–3 s/image on the 4050 [A, measure].
9. **Expected PQ:** 0.36–0.45 on local CV, with similar values on the LB [A]. The test set is distributed like train [M]: same 6 instruments, same years, 31/176 test dates also appear in train.
10. **The qualitative 30%:** we lead with the human-ceiling analysis ("model ↔ annotator agreement vs annotator ↔ annotator") and with the one-to-many / many-to-one statistics, runtime table, and post-processing ablation.

---

## Facts measured today [M]
| Item | Value |
|---|---|
| Train | 1154 annotation sets over **707 unique images**; 411 images ×1 annotator, 145 ×2, 151 ×3; 37 annotator-batch prefixes |
| Instances | 8199 total, mean 7.1 / median 7 / max 26 per annotation set; no empty sets; single polygon each |
| Area (px) | p5 316, p25 670, **median 1228**, p75 2440, p95 6963, max 37.7k; bbox fill median 0.24 (elongated) |
| Instruments | Bh/Ch/Lh/Mh/Th/Uh balanced in train and test |
| Years | 2011–2022 in both; test 180 images, 0 file overlap, **31/176 test dates present in train** |
| Inter-annotator | PQ 0.354 (median 0.357, IQR 0.24–0.47), SQ ≈ 0.61, RQ ≈ 0.55, recall of A's instances by B 55%, semantic Dice 0.62 |
| Touching | 0.4% of annotation sets have merged instances; CC > instances (some polygons rasterize into several pieces) |
| Env | **torch not installed**; system Python 3.14; numpy+PIL only; 24 GB disk free; 6 GB VRAM |

## 1. Metric diagnosis
- PQ = Σ IoU(TP) / (TP + ½FP + ½FN) = SQ × RQ, where a match requires IoU > 0.5 and is one-to-one. Dice is pixel overlap and is blind to how the pixels are split into instances.
- **Matched IoUs sit close to the 0.5 threshold.** The human SQ is 0.61. Small systematic errors in mask width therefore flip TPs into an FP+FN pair, which costs about 1.0 in the denominator and the IoU in the numerator. → Calibrate **mask thickness** (threshold and dilation) *for PQ*, not for Dice.
- **Inclusion rule** (derived from the formula): adding a candidate that matches with probability p at IoU q changes the numerator by +pq and the denominator by +0.5. It therefore helps iff **p·q > PQ/2**. With PQ ≈ 0.4 and q ≈ 0.65, keep instances with **p > ≈0.3**. Faint instances should be included fairly liberally, and the cutoff should be calibrated.
- Ranked cost of failure modes:
  1. **Fragmentation**: one GT split into two preds with IoU < 0.5 each costs 1 FN + 2 FP.
  2. Width or boundary miscalibration.
  3. Faint candidates that annotators disagree on (inter-annotator recall is only 55%, so these are coin-flips; the rescorer handles them).
  4. Over-merging of close neighbours. This is rare in GT, but a fat threshold or a large closing creates it.
  5. Tiny instances (< 300 px), whose IoU is unstable.
- Ceiling: against a single annotator, the agreement of a "consensus" predictor is typically somewhat above pairwise human agreement. My estimate is a realistic top of ~0.45 [A]. Implications: (a) train on soft targets, (b) threshold the probability near the majority-vote level (0.4–0.5) to set shape, and tune that threshold, (c) never trust Dice-optimal settings for PQ.
- **Must check in the self-eval notebook** [A]: whether PQ is aggregated over the whole dataset or averaged per image, how multi-annotator GT is represented in the test set, how images with zero predictions are handled, and which masks the Dice uses (union).

## 2. Architecture
**Primary: U-Net (smp) + `tu-tf_efficientnetv2_s.in21k_ft_in1k`, `in_channels=1`, decoder (256,128,64,32,16) with scSE attention, 3 output channels.**
- Channel 0 is the soft semantic mask. Channel 1 is the spine heatmap: the annotated spine polyline rasterized at 5 px width with Gaussian σ = 2. Channel 2 is the instance boundary band: mask minus its 3 px erosion.
- About 24M params, so weights are ~48 MB in fp16 and 2 models stay < 100 MB with no LFS.
- Why this design:
  - The stride-2 stem keeps fine skip connections, which thin barbs need. ConvNeXt's stride-4 stem loses them.
  - It is fully convolutional, so it runs on the whole 2048² image at inference (fp16, < 2 GB).
  - The spine head supervises connectivity along the filament axis and provides seeds for splitting and merging.
  - The boundary head sharpens SQ and enables watershed splits.
- Train on **768² crops at native resolution**. Use batch 4 locally (AMP, channels_last) or 1024² × batch 8 on a Kaggle T4/P100.

**Fallback (also the Day-1 baseline): U-Net-ResNet34, 1-ch, whole disk resized to 1024², semantic only.** It trains in ~1.5 h locally, runs about 4× faster at inference, and the probability map is upsampled to 2048 before thresholding.

| Alternative | Verdict for PQ here |
|---|---|
| UNet++ | Small gain at large memory cost at high resolution; not worth it with 6 GB |
| SegFormer / Mask2Former | Stride-4 outputs blur thin structures; Mask2Former is too heavy and slow to tune in 5 days with 707 images |
| SAM fine-tune | Works at 1024 input with 2× downsampling, slow inference (runtime is scored), weights > 300 MB |
| Mask R-CNN | 28×28 RoI masks are poor on diagonal elongated shapes with bbox fill 0.24; NMS suppresses neighbours |
| **Bottom-up (chosen)** | GT instances almost never touch, so CC + targeted merge/split matches the data and is fast and explainable |

## 3. Loss and training recipe
- **Targets:**
  - Semantic: the soft mean over annotators, i.e. values in {0, ⅓, ½, ⅔, 1}.
  - Spine: the mean of per-annotator spine heatmaps.
  - Boundary: the mean of per-annotator bands.
  - Precompute all targets as uint8 PNGs (~3 × 707 files). Preload the images into RAM as uint8 (2.8 GB).
- **Loss:** BCE + soft Dice (per-batch) on semantic; 0.5 × (BCE + Dice) on spine; 0.3 × BCE on boundary. Tversky/focal/clDice are not needed: the spine head stands in for clDice, and Dice handles sparsity. Try Lovász as a fine-tune only if time remains.
- **Crop sampling:** 70% centred on a random filament with ±200 px jitter; 30% uniform over the disk (hard negatives such as sunspots, plage, the limb, and cloud shadows).
- **Augmentations** (physically valid; segmentation is orientation-free, so mirroring chirality is fine):
  - D4 (flips and 90° rotations), plus arbitrary rotation ±180° with p = 0.3, plus scale 0.85–1.15.
  - Brightness/contrast ±20% and gamma 0.8–1.25.
  - A low-frequency multiplicative field to simulate clouds, haze, and limb-darkening variation (smooth random field at amplitude ±15%).
  - Gaussian blur σ 0–1.5 for seeing, Gaussian noise σ ≤ 3 DN, and JPEG quality 70–95.
  - No elastic deformation.
- **Optimizer:** AdamW, lr 3e-4 (decoder) and 1e-4 (encoder), weight decay 1e-2, 1-epoch warmup, cosine schedule to 1e-6. Grad clip 1.0, EMA 0.999, AMP fp16.
- **Budget:** 4 crops per training file per epoch (~2.3k crops). Target **30 epochs ≈ 3–3.5 h on the 4050** [A]. Measure the first epoch: if it takes more than 8 min, drop to 20 epochs or move the run to Kaggle.

## 4. Instance extraction and post-processing
Run on cached, TTA-averaged probabilities: P (semantic), S (spine), and B (boundary).
1. Restrict to the solar disk: fit the disk by thresholding the image, then keep r < R + 5 px [A: check whether any GT lies off-limb].
2. **Grouping with hysteresis:** take the components of `P > t_low` that contain a pixel with `P > t_high`. Then apply closing with radius r_c and fill holes.
3. **Shape:** within each group, keep the pixels with `P > t_shape`, then **dilate or erode by d px** (thickness calibration). This step decouples connectivity from width.
4. **Merge fragments:** merge two groups when the gap between spine endpoints is < g px, their orientations differ by < 30°, and the mean P along the bridging segment is > t_low/2.
5. **Split:** when one group contains ≥ 2 spine components longer than 40 px each and B shows a ridge between them, run watershed on B seeded by the spine components. Keep this step only if it improves validation PQ; merges are rare in GT.
6. **Filter:** apply area ≥ a_min, then rescore. The rescorer is a logistic regression (or a small GBM) on the features [area, mean P, max P, mean S, elongation, mean intensity contrast vs a 101 px background, distance to limb, instrument]. Its label is whether the instance matched GT at IoU > 0.5 on the OOF/val set. Keep instances with p ≥ τ, where τ starts at ~0.3.
7. Guarantee ≥ 1 row per test image, using the best-scoring candidate if necessary [A: confirm the scorer's behaviour].

| Param | Grid (start) |
|---|---|
| t_high | 0.4, **0.5**, 0.6 |
| t_low | 0.2, **0.3**, 0.4 |
| t_shape | 0.35, **0.45**, 0.55 |
| r_c | 0, **3**, 5, 9 |
| d | −2 … **0** … +3 |
| a_min | 100, **200**, 300, 500 |
| g | 0, **15**, 30 |
| τ | 0.2 … **0.3** … 0.5 |

Tune with Optuna (~200 trials), using multiprocessing on CPU over the cached fold probabilities, with local PQ as the objective. **Skip DenseCRF:** pydensecrf is hard to install on modern Python, it is slow, and the gain is uncertain because GT is smooth hand-drawn polygons, not pixel-precise edges. Mention it in the report as future work.

## 5. Validation protocol
- **Split:** 5 folds via `GroupKFold`, with groups = calendar date (YYYYMMDD). This keeps all annotators of a file *and* all same-day images from other sites together. Use fold 0 for every decision. The LB will read slightly higher than CV, because 18% of test dates appear in train.
- **Scoring:** use a local PQ and Dice that **exactly replicate the self-eval notebook** (`kaggle kernels pull azimahmadzadeh/self-evaluation-notebook`). On multi-annotated validation files, score against each annotator separately and average. Also compute annotator-vs-annotator agreement on the same validation files: **sanity check: the human pairwise PQ must come out ≈ 0.35**.
- **Report:** PQ, SQ, RQ, Dice, the IoU distribution of matches, one-to-many (one GT overlapped by ≥ 2 preds, each covering ≥ 10% of the GT), many-to-one, and a breakdown by instrument and by area bin.
- **Leaderboard discipline:** public LB ≈ 90 images, and per-image PQ std ≈ 0.17 [M, from annotator pairs], so SE ≈ 0.018. Expect a public/private gap of up to ±0.03. Select by CV. Make at most ~2 LB submissions per day as sanity checks, and do no LB-driven threshold tuning or probing.

## 6. Ensembling and TTA (runtime is scored)
- Best gain per second, in order: (1) hflip TTA, (2) a second model (a different fold or seed), (3) vflip. Ensembles mainly stabilise P around the threshold, which matters a lot near IoU 0.5.
- **Final: 2 models × {id, hflip} = 4 passes at 2048² fp16.** Estimated cost [A, measure on Day 2]:
  - ≈ 0.4 s per pass
  - ≈ 0.3 s of CPU post-processing
  - **≈ 2 s/image on the 4050**, or ~6 min for the whole test set
- Report a runtime table covering GPU (4050), Kaggle T4, and CPU-only, plus single-model vs ensemble PQ, so judges can see the trade-off.

## 7. Qualitative 30% and repo
```
solar-filaments/
  README.md            # results table, quickstart (3 commands), pipeline figure, runtime
  requirements.txt     # pinned (pip freeze of the uv venv, Python 3.12)
  configs/{primary,fallback}.yaml
  src/filseg/{data.py, targets.py, augment.py, model.py, losses.py, train.py,
              infer.py, postprocess.py, rescore.py, metrics.py, rle.py, viz.py}
  weights/{primary_f0.fp16.pt, primary_f1.fp16.pt, rescorer.json}   # each < 100 MB, no LFS
  notebooks/pipeline.ipynb   # load weights -> predict test -> CSV -> val metrics + figures
  scripts/{make_targets.py, make_folds.py, tune_postproc.py, make_submission.py}
  tests/test_rle_metrics.py  # RLE round-trip; PQ on toy cases; annotator PQ ≈ 0.354
  report/                    # Overleaf sources + figures
```
- **Notebook:** runs top-to-bottom in a fresh venv using only `test_images`, writes `submission.csv`, and prints timing per stage. Training code is present, but the notebook does not call it.
- **Reproducibility check on Day 5:** fresh clone, `uv venv -p 3.12`, `pip install -r requirements.txt`, run the notebook, and diff the CSV against the submitted one.
- **4-page report outline:**
  1. Intro and data, including the multi-annotator analysis and the human PQ ceiling.
  2. Method: pipeline diagram (image → U-Net with 3 heads → hysteresis/merge/split → rescoring → RLE) and the soft targets.
  3. Experiments:
     - Ablation table (fallback → primary → + soft targets → + spine head → + hysteresis → + thickness calibration → + rescoring → + TTA/ensemble)
     - PQ vs post-processing parameter curves
     - Dice and IoU histograms vs the human distribution
     - One-to-many and many-to-one counts before and after merging
     - Per-instrument PQ
     - Runtime table
  4. Qualitative figure: best, median, and worst cases with GT from 2 annotators, plus a discussion and limitations section.

## 8. Day-by-day plan
**Day 1, Nov 11: foundation and a valid entry**
- H0–1: `uv venv -p 3.12`; install torch (CUDA 12.x), segmentation-models-pytorch, timm, pycocotools, opencv-python-headless, scikit-image, albumentations, optuna, scikit-learn. `git init` the private repo.
- H1–3: pull the self-eval notebook; write `metrics.py` and `rle.py` and their tests; reproduce human PQ ≈ 0.354.
- H3–4: targets, folds, and a **classical baseline**: background-flattened image < −kσ, CC, area filter. → **Submission #1 (format check).**
- H4–8: train the fallback R34 at 1024 on fold 0 (~1.5 h); add the post-processing v0 (threshold + CC + min-area). → **Submission #2.**
- **Go/no-go:** local PQ ≥ 0.25 and the LB accepts the CSV. Otherwise debug the RLE and the scorer before doing anything else.
- Overnight: start the primary fold-0 run.

**Day 2, Nov 12: primary model**
- Evaluate the primary fold 0. Build the cached-probability pipeline and the full post-processing (hysteresis, thickness, merge).
- Set up a Kaggle notebook for training other folds, and queue folds 1–2.
- Measure inference time.
- **Go/no-go:** the primary beats the fallback by ≥ 0.02 PQ on fold 0. If not, keep the R34 recipe at native-res crops and spend the compute on seeds and folds instead.

**Day 3, Nov 13: post-processing and rescoring**
- Run the Optuna post-processing search; train the rescorer; ablate split/watershed and the boundary head.
- Submit the best CV configuration. Start the figures notebook and report skeleton.

**Day 4, Nov 14: ensemble and freeze**
- Choose the ensemble and TTA under the ~2–3 s/image budget. **Freeze models and parameters by 18:00.**
- Finish the end-to-end inference notebook and the README; draft the report.

**Day 5, Nov 15: ship**
- Morning: reproducibility test from a fresh clone; finalize the report PDF on the Overleaf template.
- Select the 2 final Kaggle submissions (best CV, and best CV that is also good on LB).
- **Make the repo public, then submit the Google form by 18:00 CDT** (6 h buffer).

**Cut order if behind schedule:** watershed split → boundary head → rescorer (fall back to the a_min threshold only) → second model (keep hflip TTA) → Kaggle folds. **Never cut:** the local scorer, the thickness calibration, the notebook, or the reproducibility test.

## 9. Risks (ranked) and the cheapest test for each
| # | Risk | Cheapest check |
|---|---|---|
| 1 | The local scorer differs from Kaggle (aggregation, multi-annotator GT, empty images) | Pull the notebook; reproduce human PQ; submit a known prediction and compare LB vs local |
| 2 | Width miscalibration around IoU 0.5 | Sweep d ∈ [−3, 3] on fold 0; plot PQ vs d |
| 3 | Fragmentation | Count one-to-many on fold 0 for the baseline; if > 10% of GT, prioritise hysteresis and merging |
| 4 | Faint filaments labelled inconsistently (RQ 0.55) | Per-annotator recall vs contrast bins; justifies the rescorer threshold |
| 5 | Env/compute: no torch, Py 3.14, 24 GB disk, laptop logout | uv venv 3.12; nohup/tmux; checkpoint every epoch; move long runs to Kaggle |
| 6 | Overfitting the public LB (~90 images) | Decide by CV only; treat Δ < 0.03 as noise |
| 7 | Repo or weights too big / LFS bandwidth limits | Save fp16 state dicts, < 100 MB per file, no LFS |
| 8 | Deadline/admin: form, public repo, Overleaf template | Download the template on Day 1; calendar reminder; submit the form 6 h early |
| 9 | Off-limb or limb artifacts, clouds | Plot the GT centroid radius distribution; inspect the 20 worst FP |
| 10 | Images near-duplicated across sites/dates inflate CV | Date-grouped folds (done); compare against a random-file split for the report |

Also to check before committing [A]:
- Whether GONG H-α JPEGs are already limb-darkening-corrected (plot the radial profile of 5 images). If not, add a radial-normalised second input channel.
- The disk radius per instrument.
- The minimum gap between neighbouring GT instances, which sets the cap on r_c and g.

## First 10 actions for today (coding agent, each with an acceptance check)
1. **Env:** `uv venv -p 3.12 .venv` and install the stack. ✅ `python -c "import torch;print(torch.cuda.is_available())"` prints True, and smp builds the primary model and runs a forward pass on a 1×1×2048×2048 tensor in fp16 without OOM.
2. **Repo:** `git init`, set the user config, create the skeleton above, `gh repo create --private`. ✅ First commit pushed.
3. **Scorer:** pull the self-eval notebook and port it to `metrics.py` (PQ, SQ, RQ, Dice, one-to-many, many-to-one). ✅ Toy unit tests pass, and annotator-vs-annotator PQ over the training files is 0.354 ± 0.005, matching `eda/agree.py`.
4. **RLE:** `rle.py` encode/decode with pycocotools, counts string only. ✅ Round-trip is exact on 50 GT masks, and the CSV columns are exactly `filament_id,segmentation_rle`.
5. **Targets:** `make_targets.py` writes soft-mask, spine, and boundary PNGs. ✅ 707 files × 3 channels; the soft-mask value set is ⊆ {0, 85, 128, 170, 255}; a visual overlay for 5 files looks right.
6. **Folds:** `make_folds.py` builds date-grouped 5 folds. ✅ No date and no file_name appears in two folds; instruments are balanced to within ±5 per fold.
7. **Classical baseline and submission #1:** ✅ The CSV has 180 images covered and the Kaggle upload is accepted. Record the LB score.
8. **Fallback training:** R34 @ 1024 on fold 0, 40 epochs, EMA, checkpoint every epoch, run under nohup. ✅ Fold-0 PQ ≥ 0.25 (post-processing v0: t = 0.5, a_min = 200).
9. **Submission #2** from the fallback model. ✅ The LB score is within ±0.05 of fold-0 PQ; a larger gap is a scorer-mismatch red flag, so return to step 3.
10. **Primary run launched overnight** (EfficientNetV2-S, 3 heads, 768 crops, 30 epochs). ✅ The first epoch finishes in < 8 min with loss decreasing; the log and checkpoints are written to `runs/primary_f0/`.
