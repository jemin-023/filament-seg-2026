# Analysis of Public Kaggle Notebooks: Solar Filament Segmentation Challenge 2026

## 1. Summary of Public Notebooks

| Notebook | Claimed LB | Model / Arch | Input Size / Tiling | Loss | Post-Processing / Instance Logic | TTA / Ensemble | Metric or LB Tricks |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| `0-9-solar-filament-baseline-pipeline-u-net` | None (~0.02) | smp U-Net (ResNet34, ImageNet) | 512×512 direct resize (no tiling) | 0.5 BCE + 0.5 Dice | Binary thr > 0.5; connected components; area < 30 discarded; fallback empty mask | None | **Encoding Bug**: Encodes 512×512 masks to RLE without upsampling to 2048×2048 |
| `best-score-magfilo-yolo11s-detect-crop-u-net` | ~0.409 (val) | 2-stage: YOLO11s detector + Crop U-Net refiner | YOLO 1280×1280; Refiner 512×512 crops (context 1.5×, pad 0.40) | 0.45 BCEWithLogits + 0.55 Soft Dice | Conf 0.28, NMS IoU 0.50; refiner thr 0.50; min area 200; greedy score dispute resolution | Horizontal flip TTA; optional box-jitter TTA (±8px) | 3-channel refiner conditioned on image, CLAHE, and padded bounding-box seed |
| `cv-solar-filament-mask-r-cnn-u-net-refiner` | None (CV Dice) | 2-stage: Mask R-CNN (ResNet-50 FPN v2) + Crop ResNet18 U-Net refiner | Detector 1024×1024; refiner 256×256 native crops | Mask R-CNN multi-task; Refiner 0.5 BCE + 0.5 Soft Dice | Score thr 0.60–0.70, mask 0.50; refiner thr 0.85; overlap merging; Hungarian matching | None | Baseline for Fususu 0.70; multi-annotator agreement study (union Dice ~0.65) |
| `lb-0-70-topology-safe-solar-filaments-handoff` | **0.70** | Mask R-CNN (ResNet-50 FPN v2) + ResNet18 Crop U-Net (frozen weights) | Detector 1024×1024; refiner 256×256 native crops | Pretrained / frozen | Detector score 0.70, mask 0.50; refiner thr 0.70; seed split (IoU 0.10, dist 8px); positive overlap merge (1,299 instances) | 4-view flip TTA with topology safety guards (area ratio [0.60, 1.70], Dice ≥ 0.62) | Submitted on pre-Aug 7 evaluator; ~7.2 masks/image matched old evaluator anomaly |
| `magfilo-v1-evaluation-aligned-solar-filament-s` | ~0.448 (val) | 2-stage: YOLO (V10/V18) detector + smp Crop U-Net refiner (V12) | YOLO train 1280×1280, infer 1536×1536; refiner 512×512 crops (pad 0.20) | YOLO detection loss; Refiner BCE + Dice | Conf 0.275, NMS IoU 0.50; refiner mask thr 0.65; strictly disjoint RLE | Single model (V15 vs V18 comparison) | `consolidated_boxes` fuses cross-annotator masks (IoU ≥ 0.40) into non-duplicated targets |
| `new-metric-solar-filament` | None (PQ aligned) | 5-fold YOLO11s-seg + 5-fold ResNet18 Crop U-Net refiners | YOLO 1024×1024; refiner 256×256 crops (context 1.8×, min crop 96) | YOLO seg loss; Refiner 0.45 BCEWithLogits + 0.55 Soft Dice | Conf 0.20 (fallback 0.05); duplicate seed removal (IoU 0.60); refiner thr 0.65; min area 96; Otsu fallback seed | 5-fold ensemble (clustering IoU ≥ 0.50, support ≥ 2); 4-view topology-safe TTA | Full alignment to official post-Aug 7 Panoptic Quality (IoU > 0.5 TP match); medoid annotators |
| `please-fix-the-leaderboard` | **1.00** | **None** (diagnostic evaluator exploit) | N/A | None | Exactly 5 empty masks (`PPP8`) submitted for only 5 of 180 test images; 0 foreground pixels | None | **Metric Exploit**: Demonstrates pre-Aug 7 evaluator gave 1.00 to incomplete, all-empty submissions |
| `simple-dark-filament-segmentation-0-46` | **0.46** | **Classical CV** (no ML model, no GPU, no training) | Native 2048×2048 full disk | None | Solar disk mask (<1000px); Gaussian blur (σ=40) residual; 90th percentile threshold; 3×3 dilation; components (area 80..150k, fill ≤ 0.99, top 120) | None | Disk-isolated background subtraction at native resolution; custom compressed byte RLE |
| `solar-filament-grouped-cv-and-strategy-writeup` | **0.66** | 5-fold U-Net (1024-base32) | 1024×1024 crops from native 2048×2048 | Soft consensus targets (0.75 consensus + 0.25 random annotator) | Connected components with area filtering (`96 <= area <= 40000`), mean prob ≥ 0.8 | 5-fold ensemble + 4-way TTA | Solar disk CLAHE (`clip_limit=2.0`, `grid=[8,8]`); off-limb masking via Otsu + erosion |
| `solar-filament-panoptic-morphology-segmentation` | **0.70+** | Mask R-CNN (ResNet-50 FPN v2) + ResNet18 Crop U-Net | Detector 1024×1024; refiner 256×256 crops (context 1.8×) | Frozen weights (repackages Fususu LB 0.70 checkpoints) | Photospheric limb norm ($0.945 R_\odot$); detector score 0.70, refiner thr 0.70; watershed seed partition (dist 8px) | 4-view topology-safe flip TTA (Dice ≥ 0.62, area ratio in [0.60, 1.70]) | Exact clone of Fususu 0.70 pipeline wrapped in Google Earth Engine styled markdown |
| `solar-filament-u-net-oof-0-45-vs-public-0-39` | **0.39** (OOF 0.450) | 10-model U-Net ensemble (5× ResNet34 + 5× EfficientNetV2-S via smp) | Native 2048×2048 full resolution | Standard smp U-Net loss on native 2048 | Solar disk norm (radius 950px); 8-connected components on uint8 map: candidate if $p > 0.35$; kept if peak $p > 0.99$, mean $p \ge 0.85$, area $\ge 200$ | 8-view TTA (D4: 4 rot90 × 2 flips); 10-model ensemble | Sparse-label model ($r \approx 0.80$) explains 0.45 vs 0.39 gap; human agreement PQ = 0.333 |
| `solar-filament-unet-segmentation-0-55` | **0.55** | Fake U-Net; actual inference uncompresses `CHAMPION_PAYLOAD` | Native 2048×2048 (in payload) | Code defines BCE + Soft Dice (unused) | Loads precomputed CSV table from base85 LZMA string; fallback disk anchor `[1000:1048, 1000:1048]` if test stem missing | None | **Payload Leak**: Bypasses inference entirely and writes hardcoded precomputed predictions |
| `yolo-u-net-solar-filament-segmentation` | None (~0.38) | 2-stage: YOLO11s-seg + Crop U-Net refiner | YOLO 1024×1024; refiner on detected instance crops | YOLO seg loss; Refiner 0.5 BCE + 0.5 Soft Dice | YOLO conf 0.25; refiner thr 0.65; min component size 20; RLE encoding | None | Visual overlay audits vs ground truth; local crop bounding-box refinement |

---

## 2. Why the High Scores (LB 0.55–0.70)

The public leaderboard top scores (0.55–0.70) stem from five concrete factors:

1. **Pre-August 7 Evaluator Vulnerability (Empty & Under-Prediction Bug)**:
   - `please-fix-the-leaderboard` proved the preliminary Kaggle evaluator completely omitted unpredicted test images and credited background true negatives:
     ```python
     # please-fix-the-leaderboard (lines 8, 46-56, 85)
     empty_512 = np.zeros((512, 512), dtype=np.uint8)
     empty_rle = mask_utils.encode(np.asfortranarray(empty_512))
     empty_counts = empty_rle['counts'].decode('ascii')  # 'PPP8'
     submission = pd.DataFrame({'filament_id': [f'{img}_1' for img in selected_5_ids], 'segmentation_rle': [empty_counts]*5})
     # "one empty mask for every test image scored 0.88, while seven empty masks per image scored 0.70.
     #  Restricting the file to a small subset produced a perfect 1.00."
     ```
   - `lb-0-70-topology-safe-solar-filaments-handoff` was submitted on **5 August 2026** (two days before the metric update) with exactly 1,299 instances across 180 images (an average of **~7.2 instances per image**), hitting the exact ~0.70 sweet spot of the unpenalized background evaluation artifact.

2. **Hardcoded Prediction Dumping**:
   - `solar-filament-unet-segmentation-0-55` achieved 0.55 by embedding an LZMA-compressed, base85-encoded precomputed CSV payload, completely bypassing neural network execution:
     ```python
     # solar-filament-unet-segmentation-0-55 (lines 312, 2005-2007)
     CHAMPION_PAYLOAD = ('{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;;|a<{#^hw8cEh=6VoInPp^DXs7ym<mp&J~...' ...)
     raw_bytes = lzma.decompress(base64.b85decode(CHAMPION_PAYLOAD.encode('ascii')))
     champion_df = pd.read_csv(io.BytesIO(raw_bytes))
     ```

3. **Detector + Native Crop Refiner Cascade**:
   - Rather than predicting on heavily downsampled images, top models use a detector (YOLO11s or Mask R-CNN) at 1024×1024 to propose bounding boxes, followed by a U-Net refiner on native-resolution crops:
     ```python
     # lb-0-70-topology-safe-solar-filaments-handoff (lines 110-114)
     IMAGE_SIZE = 1024; DETECTOR_SCORE_THRESHOLD = 0.70; DETECTOR_MASK_THRESHOLD = 0.50
     CANDIDATE_REFINER_THRESHOLD = 0.70; MAXIMUM_SEED_IOU = 0.10; MINIMUM_SEED_DISTANCE = 8.0
     ```

4. **Topology-Safe TTA**:
   - Test-time flip averaging often blurs and severs thin 1–3 pixel filament spines. Fususu implemented topological gating:
     ```python
     # lb-0-70-topology-safe-solar-filaments-handoff (lines 118-123)
     REFINER_TTA_TRANSFORMS = ("identity", "hflip", "vflip", "hvflip")
     TTA_AREA_RATIO_MIN = 0.60; TTA_AREA_RATIO_MAX = 1.70; TTA_DICE_MIN = 0.62
     # TTA adopted only when area ratio and Dice vs single-view pass guards; 9 fragile tasks fell back
     ```

5. **Solar Disk Chromosphere Normalization & Limb Masking**:
   - High-gradient limb darkening triggers hundreds of false-positive detections. High-scoring pipelines isolate the solar disk:
     ```python
     # solar-filament-u-net-oof-0-45-vs-public-0-39 (lines 77, 95)
     DISK_R = 950  # px around center; normalize with statistics strictly inside DISK
     DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2
     ```
     `solar-filament-grouped-cv` applied disk CLAHE (`clip_limit=2.0, grid=[8,8]`) and Otsu off-limb masking.

---

## 3. Leaderboard & Metric Analysis

### The Metric Mismatch: Evaluator Bug vs. Official Panoptic Quality (PQ)
1. **The August 7 Metric Shift**:
   - On 7 August 2026, the competition officially updated its evaluation from Dice to **Panoptic Quality (PQ)**:
     $$PQ = \frac{\sum_{(y, \hat{y}) \in TP} \text{IoU}(y, \hat{y})}{|TP| + 0.5|FP| + 0.5|FN|}$$
     where a predicted instance $\hat{y}$ is matched to ground truth $y$ as a True Positive **only if $\text{IoU}(y, \hat{y}) > 0.5$**.
2. **What the LB Actually Rewarded vs. What PQ Rewards**:
   - **Old Flawed Evaluator**: Background true negatives were rewarded, and unsubmitted test images incurred no penalty. This allowed empty masks to score 0.88–1.00 and sparse submissions (~7 masks/image) to score 0.66–0.70.
   - **Official PQ**: Strongly penalizes both false positives and false negatives ($0.5$ per instance in the denominator). Any prediction with $\text{IoU} \le 0.5$ counts as an FP (contributing 0 to numerator, adding 0.5 to denominator). Over-merging two filaments into one typically makes IoU < 0.5 for both, creating 1 FP and 2 FNs. Over-segmenting a filament creates at least one FP.
3. **Are 0.70 Scores Reproducible as Real Segmentation Quality?**
   - **NO.** Two human expert annotators outlining the same GONG H-alpha observations achieve an inter-annotator agreement of only **PQ = 0.333–0.340** (`solar-filament-u-net-oof-0-45-vs-public-0-39`, line 418).
   - An honest model cannot achieve PQ 0.70 because the annotators themselves disagree on fine barbs and faint endpoints.
   - The best honest out-of-fold scores in the competition are **OOF PQ 0.448–0.450** (`magfilo-v1`, `solar-filament-u-net-oof-0-45`).
   - Dariush Afshar's sparse-label derivation demonstrated that test images were labeled ~20% more sparsely than training images ($r \approx 0.80$), so even a model with nested OOF PQ of 0.450 reliably lands at **Public LB 0.39**:
     $$PQ' = \frac{r \sum \text{IoU}}{r\,TP + 0.5\,FP + 0.5(1-r)\,TP + 0.5\,r\,FN}$$
   - Any public LB score above ~0.50 is either a legacy score from the pre-Aug 7 broken evaluator, an empty-mask exploit, or a hardcoded submission leak. On the true PQ metric, **0.39–0.45 represents top-tier, human-beating performance**.

---

## 4. Ranked Changes to Raise OUR Pipeline's LB Score

Our pipeline achieves **Local Val PQ 0.449** (competitive with Dariush's 0.450 OOF), but previously scored **0.30, 0.25, and 0.02** on the public LB. Here is the ranked action plan:

### 1. Drastically Tighten Post-Processing Keep Rules (Suppress FPs)
- **Change**: Replace loose hysteresis (`t_low 0.15 / t_high 0.40, min_area 50, tau 0.35`) with strict peak, mean, and area thresholds:
  `t_lo = 0.35`, `t_hi = 0.99`, `min_meanp = 0.85`, `min_area = 200`.
- **Reference**: `solar-filament-u-net-oof-0-45-vs-public-0-39` (Dariush Afshar, lines 74, 255–257, 436–459).
- **Expected Effect**: **+0.09 to +0.14 LB** (moves LB from 0.25–0.30 to ~0.39–0.41).
- **Effort**: Minimal (~5 lines in post-processing, 0 retraining).
- **Risk**: Very low (Legit). Because test labels are sparse ($r \approx 0.80$), low thresholds produce dozens of extra faint predictions that become unmatched False Positives, destroying the PQ denominator.

### 2. Remove Morphological Closing ($r=6$) and Hole Filling
- **Change**: Disable `binary_closing(structure=disk(6))` and `binary_fill_holes`. Rely strictly on 8-connected component labeling.
- **Reference**: `solar-filament-grouped-cv-and-strategy-writeup` (lines 35, 50) and `new-metric-solar-filament` (line 32).
- **Expected Effect**: **+0.04 to +0.07 LB**.
- **Effort**: Minimal (delete 2 morphological operations).
- **Risk**: Very low (Legit). Solar filaments are thin, elongated open curves, not solid blobs. Closing with radius 6 bridges nearby parallel filaments into single giant components. Under PQ ($IoU > 0.5$), merged filaments fail the match threshold for both ground-truth instances, creating 1 FP and 2 FNs simultaneously.

### 3. Mask Off-Limb Regions & Normalize Strictly Inside the Solar Disk
- **Change**: Restrict normalization statistics and prediction output to the solar disk ($r \le 950$ px on 2048×2048 from center). Zero out everything outside.
- **Reference**: `solar-filament-u-net-oof-0-45-vs-public-0-39` (lines 77, 94–96, 104–106) and `simple-dark-filament-segmentation-0-46` (line 97).
- **Expected Effect**: **+0.02 to +0.04 LB**.
- **Effort**: Low (~10 lines of NumPy/OpenCV masking).
- **Risk**: Very low (Legit). Eliminates coronagraph border artifacts, limb darkening gradients, and off-limb noise that produce spurious high-confidence false alarms.

### 4. Audit Submission Schema, 2048×2048 Coordinate Grid & Fortran RLE Contiguity
- **Change**: Run an automated pre-submission assertion suite: verify 180 unique test image IDs, Fortran-contiguous column-major pycocotools encoding, zero overlapping pixels across instances, and exact 2048×2048 reconstruction.
- **Reference**: `solar-filament-u-net-oof-0-45-vs-public-0-39` (lines 291–358 `validate_submission`) and `0-9-solar-filament-baseline-pipeline-u-net`.
- **Expected Effect**: **Guarantees recovery from catastrophic 0.02 failure**.
- **Effort**: Low (~30 lines validation script).
- **Risk**: Zero (Essential bug prevention). Our 0.02 LB score was almost certainly caused by an RLE transpose error (C-order vs Fortran-order) or encoding a crop/resized dimension directly like `0-9-baseline`.

### 5. Calibrate Instance Density per Test Image (~7–8 Filaments per Disk)
- **Change**: Tune the rescorer threshold $\tau$ or apply an adaptive top-K instance filter so that the final submission outputs an average of 6.5–8.0 filament instances per test image.
- **Reference**: `solar-filament-u-net-oof-0-45-vs-public-0-39` (lines 411–423) and `lb-0-70-topology-safe-solar-filaments-handoff` (1,299 instances / 180 images = 7.2 instances/image).
- **Expected Effect**: **+0.02 to +0.03 LB**.
- **Effort**: Low (simple calibration on validation output).
- **Risk**: Low (Legit). Prevents submitting excessive filament counts on complex active regions while ensuring empty/sparse test images do not get flooded with false positives.
