# Solar Filament Segmentation Challenge 2026 — notes for future AI agents

Source of truth: Kaggle pages (Overview / Data / Evaluation) pasted by the user on 2026-11-10. Goal: top 3 in ~5 days.
Competition: https://www.kaggle.com/competitions/filament-segmentation-2026

## Deadlines (CRITICAL)
- Launched 2026-07-10. **Final deadline: 2026-11-15, 11:59 pm CDT** (reports + solution). Winners announced 2026-11-30 (IEEE BigData, Phoenix, Dec 14-17).
- Final submission = Google form (link on the Kaggle Overview "Final Submission" section) with:
  1. 4-page technical report, single PDF, using the organizers' Overleaf template (covers preprocessing, features, architecture, training, post-processing).
  2. Publicly accessible Git repo (GitHub/GitLab/Bitbucket), no access approval needed, public at deadline and until winners announced. Must contain `requirements.txt` (all packages + versions)
     and a Jupyter notebook showing the ENTIRE pipeline. **Model weights must be stored in the repo** and loaded in the notebook; reproduce predictions WITHOUT retraining. No extra files needed.
- Incomplete/inaccessible form info => ineligible. (Repo must be public: our global rule says new repos are private -> flip to public at submission.)

## Task
Instance-level segmentation of solar filaments (dark structures) in GONG H-Alpha images. Capture fine structure (barbs), suppress noise, keep each filament contiguous.
Challenges: thin barbs, noise/artifacts (ground-based), fragmentation ("islands") and over-merging.

## Data (already downloaded: /home/Jemin/Projects/solar/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026, ~719 MB)
```
MAGFiLO_1.0_Kaggle_2026/
  train/train_images/*.jpeg
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json   (COCO style)
  test/test_images/*.jpeg
```
- Images: 2048x2048, **grayscale 8-bit JPEG** (converted from FITS) — do NOT treat as RGB (load as single channel).
- Filename `YYYYMMDDHHMMSSII.jpeg`; II = instrument code (e.g. Bh = Big Bear; seen: Bh, Ch, Lh, Mh, Th, Uh). Time/instrument encoded -> usable for date-based splits (avoid leakage between near-in-time images) and per-site analysis.
- Same image may be annotated by multiple annotators independently (image id like `010101-20260901165702Bh`, `010102-...`; id = annotator-batch + image name). Annotators disagree => the same file can appear several times with different masks;
  group by file_name when splitting train/val; consider the union/average of annotators for soft targets.
- COCO json: `info, images, annotations, licenses, categories`. Annotation: `id` (uuid str), `image_id` (str), `category_id`, `segmentation` ([[x0,y0,...]] single closed polygon per filament),
  `area`, `spine` (centerline path), `bbox [x,y,w,h]`, `iscrowd`=0 always. Categories: 1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous (chirality classes; competition uses only masks).
- Use `pycocotools` (annToMask, decode, encode). Polygon<->RLE is lossless.
- Test images only come with images (no GT). ~50% of test images drive the public leaderboard; rest hidden (private).

## Submission format (CSV)
Columns `filament_id`, `segmentation_rle`. **One row per predicted filament (instance)**.
- `filament_id` = `<image_name_without_ext>_<k>` (e.g. `20150125172714Mh_2`); suffix just needs to be unique, image part must match the test image.
- `segmentation_rle` = pycocotools compressed RLE **counts string only** (no size; size fixed 2048x2048), no quotes. Use `encodeMask` -> `counts` (decode bytes to str).
- Number of predicted instances need not match GT; matching is by overlap. => need instance separation (e.g. connected components / instance head), not just a semantic mask.
- Pitfall: edge-case verification messages may appear after upload; images with no predictions presumably need handling (check what happens with zero rows).

## Metric / scoring
- **Leaderboard ranking: Panoptic Quality (PQ)** (Kirillov et al., CVPR 2019): PQ = sum IoU over TP / (|TP| + 0.5|FP| + 0.5|FN|), match = IoU > 0.5. Also reports mean Dice (torchmetrics.segmentation.DiceScore) — the page text is
  inconsistent about which is "primary"; leaderboard shows mean Dice for ~50% of test per one line, PQ per the ranking section. Optimize BOTH; instance quality (no fragmentation/merging) drives PQ.
- Announcements: organizers say PQ > 0.35 is "of great value" (Aug 20); Oct 23 note says "PQ greater than 3.5" (almost certainly a typo for 0.35). Realistic leaderboard scale is low. They warn about people "hacking the system" — don't.
- Self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook (use it locally to mimic scoring).
- **Final judging: 70% quantitative** (mean Dice / PQ, Dice distribution, IoU distribution, distribution of one-to-many and many-to-one GT<->pred relations; plus end-to-end computational efficiency)
  **+ 30% qualitative** (pipeline description, visual morphology of predicted segmentations, code quality: modularity + documentation). Leaderboard is only ONE consideration. Top-3 chosen by committee score.
- Prize pool up to $3,000 split among top 3 proportional to score^2 (`compute_prizes`, temperature 2). Split equally among verified team members. Needs bank info/US payments, full source release, tech report.

## Rules / constraints
- Any approach allowed (classical, ML, DL). External data allowed, BUT: at inference only the provided test H-alpha images may be used, and models may not use any other ground-truth metadata for training.
- Must be reproducible from the repo (weights included). Organizers get perpetual non-exclusive license to the code; authorship stays with us.
- Keep inference efficient (runtime is scored) and report timing.

## Related literature
- MAGFiLO dataset paper: doi 10.1038/s41597-024-03876-y. Organizer: Azim Ahmadzadeh (UMSL).
- "A Modern ConvNet for Solar Filament Detection" arXiv 2607.24525 (MORDEN; DenseCRF edge refinement + DBSCAN clustering post-processing; datasets MHAS/AHAS). Code/data links not found.
- Optional workshop: SABID 2026 (IEEE BigData) https://sites.google.com/view/sabid2026/home

## Local environment
- GPU: RTX 4050 Laptop 6 GB VRAM, 14 GB RAM, `/` 89% full (25 GB free). Use AMP, crops/tiles (2048^2 is big), small encoders; heavier training may need Kaggle/Colab GPU.
- No git repo yet in /home/Jemin/Projects/solar. Git identity: jemin-023, `user.name "Jemin Morabiya"`, `user.email "jeminmorabiyawork@gmail.com"`. Python reading dataset files: run with `-I`.
- Desktop session logged out once at 19:57 on 2026-11-10 (clean logout, not a crash); long jobs should checkpoint and run detached (nohup/tmux) in case it recurs.

## Plan (hypotheses, not facts)
1. EDA: count images/annotations per annotator, filaments per image, area/thin-ness distribution, instruments, date ranges (check train/test date overlap), annotator disagreement (IoU between annotators of same image).
2. Split by date/image (grouped by file_name). Baseline U-Net/UNet++/SegFormer-style with pretrained encoder on 1-channel input (replicate or adapt first conv), Dice+BCE/Focal (+clDice/boundary term for thin structures), 512-1024 crops, flips/rot augment.
3. Instances: connected components on thresholded probability map with morphological closing to bridge gaps (reduce fragmentation) and min-area filter; tune threshold/min-area against local PQ using the self-eval notebook logic. TTA flips. Consider watershed/DBSCAN for merged blobs.
4. Track runtime; write modular repo + notebook + requirements.txt + weights early; draft 4-page report on the Overleaf template as we go.
