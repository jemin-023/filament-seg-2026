# Progress note (updated 2026-11-11 18:00 IST)

## 2026-11-11 17:00-18:00 IST (Opus session)
- **v1 WAS submitted** (13:53 IST): public LB **0.38** (in-sample local 0.449, PQ'(r=0.8) estimate 0.401). Calibration done.
- r34_full1024_f0 finished: best raw PQ 0.4087 (ep27). Probs cached (r34_f0, r34_f0_test).
- Ensemble effv2s+r34 (fold 0, mean probs, `scripts/ens_probs.py` -> cache/probs/ens2_f0): pp tuning in progress,
  0.4454 after 24 trials vs 0.4421 single (in-sample). `runs/tune_ens2_f0.out`.
- Training queue `scripts/queue_folds.sh` (log runs/queue_folds.out): r34 + effv2s on folds 1-4, alternating, val+test
  probs cached after each. Resumable (train.py now writes runs/<name>/resume.pt each epoch). Run with setsid (a session
  teardown killed the first launch). ~15 h total.
- LB investigation: current public LB has many post-fix Nov entries at 0.56-0.70. Top-scored public kernels by score:
  please-fix-the-leaderboard, foysal, 0-9-baseline = ALL-EMPTY masks (PPP8) -> evaluator exploit. lamhuy "0.55" = hardcoded
  payload of unknown origin (not used). Kernel outputs in research/kernel_outputs/.
  Cross-PQ of submissions: ours vs dariush (LB 0.39) 0.77 (agree); both vs lamhuy ~0.50; lamhuy masks 1.27x our area, 7.5 inst/img.
- Local val: our pred area = 0.815x GT, 6.63 vs 7.61 inst/entry, yet dilate 0 is PQ-optimal on single- AND multi-annotator
  images (dil+1 -0.022, dil+2 -0.075; scripts/subset_grid.py). Union-of-annotators GT sim (scripts/union_gt_grid.py): optimum
  shifts to dil+1 / 8 inst/img but PQ still ~0.44. => no legit local GT definition explains LB 0.55+.


## State
- Env: `.venv` (Python 3.12, torch 2.15+cu132). Run code with `PYTHONPATH=src .venv/bin/python ...`.
- Scorer `src/filseg/metrics.py` == official self-eval notebook (verified, `tests/test_scorer.py`).
- Targets/folds: `cache/targets/*.png`, `cache/folds.csv` (5 date-grouped folds), `cache/centers.json`.
- Human reference (fold 0, multi-annotated files): PQ 0.333. Oracle consensus pipeline test: PQ 0.80 (leaky, sanity only).
- **Primary run** `runs/effv2s_c768_f0` (EffNetV2-S U-Net, 3 heads, 768 crops, bs3, 30 ep, ~4.25 min/epoch):
  val PQ (v0 postproc) ep3 0.368, ep6 0.388, ep9 0.393, ep12 0.395, ep15 0.406, ep18 0.410 (best.pt), ep21 0.407, ep24 0.404, ep27 0.408.
  FP count noisy (+-50), SQ rising 0.661->0.682. Also cache last.pt (ep30) probs and compare vs best.pt after full pp + rescorer. Log: `runs/runs_effv2s_c768_f0.out`,
  `runs/effv2s_c768_f0/log.jsonl`; checkpoints `best.pt` / `last.pt` every 3 epochs. Expected finish ~03:05 IST.
- **Queued** (`scripts/queue_after_primary.sh`, log `runs/queue.out`): after primary exits ->
  1. cache fold-0 val probs (TTA id+h) -> `cache/probs/effv2s_f0/`
  2. cache test probs -> `cache/probs/effv2s_f0_test/`
  3. train `runs/r34_full1024_f0` (ResNet34, whole disk at 1024, bs2, 30 ep, ~1.6 h)
- Leaderboard: public top ~0.70 PQ but local official-scorer human agreement is 0.33-0.37 -> LB scale differs;
  user must approve ONE calibration submission (not yet approved). Kaggle deadline 2026-11-15 06:00 UTC (11:30 IST).

- 11:40 IST: queue had hung all night (its `pgrep -f` wait matched its own launcher shell). Fixed + relaunched 11:41.
  Primary finished 03:11 (best PQ 0.4098 @ ep18, last ep30 0.4072). Val cache: 141 imgs, 0.64 s/img (2 TTA), 115 s.
  Then test cache, then r34_full1024_f0 training (log `runs/queue.out`, ~1.6 h -> expected ~13:30 IST).
- Tuning running: `runs/tune_effv2s_f0.out` (~10 s/trial, 150 trials).

- 13:30 IST: machine restarted ~12:40 and killed jobs. r34 had reached ep12 (PQ 0.387, kept in runs/r34_full1024_f0_killed_ep12).
  r34 relaunched from scratch 13:20 IST under systemd-inhibit (log `runs/r34_full1024_f0.out`, ~2 h -> ~15:20 IST). No resume support.
- PP tuning (150 trials, in-sample fold 0): PQ 0.4421 (configs/pp_effv2s_f0.json). Rescorer (min_area 50, OOF tau 0.35): PQ 0.4494
  vs 0.4171 un-rescored at min_area 50 (configs/rescorer_effv2s_f0.json, weights/rescorer.pkl). Rescorer kept.
- v1 submission: submissions/v1_effv2s_rescore.csv (effv2s best.pt + tuned pp + rescorer; built from cached test probs via
  `make_submission.py --probs effv2s_f0_test`). Format check OK, 1213 rows, 177/180 images, 6.7 inst/img (val: 6.63 -> no shift).
  NOT submitted; needs user approval. weights/effv2s_c768_f0.pt committed; notebooks/pipeline.ipynb rebuilt (not yet executed end-to-end;
  run it when GPU is free).

## Next steps (in order)
1. User: approve v1 calibration submission (submissions/v1_effv2s_rescore.csv). Repo must be made PUBLIC for final form.
2. r34 epoch 1 done 13:30 IST (3.2 min/ep, ETA ~15:10). `scripts/queue_after_r34.sh` (log runs/queue_r34.out) then caches
   cache/probs/r34_f0 + r34_f0_test automatically. Then ensemble (mean) with effv2s on fold 0, re-tune pp + refit rescorer; keep ensemble only if
   PQ beats 0.4494. Then copy r34 weights to weights/ (the notebook loads every weights/*.pt, so remove it if not used).
3. Execute notebooks/pipeline.ipynb end-to-end on GPU; check its CSV equals the make_submission output.
4. README (agy), report figures.

## PAUSED 2026-11-11 ~18:10 IST (user moving laptop)
- Ensemble pp tuning finished: PQ 0.4501 in-sample (configs/pp_ens2_f0.json) vs 0.4421 single effv2s.
- r34_full1024_f1 stopped at epoch ~10/30; resume.pt saved at the last completed epoch.
- To resume: `setsid nohup systemd-inhibit --what=sleep:shutdown --why="filseg training" scripts/queue_folds.sh >> runs/queue_folds.out 2>&1 < /dev/null &`
  (queue skips finished runs and resumes r34 f1 from resume.pt).

## 2026-11-11 18:40 IST — resumed in QUIET mode (user request)
- Two OOM kills (14 GB RAM): fit_rescorer returning int32 labels with 4 workers + 6 train loader workers. Fixed: labels int16,
  rescorer/tuning run with --jobs 2 + nice 19; train workers 3.
- Quiet mode: `train.py --throttle 0.8` (sleeps 0.8x step time -> GPU ~60% duty, ~29 W avg vs 40 W cap; no sudo for power cap).
  Default in queue_folds.sh via QUIET env; full speed: `QUIET="" scripts/queue_folds.sh`. Runs ~1.8x slower:
  r34 fold ~3 h, effv2s fold ~3.8 h -> all 8 fold runs ~27 h (ETA ~Nov 12 22:00 IST).
- rescore.py: instance_features returns (0, F) for empty labels (crashed fit_rescorer before).
- Ensemble effv2s+r34 fold 0 + rescorer (tau 0.35): PQ 0.4523 (in-sample) vs 0.4494 single -> +0.003, within noise.
  cache/probs/ens2_f0/_tuning/rescorer.pkl. Needs multi-fold honest eval before adopting.
