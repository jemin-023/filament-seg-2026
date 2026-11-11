# Progress note (updated 2026-11-11 11:50 IST)

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
2. When r34 finishes: cache r34 val+test probs, ensemble (mean) with effv2s on fold 0, re-tune pp + refit rescorer; keep ensemble only if
   PQ beats 0.4494. Then copy r34 weights to weights/ (the notebook loads every weights/*.pt, so remove it if not used).
3. Execute notebooks/pipeline.ipynb end-to-end on GPU; check its CSV equals the make_submission output.
4. README (agy), report figures.
