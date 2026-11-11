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

## Next steps (in order)
1. (done) queue fixed and running.
2. `scripts/tune_postproc.py --name effv2s_f0 --fold 0 --trials 150 --jobs 4` -> `cache/probs/effv2s_f0/_tuning/best.json`.
3. `scripts/fit_rescorer.py --name effv2s_f0 --fold 0 --pp cache/probs/effv2s_f0/_tuning/best.json --min-area 50`
   -> `weights/rescorer.pkl`; compare PQ vs no rescoring.
4. Build submission CSV: `scripts/make_submission.py --ckpt runs/effv2s_c768_f0/best.pt --rescorer weights/rescorer.pkl --tta id h --out submissions/v1.csv`
   (uses `src/filseg/pipeline.py`; `--pp best.json` instead of `--rescorer` if rescoring doesn't help) -> ask user before submitting.
   Notebook: `scripts/build_notebook.py` -> `notebooks/pipeline.ipynb` (loads all `weights/*.pt` + `weights/rescorer.pkl`;
   copy final checkpoints into `weights/` as fp16). Folds copied to `data/folds.csv` for the repo.
5. When r34 finishes: ensemble probs (effv2s + r34) on fold 0, re-tune, decide final ensemble.
6. Notebook, README, requirements.txt pin, report figures.
