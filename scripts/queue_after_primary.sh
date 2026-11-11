#!/usr/bin/env bash
# Cache primary-run probabilities, then train the second model. Run with nohup.
# (Earlier version waited via `pgrep -f`, which matched its own launcher shell and hung forever.)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Wait for the primary run only if it is genuinely running (match python processes only).
while pgrep -f "^[^ ]*python[^ ]* -m filseg.train --name effv2s_c768_f0" > /dev/null; do sleep 30; done
echo "[$(date)] cache val"; t0=$(date +%s)
.venv/bin/python scripts/cache_probs.py --ckpt runs/effv2s_c768_f0/best.pt --name effv2s_f0 --split val --fold 0 --tta id h
echo "[$(date)] val cache took $(( $(date +%s) - t0 ))s"; t0=$(date +%s)
.venv/bin/python scripts/cache_probs.py --ckpt runs/effv2s_c768_f0/best.pt --name effv2s_f0_test --split test --tta id h
echo "[$(date)] test cache took $(( $(date +%s) - t0 ))s"
.venv/bin/python -m filseg.train --name r34_full1024_f0 --arch r34 --mode full --size 1024 --bs 2 --epochs 30 --spi 2 --val-every 3
echo "[$(date)] queue done"
