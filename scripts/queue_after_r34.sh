#!/usr/bin/env bash
# Wait for r34 training (PID given as $1), then cache its fold-0 val and test probs for ensembling.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[$(date)] r34 done; caching"
.venv/bin/python scripts/cache_probs.py --ckpt runs/r34_full1024_f0/best.pt --name r34_f0 --split val --fold 0 --tta id h
.venv/bin/python scripts/cache_probs.py --ckpt runs/r34_full1024_f0/best.pt --name r34_f0_test --split test --tta id h
echo "[$(date)] queue done"
