#!/usr/bin/env bash
# Wait for the primary run to finish, then cache its probabilities and start the second model.
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "filseg.train --name effv2s_c768_f0" > /dev/null; do sleep 30; done
.venv/bin/python scripts/cache_probs.py --ckpt runs/effv2s_c768_f0/best.pt --name effv2s_f0 --split val --fold 0 --tta id h
.venv/bin/python scripts/cache_probs.py --ckpt runs/effv2s_c768_f0/best.pt --name effv2s_f0_test --split test --tta id h
.venv/bin/python -m filseg.train --name r34_full1024_f0 --arch r34 --mode full --size 1024 --bs 2 --epochs 30 --spi 2 --val-every 3
