#!/usr/bin/env bash
# Third ensemble member (effv2s, 1024 crops, seed 1) on the folds Kaggle is not training. Resumable.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for f in ${FOLDS:-2 3 4}; do
  name=effv2sB_c1024_f$f
  if [ ! -f runs/$name/done ]; then
    echo "[$(date)] train $name"
    .venv/bin/python -m filseg.train --name $name --arch effv2s --fold $f --mode crop --crop 1024 --bs 2 --spi 4 \
      --epochs 30 --seed 1 --workers 4 > runs/$name.out 2>&1 && touch runs/$name/done
  fi
  [ -d cache/probs/effv2sB_f$f ] || .venv/bin/python scripts/cache_probs.py --ckpt runs/$name/best.pt --name effv2sB_f$f --split val --fold $f --tta id h
  [ -d cache/probs/effv2sB_f${f}_test ] || .venv/bin/python scripts/cache_probs.py --ckpt runs/$name/best.pt --name effv2sB_f${f}_test --split test --tta id h
done
echo "[$(date)] queue_effB done"
