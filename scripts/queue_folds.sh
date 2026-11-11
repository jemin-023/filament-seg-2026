#!/usr/bin/env bash
# Train fold 1-4 models for both recipes (alternating), caching val + test probs after each. Resumable.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
QUIET=${QUIET---workers 3 --throttle 0.8}   # quiet/low-power mode (override: QUIET="" for full speed)
run() {  # name arch fold extra...
  local name=$1 arch=$2 fold=$3; shift 3
  if [ ! -f runs/$name/done ]; then
    echo "[$(date)] train $name"
    nice -n 10 .venv/bin/python -m filseg.train --name $name --arch $arch --fold $fold "$@" $QUIET > runs/$name.out 2>&1 && touch runs/$name/done
  fi
  local tag=${name%%_*}_f$fold
  [ -d cache/probs/$tag ] || .venv/bin/python scripts/cache_probs.py --ckpt runs/$name/best.pt --name $tag --split val --fold $fold --tta id h
  [ -d cache/probs/${tag}_test ] || .venv/bin/python scripts/cache_probs.py --ckpt runs/$name/best.pt --name ${tag}_test --split test --tta id h
}
for f in 1 2 3 4; do
  run r34_full1024_f$f r34 $f --mode full --size 1024 --bs 2 --spi 2 --epochs 30
  run effv2s_c768_f$f effv2s $f --mode crop --crop 768 --bs 3 --spi 4 --epochs 30
done
echo "[$(date)] queue_folds done"
