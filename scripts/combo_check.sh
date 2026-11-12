#!/usr/bin/env bash
# 2-fold honest comparison of ensemble member sets on folds 0,1.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for combo in "effv2s effv2sB r34" "effv2s effv2sB" "effv2sB"; do
  tag=c_$(echo $combo | tr ' ' '_')
  for k in 0 1; do
    src=""; for m in $combo; do src="$src ${m}_f$k"; done
    [ -d cache/probs/${tag}_f$k ] || .venv/bin/python scripts/ens_probs.py --src $src --out ${tag}_f$k
  done
  nice -n 15 .venv/bin/python scripts/honest_eval.py --tmpl "${tag}_f{}" --folds 0 1 --trials 25 --jobs 1 \
    --out runs/honest_${tag}_2fold.json > runs/honest_${tag}_2fold.out 2>&1
  echo "$combo: $(grep POOLED runs/honest_${tag}_2fold.out | tr '\n' ' ')"
done
echo "combo_check done"
