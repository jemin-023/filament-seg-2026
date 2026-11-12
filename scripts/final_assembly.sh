#!/usr/bin/env bash
# Final assembly from cached OOF/test probs. MEMBERS = model tags (cache/probs/<tag>_f<k>, <tag>_f<k>_test).
# Usage: MEMBERS="effv2s r34" OUT=v2_ens10 scripts/final_assembly.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
MEMBERS=${MEMBERS:-"effv2s r34"}
OUT=${OUT:-v2_ens}
ENS=ens_$(echo $MEMBERS | tr ' ' '_')
TRIALS=${TRIALS:-25}
JOBS=${JOBS:-3}
for k in 0 1 2 3 4; do
  src=""; for m in $MEMBERS; do src="$src ${m}_f$k"; done
  [ -d cache/probs/${ENS}_f$k ] || .venv/bin/python scripts/ens_probs.py --src $src --out ${ENS}_f$k
done
tst=""; for m in $MEMBERS; do for k in 0 1 2 3 4; do tst="$tst ${m}_f${k}_test"; done; done
[ -d cache/probs/${ENS}_test ] || .venv/bin/python scripts/ens_probs.py --src $tst --out ${ENS}_test
echo "[$(date)] honest 5-fold eval"
nice -n 10 .venv/bin/python scripts/honest_eval.py --tmpl "${ENS}_f{}" --folds 0 1 2 3 4 --trials $TRIALS --jobs $JOBS \
  --out runs/honest_${ENS}_5fold.json > runs/honest_${ENS}_5fold.out 2>&1
grep POOLED runs/honest_${ENS}_5fold.out
echo "[$(date)] final fit on all OOF"
nice -n 10 .venv/bin/python scripts/honest_eval.py --tmpl "${ENS}_f{}" --folds 0 1 2 3 4 --trials $TRIALS --jobs $JOBS \
  --final > runs/final_fit_${ENS}.out 2>&1
tail -3 runs/final_fit_${ENS}.out
# keep the rescorer only if it beat pp-only in the honest 5-fold eval
use=$(.venv/bin/python -c "import json; r=json.load(open('runs/honest_${ENS}_5fold.json')); print('resc' if r['pooled_resc']['pq'] > r['pooled_pp']['pq'] else 'pp')")
echo "honest choice: $use"
if [ "$use" = resc ]; then
  .venv/bin/python scripts/make_submission.py --probs ${ENS}_test --rescorer weights/rescorer.pkl --out submissions/$OUT.csv
else
  .venv/bin/python scripts/make_submission.py --probs ${ENS}_test --pp configs/pp_final.json --out submissions/$OUT.csv
fi
echo "[$(date)] final assembly done -> submissions/$OUT.csv ($use)"
