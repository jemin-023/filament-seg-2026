#!/usr/bin/env bash
# Final fit + submission only (honest eval already done). Usage: ENS=ens_effv2s_r34 OUT=v2_ens10 scripts/final_fit_submit.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
nice -n 10 .venv/bin/python scripts/honest_eval.py --tmpl "${ENS}_f{}" --folds 0 1 2 3 4 --trials ${TRIALS:-25} --jobs ${JOBS:-2} \
  --final > runs/final_fit_${ENS}.out 2>&1
tail -3 runs/final_fit_${ENS}.out
use=$(.venv/bin/python -c "import json; r=json.load(open('runs/honest_${ENS}_5fold.json')); print('resc' if r['pooled_resc']['pq'] > r['pooled_pp']['pq'] else 'pp')")
cp weights/rescorer.pkl runs/rescorer_${ENS}.pkl; cp configs/pp_final.json runs/pp_final_${ENS}.json
if [ "$use" = resc ]; then
  .venv/bin/python scripts/make_submission.py --probs ${ENS}_test --rescorer weights/rescorer.pkl --out submissions/$OUT.csv
else
  .venv/bin/python scripts/make_submission.py --probs ${ENS}_test --pp configs/pp_final.json --out submissions/$OUT.csv
fi
echo "[$(date)] done -> submissions/$OUT.csv ($use)"
