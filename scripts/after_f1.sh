#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until [ -d cache/probs/effv2s_f1_test ] && [ $(ls cache/probs/effv2s_f1_test | wc -l) -ge 180 ]; do sleep 60; done
sleep 20
.venv/bin/python scripts/ens_probs.py --src effv2s_f1 r34_f1 --out ens2_f1
nice -n 19 .venv/bin/python scripts/honest_eval.py --tmpl "effv2s_f{}" --folds 0 1 --trials 40 --jobs 2 > runs/honest_effv2s.out 2>&1
nice -n 19 .venv/bin/python scripts/honest_eval.py --tmpl "ens2_f{}" --folds 0 1 --trials 40 --jobs 2 > runs/honest_ens2.out 2>&1
echo "[$(date)] after_f1 done" >> runs/after_f1.log
