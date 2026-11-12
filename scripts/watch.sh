#!/usr/bin/env bash
# Wake-up watcher: returns when the local queue ends, a Kaggle kernel leaves RUNNING, or after $1 seconds (default 3000).
cd "$(dirname "$0")/.."
KERNEL=${KERNEL:-jeminm/filseg-train-effv2s-f34}
Q=${Q:-queue_folds}   # queue script name (log runs/$Q.out)
end=$(( $(date +%s) + ${1:-3000} ))
while true; do
  sleep 300
  if [ "$Q" = none ]; then q=running
  elif grep -q "$Q done" runs/$Q.out; then q=done
  elif pgrep -f "scripts/$Q[.]sh" >/dev/null; then q=running
  else q=dead; fi
  stop=0; k=""
  for kk in $KERNEL; do   # space-separated list: wake when ANY leaves RUNNING/QUEUED
    s1=$(timeout 60 .venv/bin/kaggle kernels status "$kk" 2>&1 | grep -o "RUNNING\|QUEUED\|COMPLETE\|ERROR\|CANCEL[A-Z]*" | head -1)
    k="$k ${kk##*/}=$s1"
    [ -n "$s1" ] && [ "$s1" != RUNNING ] && [ "$s1" != QUEUED ] && stop=1
  done
  [ "$q" != running ] && break
  [ $stop = 1 ] && break
  [ "$(date +%s)" -ge "$end" ] && break
done
echo "$(date '+%H:%M') queue=$q kaggle=$k"
[ "$Q" != none ] && tail -2 runs/$Q.out
for f in ${LOGS:-runs/effv2sB_c1024_f2.out runs/effv2sB_c1024_f3.out runs/effv2sB_c1024_f4.out runs/final_assembly_ens10.out}; do
  [ -f "$f" ] && echo "$f: $(tail -1 "$f" | cut -c1-80)"
done
free -m | sed -n 2p
