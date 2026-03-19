#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."   # cd to transfer/

BASE=inputs/open_h/nonexpert-2026-01-15_16-50-48
OUT=outputs/open_h/01_2026_03_18/nonexpert-2026-01-15_16-50-48
LOG=$OUT/run_ep043.log

mkdir -p $OUT

run() {
    local label=$1; shift
    echo "[$(date '+%H:%M:%S')] Starting $label" | tee -a $LOG
    .venv/bin/python examples/inference.py "$@" >> $LOG 2>&1
    echo "[$(date '+%H:%M:%S')] Done $label" | tee -a $LOG
}

run edge  -i $BASE/ep043_edge.json  -o $OUT edge
run depth -i $BASE/ep043_depth.json -o $OUT depth
run vis   -i $BASE/ep043_vis.json   -o $OUT vis
run seg   -i $BASE/ep043_seg.jsonl  -o $OUT seg
run multi -i $BASE/ep043_multi.json -o $OUT depth

echo "[$(date '+%H:%M:%S')] All done" | tee -a $LOG
