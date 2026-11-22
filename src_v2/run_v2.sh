#!/usr/bin/env bash
# Full v2 analysis after the downloads.
#   assemble -> [v2-1 | v2-2 | v2-5 (GPU)] -> v2-3 -> v2-5 anchor interp -> v2-4 -> audit, summary, figures
# A failing step is logged to results_v2/progress.md and the remaining steps still run.
cd "$(dirname "$0")"
PY="${PY:-../../.venv/Scripts/python.exe}"
LOG=../results_v2/logs
PROG=../results_v2/progress.md
mkdir -p "$LOG"

step() {  # step <name> <command...>
  local name=$1; shift
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "- $(date +%H:%M) step $name ok" >> $PROG
  else
    echo "- $(date +%H:%M) STEP $name FAILED (see results_v2/logs/$name.log)" >> $PROG
  fi
}

(cd ../gee && step assemble "$PY" assemble_v2.py --cache ../data/raw_v2/ee_cache --scan ../data/raw_v2/scan_hourly \
    --stations ../data/raw_v2/stations_v2.csv --out ../data/processed_v2)

step v2_1 "$PY" exp_v2_1_before.py &
P1=$!
step v2_5 "$PY" exp_v2_5_daily.py &
P5=$!
step v2_2 "$PY" exp_v2_2_protocols.py
step v2_3 "$PY" exp_v2_3_models.py
wait $P5
step v2_5_interp "$PY" exp_v2_5_daily.py --interp-only
step v2_4 "$PY" exp_v2_4_fewshot.py
step v2_6 "$PY" exp_v2_6_ablation.py
wait $P1
step dataset "$PY" dataset_stats.py
step audit "$PY" leakage_audit.py
step summary "$PY" summary_v2.py
step paper16 "$PY" paper16_subset.py
step figures "$PY" make_figures_v2.py
echo "- $(date +%H:%M) pipeline finished" >> $PROG
