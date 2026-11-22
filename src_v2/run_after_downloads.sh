#!/usr/bin/env bash
# Hands-off chain: wait for the running downloads, re-run them until every station x product (and
# station-year of SCAN) is on disk (both skip what is cached; retries wait for the network to come
# back), then run the full analysis offline (run_v2.sh).
cd "$(dirname "$0")"
PY="${PY:-../../.venv/Scripts/python.exe}"
RAW=../data/raw_v2
PROG=../results_v2/progress.md
FLAG=../results_v2/.pipeline_done
rm -f "$FLAG"
"$PY" keep_awake.py "$FLAG" &

until grep -q "finished:" $RAW/ee_cache/extract.log && grep -q "^done" $RAW/scan_run.out; do sleep 30; done

complete() {
  "$PY" - <<'PY'
import glob, sys, pandas as pd
st = pd.read_csv("../data/raw_v2/stations_v2.csv", dtype={"sid": str})
ALL = ["static", "s1", "s2", "chirps", "mod13", "nbar", "lai", "lstT", "lstA", "era5l", "smapl3e", "smapl4"]
PAPER = ["static", "s1", "s2", "chirps", "mod13"]
need = sum(len(PAPER if s == "paper_coords" else ALL) for s in st["set"])
have = len(glob.glob("../data/raw_v2/ee_cache/*/*.parquet"))
scan_need = (st["set"] == "scan").sum() * 5
scan_have = len(glob.glob("../data/raw_v2/scan_hourly/*.parquet"))
print(f"EE {have}/{need}, SCAN {scan_have}/{scan_need}")
sys.exit(0 if have >= need and scan_have >= scan_need else 1)
PY
}

for attempt in $(seq 1 24); do
  msg=$(complete) && { echo "- $(date +%H:%M) downloads complete: $msg" >> $PROG; break; }
  echo "- $(date +%H:%M) downloads incomplete ($msg), retry $attempt (network may be down)" >> $PROG
  "$PY" ../gee/gee_extract_v2.py --stations $RAW/stations_v2.csv --cache $RAW/ee_cache --workers 8 >> $RAW/ee_run.out 2>&1
  "$PY" ../gee/insitu_v2.py scan $RAW/stations_v2.csv $RAW/scan_hourly >> $RAW/scan_run.out 2>&1
  complete > /dev/null && continue
  sleep 600
done
msg=$(complete) || echo "- $(date +%H:%M) WARNING: running the analysis with incomplete downloads ($msg)" >> $PROG

echo "- $(date +%H:%M) analysis pipeline started (offline)" >> $PROG
export HF_HUB_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
bash run_v2.sh || echo "- $(date +%H:%M) PIPELINE STOPPED WITH AN ERROR - see results_v2/logs/" >> $PROG
touch "$FLAG"
