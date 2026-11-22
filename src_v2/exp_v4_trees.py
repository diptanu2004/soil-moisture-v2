"""
Experiments v4-1 / v4-2 (pre-registered 2026-11-22, see progress.md): v2 tree ensemble + POLARIS, with
  --polaris {approx,exact}   v3 POLARIS table (stations.csv coordinates) or the one re-extracted at the
                             NRCS AWDB coordinates (data/raw_v2/polaris_stations_exact.parquet)
  --target  {raw,se}         raw: sm_5cm as in v2/v3.  se: relative wetness
                             Se = (sm/100 - theta_r) / (theta_s - theta_r), POLARIS 0-5 cm window means
                             (fallback where POLARIS is missing: theta_s = SoilGrids Saxton-Rawls theta_sat,
                             theta_r = 0); denominator floored at 0.05. Predictions are back-transformed to % vol.
theta_s / theta_r are static map values (no in-situ input); the target transform uses only them.
Same fit_predict_trees(paper=False), seeds and folds as v2-2 / v3-3. Per-fold checkpoints.

Writes results_v2/tables/v4_trees_<polaris>_<target>_<tag>.parquet  (site, date, <column name>)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common_v2 import IMPROVED_V2, PROC, TAB, TARGET, Timer, fit_predict_trees, folds_for, log_progress
from exp_v2_2_protocols import load_eval_rows

PROTOS = ["leave-stations-out", "leave-one-region-out"]
FILES = {"approx": "polaris_stations.parquet", "exact": "polaris_stations_exact.parquet"}


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


def colname(polaris, target):
    inner = "POLARIS" + ("(exact" if polaris == "exact" else "(") + (", Se)" if target == "se" else ")")
    return "Trees+" + inner.replace("()", "").replace("(, ", "(")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polaris", choices=list(FILES), required=True)
    ap.add_argument("--target", choices=["raw", "se"], required=True)
    a = ap.parse_args()
    name = colname(a.polaris, a.target)
    df = load_eval_rows()
    pol = pd.read_parquet(os.path.join(os.path.dirname(PROC), "raw_v2", FILES[a.polaris]))
    pcols = [c for c in pol.columns if c != "site"]
    df = df.merge(pol, on="site", how="left")
    X = df[IMPROVED_V2 + pcols].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    ts = df["pol_theta_s_0_5_win"].fillna(df["theta_sat"] / 100).to_numpy(float)
    tr_ = df["pol_theta_r_0_5_win"].fillna(0.0).to_numpy(float)
    span = np.maximum(ts - tr_, 0.05)
    yfit = (y / 100 - tr_) / span if a.target == "se" else y
    back = (lambda p, i: 100 * (tr_[i] + p * span[i])) if a.target == "se" else (lambda p, i: p)
    print(f"{name}: {len(pcols)} POLARIS columns, target {a.target}, "
          f"Se range [{np.nanmin(yfit):.2f}, {np.nanmax(yfit):.2f}]" if a.target == "se" else name, flush=True)
    t = Timer()
    log_progress(f"v4 trees {name} started")
    for proto in PROTOS:
        ck = os.path.join(TAB, "v4_trees_ck", f"{a.polaris}_{a.target}", tag(proto))
        os.makedirs(ck, exist_ok=True)
        folds = folds_for(proto, df)
        for f, (tr, te) in enumerate(folds):
            fn = os.path.join(ck, f"f{f}.npy")
            if not os.path.exists(fn):
                p = fit_predict_trees(X[tr], yfit[tr], X[te], paper=False)["Ensemble"]
                np.save(fn, np.c_[te, back(p, te)])
            print(f"{name} {proto} fold {f} ({t()})", flush=True)
        P = np.full(len(df), np.nan)
        for f in range(len(folds)):
            arr = np.load(os.path.join(ck, f"f{f}.npy"))
            P[arr[:, 0].astype(int)] = arr[:, 1]
        out = df[["site", "date"]].copy()
        out[name] = P
        out.to_parquet(os.path.join(TAB, f"v4_trees_{a.polaris}_{a.target}_{tag(proto)}.parquet"))
        log_progress(f"v4 trees {name} {proto} done ({t()})")


if __name__ == "__main__":
    main()
