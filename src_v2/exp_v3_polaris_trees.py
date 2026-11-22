"""
Experiment v3-3 (exploratory, added after v3-2 showed the station offset is not predictable from the v2
features): the v2 tree ensemble with POLARIS 30 m soil properties added (static, map-derived, no in-situ input).

Identical to the "improved v2, satellite-only" trees of v2-2 (fit_predict_trees, paper=False, same seeds,
same folds) except for the extra columns, so Trees vs Trees+POLARIS isolates the soil data.
Per-fold checkpoints; resumable.

Writes results_v2/tables/v3_polaris_trees_<tag>.parquet  (site, date, Trees+POLARIS)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_v2 import IMPROVED_V2, TAB, TARGET, Timer, fit_predict_trees, folds_for, log_progress
from exp_v2_2_protocols import load_eval_rows
from exp_v3_station_mean import POLARIS

PROTOS = ["leave-stations-out", "leave-one-region-out"]


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


def main():
    df = load_eval_rows()
    pol = pd.read_parquet(POLARIS)
    pcols = [c for c in pol.columns if c != "site"]
    df = df.merge(pol, on="site", how="left")
    feats = IMPROVED_V2 + pcols
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
    t = Timer()
    log_progress(f"v3-3 POLARIS trees started ({len(pcols)} POLARIS columns)")
    for proto in PROTOS:
        ck = os.path.join(TAB, "v3_polaris_ck", tag(proto))
        os.makedirs(ck, exist_ok=True)
        folds = folds_for(proto, df)
        for f, (tr, te) in enumerate(folds):
            fn = os.path.join(ck, f"f{f}.npy")
            if not os.path.exists(fn):
                np.save(fn, np.c_[te, fit_predict_trees(X[tr], y[tr], X[te], paper=False)["Ensemble"]])
            print(f"{proto} fold {f} ({t()})", flush=True)
        P = np.full(len(df), np.nan)
        for f in range(len(folds)):
            a = np.load(os.path.join(ck, f"f{f}.npy"))
            P[a[:, 0].astype(int)] = a[:, 1]
        out = df[["site", "date"]].copy()
        out["Trees+POLARIS"] = P
        out.to_parquet(os.path.join(TAB, f"v3_polaris_trees_{tag(proto)}.parquet"))
        log_progress(f"v3-3 {proto} done ({t()})")
    log_progress("v3-3 finished")


if __name__ == "__main__":
    main()
