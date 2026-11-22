"""
Experiment v2-2: all CONUS SCAN stations (2019-2023). Paper features as published vs fixed vs
improved feature sets, under four protocols (random 80/20 = the paper's, leave-stations-out,
leave-one-region-out, temporal 2019-21 -> 2022-23), each with and without the in-situ lag,
always next to the persistence baseline. All feature sets are scored on the same rows.

Writes results_v2/tables/v2_2_protocols.csv and v2_2_oof_<protocol>.parquet
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_v2 import (IMPROVED_SAT, IMPROVED_V2, PAPER16, PAPER16_NOLAG, PROC, PROTOCOLS, TAB, TARGET, Timer,
                       collect, fit_predict_trees, folds_for, load_anchor, log_progress, with_lag)

FIXED16 = ["B_VV", "B_VH", "B_CR_dB", "B_RVI", "B_SPAN_dB", "M_S2_NDVI", "M_S2_NDMI", "M_S2_EVI",
           "P_MODIS_NDVI", "P_MODIS_EVI", "P_elev", "P_slope", "P_DOY", "P_Month", "P_precip"]

FEATSETS = [
    # name, features, paper hyper-parameters, paper row-mean fill, uses in-situ lag
    ("paper16 as published (+lag prevS1)", PAPER16, True, True, True),
    ("paper16 as published, no lag", PAPER16_NOLAG, True, True, False),
    ("paper16 fixed (+lag 1d)", with_lag(FIXED16), True, False, True),
    ("paper16 fixed, no lag", FIXED16, True, False, False),
    ("improved v1, satellite-only", IMPROVED_SAT, False, False, False),
    ("improved v1 (+lag 1d)", with_lag(IMPROVED_SAT), False, False, True),
    ("improved v2, satellite-only", IMPROVED_V2, False, False, False),
    ("improved v2 (+lag 1d)", with_lag(IMPROVED_V2), False, False, True),
]


def load_eval_rows():
    a = load_anchor()
    pa = pd.read_parquet(os.path.join(PROC, "paper_anchor.parquet"))
    pa["site"] = pa["site"].astype(str)
    pcols = sorted(set(PAPER16 + FIXED16) - {"SM_lag_prevS1"})
    a = a.merge(pa[["site", "date"] + pcols], on=["site", "date"], how="left")
    a = a.dropna(subset=["SM_lag1d", "SM_lag_prevS1"]).reset_index(drop=True)
    a["Persistence(prevS1)"] = a["SM_lag_prevS1"]
    a["Persistence(1d)"] = a["SM_lag1d"]
    return a


def main():
    df = load_eval_rows()
    log_progress(f"v2-2 started: {len(df)} rows, {df.site.nunique()} stations, {df.region.nunique()} regions")
    t = Timer()
    rows = []
    for proto in PROTOCOLS:
        folds = folds_for(proto, df)
        te_all = np.concatenate([te for _, te in folds])
        oof = df.loc[te_all, ["site", "date", "region", TARGET, "Persistence(prevS1)", "Persistence(1d)"]].copy()
        for name, feats, paper, fill, _ in FEATSETS:
            P = {k: np.full(len(df), np.nan) for k in ("RF", "XGB", "LGBM", "Ensemble")}
            X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
            for tr, te in folds:
                p = fit_predict_trees(X[tr], y[tr], X[te], paper=paper, paper_fill=fill)
                for k in P:
                    P[k][te] = p[k]
            ev = df.loc[te_all, ["site", "date", "region", TARGET]].copy()
            for k in P:
                ev[k] = P[k][te_all]
            collect(rows, ev, list(P), protocol=proto, features=name)
            oof[name] = P["Ensemble"][te_all]
            print(proto, name, t(), flush=True)
        ev = df.loc[te_all]
        collect(rows, ev, ["Persistence(prevS1)", "Persistence(1d)"], protocol=proto, features="baseline")
        oof.to_parquet(f"{TAB}/v2_2_oof_{proto.replace('/', '_').replace('(', '_').replace(')', '')}.parquet")
        pd.DataFrame(rows).to_csv(f"{TAB}/v2_2_protocols.csv", index=False)
        log_progress(f"v2-2 protocol {proto} done ({t()})")
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(out[out.model.isin(["Ensemble", "Persistence(prevS1)", "Persistence(1d)"])][
            ["protocol", "features", "model", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]].round(3).to_string())
    log_progress("v2-2 finished")


if __name__ == "__main__":
    main()
