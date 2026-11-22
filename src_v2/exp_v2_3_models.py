"""
Experiment v2-3: satellite-only retrieval models on all stations (same rows as v2-2).

  Trees      regularised RF/XGB/LGBM ensemble on the improved v2 features (from v2-2)
  TempoSAR   Transformer over the last 10 Sentinel-1 acquisitions of the station (src/deep.py, GPU)
  TabPFN-v2  pretrained tabular foundation model (Hollmann et al., Nature 2025), in-context
             learning with a context set of <= 10,000 training rows (its pretraining limit):
             stratified by station (leave-stations-out, temporal) or balanced by region
             (leave-one-region-out); no gradient training, no hyper-parameter tuning
  Hybrids    equal-weight means (no weights fitted on test data)

Protocols: leave-stations-out (10 folds), leave-one-region-out, temporal (2022-23).
TabPFN is also run with the 1-day in-situ lag for comparison with persistence.

Writes results_v2/tables/v2_3_models.csv and v2_3_oof_<protocol>.parquet
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common_v2 import (IMPROVED_V2, TAB, TARGET, TABPFN_MAX_CONTEXT, Timer, collect, context_subsample, folds_for,
                       load_full_anchor, log_progress, tabpfn_fit_predict, with_lag)
from deep_v2 import Prepared, deep_fit_predict
from exp_v2_2_protocols import load_eval_rows

PROTOS = ["leave-stations-out", "leave-one-region-out", "temporal(2022-23)"]


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


def main(protos=PROTOS, skip_deep=False):
    df = load_eval_rows()
    full = load_full_anchor()
    full = full[full["site"].isin(df["site"].unique())].reset_index(drop=True)
    P = Prepared(df, full, L=10)
    sites = df["site"].to_numpy()
    log_progress(f"v2-3 started ({len(df)} rows)")
    t = Timer()
    rows = []
    for proto in protos:
        folds = folds_for(proto, df)
        te_all = np.concatenate([te for _, te in folds])
        oof22 = pd.read_parquet(f"{TAB}/v2_2_oof_{tag(proto)}.parquet")
        res = df.loc[te_all, ["site", "date", "region", TARGET, "SM_lag1d"]].copy()
        res = res.merge(oof22[["site", "date", "improved v2, satellite-only", "improved v2 (+lag 1d)"]],
                        on=["site", "date"], how="left").rename(
            columns={"improved v2, satellite-only": "Trees", "improved v2 (+lag 1d)": "Trees(+lag1d)"})
        pred = {k: np.full(len(df), np.nan) for k in ("TempoSAR", "TabPFN", "TabPFN(+lag1d)")}
        X = df[IMPROVED_V2].to_numpy(np.float32)
        XL = df[with_lag(IMPROVED_V2)].to_numpy(np.float32)
        y = df[TARGET].to_numpy(np.float32)
        for f, (tr, te) in enumerate(folds):
            bal = df.loc[tr, "region"].to_numpy() if proto == "leave-one-region-out" else None
            ctx = tr[context_subsample(sites[tr], TABPFN_MAX_CONTEXT, seed=f, balance=bal)]
            pred["TabPFN"][te] = tabpfn_fit_predict(X[ctx], y[ctx], X[te])
            pred["TabPFN(+lag1d)"][te] = tabpfn_fit_predict(XL[ctx], y[ctx], XL[te])
            if not skip_deep:
                pred["TempoSAR"][te] = deep_fit_predict(P, tr, te, sites)[0]
            print(proto, "fold", f, t(), flush=True)
        for k, v in pred.items():
            res[k] = v[te_all]
        res["Hybrid(TempoSAR+Trees)"] = 0.5 * (res["TempoSAR"] + res["Trees"])
        res["Hybrid(TabPFN+Trees)"] = 0.5 * (res["TabPFN"] + res["Trees"])
        res["Hybrid(TabPFN+TempoSAR+Trees)"] = (res["TabPFN"] + res["TempoSAR"] + res["Trees"]) / 3
        res["Persistence(1d)"] = res["SM_lag1d"]
        cols = ["Trees", "TempoSAR", "TabPFN", "Hybrid(TempoSAR+Trees)", "Hybrid(TabPFN+Trees)",
                "Hybrid(TabPFN+TempoSAR+Trees)", "Trees(+lag1d)", "TabPFN(+lag1d)", "Persistence(1d)"]
        if skip_deep:
            cols = [c for c in cols if "TempoSAR" not in c]
        collect(rows, res, cols, protocol=proto)
        res.to_parquet(f"{TAB}/v2_3_oof_{tag(proto)}.parquet")
        pd.DataFrame(rows).to_csv(f"{TAB}/v2_3_models.csv", index=False)
        log_progress(f"v2-3 {proto} done ({t()})")
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220):
        print(out[["protocol", "model", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]].round(3).to_string())
    log_progress("v2-3 finished")


if __name__ == "__main__":
    main(skip_deep="--no-deep" in sys.argv)
