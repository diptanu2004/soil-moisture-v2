"""
Experiment 1 - reproduce the paper's protocol on the multi-region dataset and
fix the evaluation (persistence / no-lag baselines, leave-stations-out,
leave-one-region-out, temporal hold-out).

Writes results/tables/exp1_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (IMPROVED_LAG, IMPROVED_SAT, PAPER_FEATS, PAPER_FEATS_NOLAG, TAB, TARGET,
                    fit_predict_trees, group_folds, load_anchor, metrics, random_split, summarize)


def run_random(df, feats, paper=True):
    tr, te = random_split(len(df))
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
    preds = fit_predict_trees(X[tr], y[tr], X[te], paper=paper)
    test = df.iloc[te].copy()
    for k, v in preds.items():
        test[k] = v
    return test


def run_groupcv(df, feats, paper=True, n_splits=10, by="site"):
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
    out = df.copy()
    for k in ("RF", "XGB", "LGBM", "Ensemble"):
        out[k] = np.nan
    if by == "site":
        folds = group_folds(df["site"].to_numpy(), n_splits)
    else:  # leave-one-region-out
        folds = [(np.where(df[by] != r)[0], np.where(df[by] == r)[0]) for r in sorted(df[by].unique())]
    for tr, te in folds:
        preds = fit_predict_trees(X[tr], y[tr], X[te], paper=paper)
        for k, v in preds.items():
            out.iloc[te, out.columns.get_loc(k)] = v
    return out


def run_temporal(df, feats, paper=True, split_date="2018-01-01"):
    tr = np.where(df["date"] < split_date)[0]
    te = np.where(df["date"] >= split_date)[0]
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
    preds = fit_predict_trees(X[tr], y[tr], X[te], paper=paper)
    test = df.iloc[te].copy()
    for k, v in preds.items():
        test[k] = v
    return test


def collect(rows, res, protocol, featset, extra_baselines=()):
    for k in ("RF", "XGB", "LGBM", "Ensemble") + tuple(extra_baselines):
        m = summarize(res, k)
        rows.append(dict(protocol=protocol, features=featset, model=k, **m))


def main():
    df_all = load_anchor()
    rows = []

    # ---------------- A. paper replica (paper features, rows with all features) ----------------
    paper_df = df_all.dropna(subset=PAPER_FEATS).reset_index(drop=True)
    paper_df["Persistence(prevS1)"] = paper_df["SM_lag_prevS1"]
    print("paper-analogue rows:", len(paper_df), "sites:", paper_df.site.nunique())

    for proto, fn in [("random80/20", run_random),
                      ("leave-stations-out", run_groupcv),
                      ("temporal(2018)", run_temporal)]:
        res = fn(paper_df, PAPER_FEATS)
        collect(rows, res, proto, "paper16+lagPrevS1", ["Persistence(prevS1)"])
        res = fn(paper_df, PAPER_FEATS_NOLAG)
        collect(rows, res, proto, "paper16-noLag")
        if proto == "leave-stations-out":
            res[["site", "date", "region", TARGET, "Ensemble"]].to_parquet(f"{TAB}/exp1_loso_paper_nolag_preds.parquet")
        print(proto, "done")
    res = run_groupcv(paper_df, PAPER_FEATS, by="region")
    collect(rows, res, "leave-one-region-out", "paper16+lagPrevS1", ["Persistence(prevS1)"])
    res = run_groupcv(paper_df, PAPER_FEATS_NOLAG, by="region")
    collect(rows, res, "leave-one-region-out", "paper16-noLag")

    # ---------------- B. improved features (all 84 sites; trees handle missing VH) ----------------
    df = df_all.copy()
    df["Persistence(1d)"] = df["SM_lag1d"]
    df = df.dropna(subset=["SM_lag1d"]).reset_index(drop=True)
    for proto, fn in [("random80/20", run_random),
                      ("leave-stations-out", run_groupcv),
                      ("temporal(2018)", run_temporal)]:
        for name, feats in [("improved-satOnly", IMPROVED_SAT), ("improved+lag1d", IMPROVED_LAG)]:
            res = fn(df, feats, paper=False)
            collect(rows, res, proto, name, ["Persistence(1d)"] if "lag" in name else ())
        print(proto, "improved done")
    for name, feats in [("improved-satOnly", IMPROVED_SAT), ("improved+lag1d", IMPROVED_LAG)]:
        res = run_groupcv(df, feats, paper=False, by="region")
        collect(rows, res, "leave-one-region-out", name, ["Persistence(1d)"] if "lag" in name else ())
        if name == "improved-satOnly":
            res[["site", "region", "date", TARGET, "Ensemble", "XGB"]].to_parquet(
                f"{TAB}/exp1_loro_satonly_preds.parquet")

    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/exp1_baselines.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out[["protocol", "features", "model", "n", "R2", "RMSE", "ubRMSE", "r",
                   "site_median_r", "site_median_ubRMSE"]].round(3))


if __name__ == "__main__":
    main()
