"""
Experiment 3 - does adding more stations / more regions help?

3A  learning curve: skill at held-out stations vs number of training stations
    (random station subsets, 3 repeats, 10 station folds)
3B  Great-Plains/South test stations (the paper's study area): train on
    same-region stations only vs. same-region + all other regions
3C  cross-region transfer matrix (train on region A only -> test on region B)

LightGBM (regularised) on the improved satellite-only feature set; the
in-situ-assisted variant (with a consistent 1-day lag) is reported for 3A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import IMPROVED_LAG, IMPROVED_SAT, TAB, TARGET, group_folds, load_anchor, make_tree_models, summarize


def lgbm():
    return make_tree_models(paper=False)["LGBM"]


def fit_pred(df, feats, tr, te):
    m = lgbm()
    m.fit(df.iloc[tr][feats].to_numpy(float), df.iloc[tr][TARGET].to_numpy(float))
    return m.predict(df.iloc[te][feats].to_numpy(float))


def part_a(df):
    sites = df["site"].to_numpy()
    rows = []
    for feats_name, feats in [("satOnly", IMPROVED_SAT), ("+lag1d", IMPROVED_LAG)]:
        for N in (5, 10, 15, 25, 40, 60, "all"):
            for rep in range(3 if N != "all" else 1):
                rng = np.random.RandomState(1000 * rep + (N if N != "all" else 999))
                pred = np.full(len(df), np.nan)
                for tr, te in group_folds(sites, 10):
                    pool = np.unique(sites[tr])
                    n = len(pool) if N == "all" else min(N, len(pool))
                    chosen = rng.choice(pool, n, replace=False)
                    tr2 = tr[np.isin(sites[tr], chosen)]
                    pred[te] = fit_pred(df, feats, tr2, te)
                d = df.assign(p=pred)
                rows.append(dict(features=feats_name, n_train_sites=str(N), rep=rep, **summarize(d, "p")))
            print(feats_name, N, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/exp3a_learning_curve_raw.csv", index=False)
    agg = res.groupby(["features", "n_train_sites"], sort=False)[["R2", "RMSE", "site_median_r", "site_median_ubRMSE"]].agg(["mean", "std"])
    agg.to_csv(f"{TAB}/exp3a_learning_curve.csv")
    print(agg.round(3))


def part_b(df, target_regions=("South", "N.Rockies&Plains")):
    """Paper's study area = Great Plains / South. Leave-one-station-out within it."""
    rows = []
    tgt_sites = df.loc[df.region.isin(target_regions), "site"].unique()
    preds = {k: [] for k in ("same-region only", "same-region + other regions", "other regions only")}
    for s in tgt_sites:
        te = np.where(df.site == s)[0]
        same = np.where(df.region.isin(target_regions) & (df.site != s))[0]
        other = np.where(~df.region.isin(target_regions))[0]
        for k, tr in [("same-region only", same), ("same-region + other regions", np.concatenate([same, other])),
                      ("other regions only", other)]:
            preds[k].append(df.iloc[te][["site", "date", "region", TARGET]].assign(p=fit_pred(df, IMPROVED_SAT, tr, te)))
    for k, v in preds.items():
        d = pd.concat(v)
        rows.append(dict(training=k, test_sites=len(tgt_sites), **summarize(d, "p")))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/exp3b_more_regions.csv", index=False)
    print(res.round(3).to_string())


def part_c(df, min_sites=5):
    regs = [r for r, n in df.groupby("region")["site"].nunique().items() if n >= min_sites]
    rows = []
    for a in regs:
        for b in regs:
            te_idx = np.where(df.region == b)[0]
            if a == b:   # within-region leave-stations-out
                sub = df.iloc[te_idx].reset_index(drop=True)
                pred = np.full(len(sub), np.nan)
                for tr, te in group_folds(sub["site"].to_numpy(), min(5, sub.site.nunique())):
                    pred[te] = fit_pred(sub, IMPROVED_SAT, tr, te)
                d = sub.assign(p=pred)
            else:
                tr_idx = np.where(df.region == a)[0]
                d = df.iloc[te_idx].assign(p=fit_pred(df, IMPROVED_SAT, tr_idx, te_idx))
            rows.append(dict(train_region=a, test_region=b, **summarize(d, "p")))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/exp3c_cross_region.csv", index=False)
    print(res.pivot(index="train_region", columns="test_region", values="site_median_r").round(2))
    print(res.pivot(index="train_region", columns="test_region", values="RMSE").round(2))


if __name__ == "__main__":
    df = load_anchor().dropna(subset=["SM_lag1d"]).reset_index(drop=True)
    part_b(df)
    part_c(df)
    part_a(df)
