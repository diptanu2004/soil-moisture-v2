"""
Experiment 4 - daily reconstruction (the paper's "gap-fill" stage).

G1  persistence skill (y_t = y_{t-1}) at all CONUS SCAN sites and at the
    paper's 16 stations (2015-2018 record)
G2  persistence skill vs. age of the last in-situ value (1..30 days)
G3  lag-definition mismatch: a gap-fill model trained with the lag at the
    previous Sentinel-1 date (paper) but applied with a 1-day lag, versus
    the same model trained and applied with a 1-day lag (consistent)
G4  satellite-only daily reconstruction at held-out stations from the
    Sentinel-1 anchor estimates (no in-situ input)

Writes results/tables/exp4_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import TAB, group_folds, load_daily, metrics, summarize

PAPER_STATIONS = {  # SCAN ids of the paper's 15 training stations + Shagbark Hills
    2092: "Abrams", 2031: "Ames", 2006: "Bushland #1", 2048: "Dexter", 2022: "Fort Reno #1",
    2147: "Ku-Nesa", 2105: "Levelland", 2004: "Mason #1", 2060: "Mt Vernon", 2093: "Phillipsburg",
    2104: "Reese Center", 2199: "Riesel", 2001: "Rogers Farm #1", 2176: "Selma", 2047: "Spickard",
    2068: "Shagbark Hills",
}


def daily_frame():
    d = load_daily()[["site", "date", "sm_5cm"]].sort_values(["site", "date"])
    # reindex to a full calendar per site so that lags are true calendar lags
    out = []
    for s, g in d.groupby("site"):
        g = g.set_index("date").asfreq("D")
        g["site"] = s
        out.append(g.reset_index())
    d = pd.concat(out, ignore_index=True)
    for h in (1, 2, 4, 8, 12, 30):
        d[f"lag{h}"] = d.groupby("site")["sm_5cm"].shift(h)
    d["DOY"] = d["date"].dt.dayofyear
    d["Month"] = d["date"].dt.month
    return d


def g1_g2(d):
    rows = []
    v = d.dropna(subset=["sm_5cm", "lag1"])
    m = summarize(v.assign(p=v["lag1"]), "p")
    rows.append(dict(set="all CONUS SCAN sites", sites=v.site.nunique(), **m))
    ps = v[v.site.isin(PAPER_STATIONS)]
    m = summarize(ps.assign(p=ps["lag1"]), "p")
    rows.append(dict(set="paper's 16 stations", sites=ps.site.nunique(), **m))
    sh = v[v.site == 2068]
    m = summarize(sh.assign(p=sh["lag1"]), "p")
    rows.append(dict(set="Shagbark Hills (2068)", sites=1, **m))
    g1 = pd.DataFrame(rows)
    per = []
    for sid, name in PAPER_STATIONS.items():
        g = v[v.site == sid]
        if len(g) > 30:
            per.append(dict(site=sid, name=name, **metrics(g["sm_5cm"], g["lag1"])))
    per = pd.DataFrame(per)

    rows = []
    for h in (1, 2, 4, 8, 12, 30):
        vv = d.dropna(subset=["sm_5cm", f"lag{h}"])
        rows.append(dict(horizon_days=h, **summarize(vv.assign(p=vv[f"lag{h}"]), "p")))
    g2 = pd.DataFrame(rows)
    return g1, per, g2


def g3_lag_mismatch(d, anchor):
    """Gap-fill model with the paper's features that are available here
    (lag, DOY, Month, elevation, site rain climatology) trained on S1-date rows."""
    from xgboost import XGBRegressor

    a = anchor.dropna(subset=["SM_lag_prevS1", "SM_lag1d"]).copy()
    feats_static = ["elev", "DOY", "Month"]
    d = d.merge(a.groupby("site")[["elev"]].first(), left_on="site", right_index=True, how="inner")
    d = d.dropna(subset=["sm_5cm", "lag1"])
    rows = []
    preds = []
    for tr, te in group_folds(a["site"].to_numpy(), 10):
        tr_sites = a.iloc[tr]["site"].unique()
        te_sites = a.iloc[te]["site"].unique()
        A = a[a.site.isin(tr_sites)]
        D = d[d.site.isin(te_sites)].copy()
        cfg = dict(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=2)
        # paper: trained with lag at previous S1 date, applied with previous-day lag
        m1 = XGBRegressor(**cfg).fit(A[["SM_lag_prevS1"] + feats_static].to_numpy(), A["sm_5cm"])
        D["paper_mismatch"] = m1.predict(D[["lag1"] + feats_static].to_numpy())
        # consistent: trained with previous-day lag, applied with previous-day lag
        m2 = XGBRegressor(**cfg).fit(A[["SM_lag1d"] + feats_static].to_numpy(), A["sm_5cm"])
        D["consistent_lag1d"] = m2.predict(D[["lag1"] + feats_static].to_numpy())
        D["persistence"] = D["lag1"]
        preds.append(D)
    P = pd.concat(preds)
    for k in ["paper_mismatch", "consistent_lag1d", "persistence"]:
        rows.append(dict(model=k, **summarize(P, k)))
    return pd.DataFrame(rows), P


def g4_satellite_only(d, anchor_preds):
    """Daily series at held-out stations built only from satellite anchor estimates.
    anchor_preds: columns site, date, pred (out-of-station predictions on S1 dates)."""
    rows, series = [], []
    for s, g in d.groupby("site"):
        ap = anchor_preds[anchor_preds.site == s].set_index("date")["pred"].sort_index()
        if len(ap) < 5:
            continue
        g = g.set_index("date")
        idx = g.index
        # (a) hold last anchor (causal, near-real-time)
        hold = ap.reindex(idx, method="ffill")
        # (b) linear interpolation between anchors (offline reanalysis)
        lin = ap.reindex(idx.union(ap.index)).interpolate("time").reindex(idx)
        # (c) climatology of the anchor estimates by month (smooth seasonal product)
        clim = ap.groupby(ap.index.month).mean().reindex(idx.month).to_numpy()
        out = pd.DataFrame({"site": s, "sm_5cm": g["sm_5cm"], "hold": hold, "linear": lin,
                            "anchor_monthly_clim": clim}, index=idx).reset_index()
        series.append(out)
    S = pd.concat(series, ignore_index=True)
    S = S[(S.date >= anchor_preds.date.min()) & (S.date <= anchor_preds.date.max())]
    S = S.dropna(subset=["sm_5cm", "hold", "linear"])
    for k in ["hold", "linear", "anchor_monthly_clim"]:
        rows.append(dict(method=k, **summarize(S, k)))
    return pd.DataFrame(rows), S


def main():
    d = daily_frame()
    g1, per, g2 = g1_g2(d)
    g1.to_csv(f"{TAB}/exp4_g1_persistence.csv", index=False)
    per.to_csv(f"{TAB}/exp4_g1_paper_stations.csv", index=False)
    g2.to_csv(f"{TAB}/exp4_g2_persistence_horizon.csv", index=False)
    print(g1.round(3).to_string())
    print(per.round(3).to_string())
    print(g2.round(3).to_string())

    from common import load_anchor
    anchor = load_anchor()
    g3, P = g3_lag_mismatch(d, anchor)
    g3.to_csv(f"{TAB}/exp4_g3_lag_mismatch.csv", index=False)
    P[P.site == P.site.iloc[0]].to_parquet(f"{TAB}/exp4_g3_example_series.parquet")
    P[["site", "date", "sm_5cm", "paper_mismatch", "consistent_lag1d", "persistence"]].to_parquet(
        f"{TAB}/exp4_g3_preds.parquet")
    print(g3.round(3).to_string())

    import os
    p = f"{TAB}/exp2_loso_satonly_best_preds.parquet"
    if os.path.exists(p):
        ap = pd.read_parquet(p)
        g4, S = g4_satellite_only(d, ap)
        g4.to_csv(f"{TAB}/exp4_g4_satellite_only_daily.csv", index=False)
        S.to_parquet(f"{TAB}/exp4_g4_series.parquet")
        print(g4.round(3).to_string())


if __name__ == "__main__":
    main()
