"""
v4 daily: the final (declared v4 candidate) hybrid and the v4 reference on EVERY labelled day.

The anchor models only predict on Sentinel-1 dates (they need that day's S1 features), so as in v2-5
their out-of-fold anchor estimates (v3_oof_<protocol>.parquet) are held / linearly interpolated onto
every day at each station. Nothing is fitted: interpolation uses only the model's own satellite-only
predictions at the same (held-out) station.

Rows are scored on all days, Sentinel-1 days and non-Sentinel-1 days (has_s1 from v2_5_daily_preds),
with a station-cluster bootstrap and paired dR2 vs the v2 daily anchor interpolation.
Extra exploratory row (LOSO only, no fitting): equal-weight mean of the v4 linear interpolation and
the v2-5 LGBM-daily satOnly predictions (both out-of-fold for the station).

Writes results_v2/tables/v4_daily.csv and v4_daily_preds.parquet
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_v2 import TAB, TARGET, summarize
from exp_v3_combine import REF4, V4_CAND, bootstrap, tag

V2_HEAD = "Hybrid(TabPFN+TempoSAR+Trees)"
NAMES = {V2_HEAD: "v2 hybrid", REF4: "v4 reference", V4_CAND: "v4 final"}


def anchor_interp(d, ap):
    """hold-last / linear interpolation of anchor estimates onto every day (same as v2-5)."""
    hold, lin = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for s, g in d.groupby("site"):
        a = ap[ap.site == s].set_index("date")["pred"].sort_index()
        a = a[~a.index.duplicated()]
        if len(a) < 5:
            continue
        idx = pd.DatetimeIndex(g["date"])
        hold[g.index] = a.reindex(idx, method="ffill").to_numpy()
        lin[g.index] = a.reindex(idx.union(a.index)).interpolate("time", limit_area="inside").reindex(idx).to_numpy()
    return hold, lin


def main():
    D = pd.read_parquet(os.path.join(TAB, "v2_5_daily_preds.parquet"))
    D = D[["site", "date", "region", TARGET, "has_s1", "LGBM-daily satOnly"]].reset_index(drop=True)
    D["date"] = pd.to_datetime(D["date"])
    rows, out = [], D.copy()
    for proto in ["leave-stations-out", "leave-one-region-out"]:
        oof = pd.read_parquet(os.path.join(TAB, f"v3_oof_{tag(proto)}.parquet"))
        oof["date"] = pd.to_datetime(oof["date"])
        R = D.copy()
        cols = []
        for m, short in NAMES.items():
            hold, lin = anchor_interp(R, oof[["site", "date"]].assign(pred=oof[m]))
            for kind, v in [("hold-last", hold), ("linear interp", lin)]:
                c = f"{short}, {kind}"
                R[c] = v
                cols.append(c)
        if proto == "leave-stations-out":
            c = "v4 final interp + LGBM-daily (mean)"
            R[c] = (R["v4 final, linear interp"] + R["LGBM-daily satOnly"]) / 2
            cols += ["LGBM-daily satOnly", c]
        R = R.dropna(subset=cols)                      # common days: every row scored on the same days
        ref = "v2 hybrid, linear interp"
        for split, g in [("all days", R), ("Sentinel-1 days", R[R.has_s1 == 1]),
                         ("non-Sentinel-1 days", R[R.has_s1 == 0])]:
            g = g.reset_index(drop=True)
            bs = bootstrap(g, [c for c in cols if c != ref], ref=ref)
            for c in cols:
                b = bs.get(c, {})
                rows.append(dict(protocol=proto, days=split, model=c, **summarize(g, c),
                                 R2_lo=b.get("R2_lo"), R2_hi=b.get("R2_hi"),
                                 dR2_vs_v2interp=b.get("dR2_vs_head", 0.0), dR2_lo=b.get("dR2_lo"),
                                 dR2_hi=b.get("dR2_hi")))
        out = out.merge(R[["site", "date"] + [c for c in cols if c != "LGBM-daily satOnly"]]
                        .rename(columns={c: f"{c} [{proto}]" for c in cols}), on=["site", "date"], how="left")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TAB, "v4_daily.csv"), index=False)
    out.to_parquet(os.path.join(TAB, "v4_daily_preds.parquet"))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(res[["protocol", "days", "model", "n", "R2", "R2_lo", "R2_hi", "RMSE", "bias", "site_median_r",
                   "dR2_vs_v2interp", "dR2_lo", "dR2_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
