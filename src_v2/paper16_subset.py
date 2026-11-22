"""Before/after on the paper's 16 stations only, taken from the all-station leave-stations-out
predictions (each station predicted by models that never saw it). Writes
results_v2/tables/v2_7_paper16_subset.csv"""
from __future__ import annotations

import os

import pandas as pd

from common_v2 import ROOT, TAB, summarize

P16 = [str(s) for s in pd.read_csv(os.path.join(ROOT, "gee", "stations_paper16_corrected.csv")).scan_id]


def load(name):
    p = os.path.join(TAB, name)
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d["site"] = d["site"].astype(str)
    return d[d.site.isin(P16)]


rows = []
for proto, tag in (("leave-stations-out", "leave-stations-out"), ("leave-one-region-out", "leave-one-region-out"),
                   ("temporal(2022-23)", "temporal_2022-23")):
    a, b = load(f"v2_2_oof_{tag}.parquet"), load(f"v2_3_oof_{tag}.parquet")
    for df, col, label in (
            (a, "paper16 as published, no lag", "BEFORE paper model, satellite-only"),
            (a, "paper16 as published (+lag prevS1)", "BEFORE paper model + lag (previous S1 date)"),
            (a, "Persistence(prevS1)", "persistence (lag at previous S1 date)"),
            (b, "Trees", "AFTER improved trees, satellite-only"),
            (b, "TempoSAR", "AFTER TempoSAR, satellite-only"),
            (b, "TabPFN", "AFTER TabPFN v2, satellite-only"),
            (b, "Hybrid(TabPFN+TempoSAR+Trees)", "AFTER proposed hybrid, satellite-only"),
            (b, "TabPFN(+lag1d)", "AFTER TabPFN v2 + consistent 1-day lag"),
            (b, "Persistence(1d)", "persistence (in-situ yesterday)")):
        if df is not None and col in df:
            rows.append(dict(protocol=proto, setting=label, stations=df.site.nunique(),
                             **summarize(df.dropna(subset=[col]), col)))
d = load("v2_5_daily_preds.parquet")
if d is not None:
    for col in [c for c in d.columns if c.startswith(("LGBM", "DailyLSTM", "Anchor", "Persistence"))]:
        rows.append(dict(protocol="daily, 5-fold leave-stations-out", setting=col, stations=d.site.nunique(),
                         **summarize(d.dropna(subset=[col]), col)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB, "v2_7_paper16_subset.csv"), index=False)
print(out[["protocol", "setting", "n", "R2", "RMSE", "ubRMSE", "site_median_r"]].round(3).to_string())
