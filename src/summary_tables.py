"""Headline comparison tables for the report (computed from saved predictions)."""
from __future__ import annotations

import pandas as pd

from common import TAB, TARGET, summarize


def main():
    paper = pd.read_parquet(f"{TAB}/exp1_loso_paper_nolag_preds.parquet")
    a = pd.read_parquet(f"{TAB}/exp2a_loso_preds.parquet")
    common = a.merge(paper[["site", "date", "Ensemble"]].rename(columns={"Ensemble": "Paper16(noLag)"}),
                     on=["site", "date"], how="inner")
    rows = []
    for name, col in [("Paper features + paper RF/XGB/LGBM ensemble", "Paper16(noLag)"),
                      ("Improved features + regularised tree ensemble", "TreeEns"),
                      ("TempoSAR (Transformer over S1 history)", "TempoSAR"),
                      ("Hybrid (TempoSAR + tree ensemble)", "Hybrid")]:
        rows.append(dict(eval_set=f"common rows (n={len(common)}, {common.site.nunique()} sites)", model=name,
                         **summarize(common, col)))
    for name, col in [("Improved features + regularised tree ensemble", "TreeEns"),
                      ("TempoSAR (Transformer over S1 history)", "TempoSAR"),
                      ("Hybrid (TempoSAR + tree ensemble)", "Hybrid")]:
        rows.append(dict(eval_set=f"all rows (n={len(a)}, {a.site.nunique()} sites)", model=name, **summarize(a, col)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/summary_satellite_only_loso.csv", index=False)
    with pd.option_context("display.width", 220):
        print(res[["eval_set", "model", "n", "R2", "RMSE", "ubRMSE", "bias", "r", "site_median_r",
                   "site_median_ubRMSE"]].round(3).to_string())


if __name__ == "__main__":
    main()
