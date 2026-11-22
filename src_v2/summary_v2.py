"""Before/after summary table, built only from results_v2/tables/*.csv.

Writes results_v2/tables/v2_summary_before_after.csv and results_v2/tables/v2_summary_before_after.md
"""
from __future__ import annotations

import os

import pandas as pd

from common_v2 import TAB

METRICS = ["n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]
PROPOSED = "Hybrid(TabPFN+TempoSAR+Trees)"


def _t(name):
    p = os.path.join(TAB, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def pick(df, label, block, **cond):
    if df is None:
        return []
    m = pd.Series(True, index=df.index)
    for k, v in cond.items():
        m &= df[k] == v
    x = df[m]
    if x.empty:
        return []
    r = x.iloc[0]
    return [dict(block=block, setting=label, **{k: r[k] for k in METRICS})]


def build():
    a, lad, loop = _t("v2_1a_before_paper_protocol.csv"), _t("v2_1b_fix_ladder.csv"), _t("v2_1c_daily_loop.csv")
    p2, p3, fs, dl = _t("v2_2_protocols.csv"), _t("v2_3_models.csv"), _t("v2_4_fewshot.csv"), _t("v2_5_daily.csv")
    R = []
    B = "1 BEFORE: paper model as published, paper protocol (16 stations, paper coordinates)"
    R += pick(a, "random 80/20 test rows, 15 training stations (paper's headline protocol)", B, coords="paper",
              features="paper16 (+lag prevS1)", evaluation="random 20% test rows (15 stations)", model="Ensemble")
    R += pick(a, "Shagbark Hills, independent validation station", B, coords="paper",
              features="paper16 (+lag prevS1)", evaluation="Shagbark Hills (unseen station)", model="Ensemble")
    R += pick(a, "Shagbark Hills, persistence of the lag (same rows with a lag)", B, coords="paper",
              features="paper16 (+lag prevS1)", evaluation="Shagbark Hills (unseen station) [rows with lag]",
              model="Persistence(prevS1)")
    R += pick(loop, "daily loop at Shagbark Hills (recursive, as in the paper)", B, coords="paper",
              method="paper loop (recursive, no drift control)")
    B = "2 BEFORE, honest evaluation: same model, all CONUS SCAN stations"
    for proto in ("random80/20", "leave-stations-out", "leave-one-region-out", "temporal(2022-23)"):
        R += pick(p2, f"{proto}: paper model with lag (prevS1)", B, protocol=proto,
                  features="paper16 as published (+lag prevS1)", model="Ensemble")
        R += pick(p2, f"{proto}: persistence (lag at previous S1 date)", B, protocol=proto, features="baseline",
                  model="Persistence(prevS1)")
        R += pick(p2, f"{proto}: paper model without lag (satellite-only)", B, protocol=proto,
                  features="paper16 as published, no lag", model="Ensemble")
    B = "3 FIXES on the paper's 16 stations (leave-one-station-out, satellite-only)"
    if lad is not None:
        for s in dict.fromkeys(lad.step):
            R += pick(lad, s, B, step=s, protocol="leave-one-station-out", variant="no lag")
    B = "4 AFTER: satellite-only models, all stations (no in-situ input)"
    for proto in ("leave-stations-out", "leave-one-region-out", "temporal(2022-23)"):
        R += pick(p2, f"{proto}: paper16 fixed, trees", B, protocol=proto, features="paper16 fixed, no lag",
                  model="Ensemble")
        for m in ("Trees", "TempoSAR", "TabPFN", PROPOSED):
            R += pick(p3, f"{proto}: {m}", B, protocol=proto, model=m)
    B = "5 AFTER: with the consistent 1-day in-situ lag (nowcasting), vs persistence"
    for proto in ("leave-stations-out", "leave-one-region-out", "temporal(2022-23)"):
        for m in ("Trees(+lag1d)", "TabPFN(+lag1d)", "Persistence(1d)"):
            R += pick(p3, f"{proto}: {m}", B, protocol=proto, model=m)
    B = "6 AFTER: few-shot adaptation to a new station (k = 10 first samples)"
    for m in (f"{PROPOSED.replace('+TempoSAR', '')}|none", "Trees|bias", "FT-TempoSAR", "TabPFN in-context",
              "Hybrid(TabPFN in-context+Trees|bias)", "Hybrid(FT-TempoSAR+Trees|bias)"):
        R += pick(fs, m, B, k=10, method=m)
    B = "7 AFTER: daily reconstruction at unseen stations"
    if dl is not None:
        for m in dl[dl.eval_set.str.startswith("common")].model:
            R += pick(dl, m, B, eval_set=dl[dl.eval_set.str.startswith("common")].eval_set.iloc[0], model=m)
    if loop is not None:
        for m in dict.fromkeys(loop[loop.coords == "nrcs"].method):
            R += pick(loop, f"paper's 16 stations, 4-fold LOSO: {m}", B, coords="nrcs", method=m)
    ab = _t("v2_6_ablation.csv")
    B = "8 WHERE THE SKILL COMES FROM: raw products and feature-group ablation (trees, leave-stations-out)"
    if ab is not None:
        for m in dict.fromkeys(ab.model):
            R += pick(ab, m, B, protocol="leave-stations-out", model=m)
    return pd.DataFrame(R)


def to_md(df):
    out = []
    for blk, g in df.groupby("block", sort=False):
        out.append(f"\n### {blk}\n")
        out.append("| setting | n | R² | RMSE | ubRMSE | bias | median station r |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in g.itertuples():
            out.append(f"| {r.setting} | {int(r.n)} | {r.R2:.3f} | {r.RMSE:.2f} | {r.ubRMSE:.2f} | {r.bias:+.2f} | "
                       f"{r.site_median_r:.3f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    S = build()
    S.to_csv(os.path.join(TAB, "v2_summary_before_after.csv"), index=False)
    with open(os.path.join(TAB, "v2_summary_before_after.md"), "w", encoding="utf-8") as f:
        f.write("# Before / after summary (all values from results_v2/tables)\n"
                "RMSE, ubRMSE, bias in % volumetric soil moisture (5 cm).\n")
        f.write(to_md(S))
    print(to_md(S))
