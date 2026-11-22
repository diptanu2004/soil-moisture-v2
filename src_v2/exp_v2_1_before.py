"""
Experiment v2-1: the paper's model, unchanged, on the paper's 16 stations (2019-2023), then the
bug fixes one at a time (fix ladder), then the paper's daily gap-fill loop.

A  "before" (paper protocol): paper coordinates, the paper's 16 features built the paper's way,
   row.fillna(row.mean()), RF/XGB/LGBM paper ensemble, random 80/20 split over the anchor rows
   of the 15 training stations + Shagbark Hills as the independent validation station.
B  fix ladder (16 stations): each step adds one fix; evaluated with the paper's random 80/20 split
   and with leave-one-station-out, with and without the lag, against persistence.
C  daily reconstruction: the paper's loop (anchor model on S1 days, gap-fill model otherwise,
   both trained with the lag at the previous S1 date, applied recursively with the previous
   day's value) vs the same loop fed with in-situ values, vs the drift-controlled loop
   (consistent 1-day lag + reset to a satellite-only anchor estimate on every S1 date),
   vs persistence.

Writes results_v2/tables/v2_1_*.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_v2 import (PAPER16, PAPER16_NOLAG, PAPER_GAPFILL, PROC, ROOT, TAB, TARGET, Timer, collect,
                       fit_predict_trees, group_folds, log_progress, predict_trees, random_split, summarize)

ROLES = pd.read_csv(os.path.join(ROOT, "gee", "stations_paper16_corrected.csv"))
VALIDATION = os.environ.get("SM_VALIDATION", str(ROLES.loc[ROLES.role == "validation", "scan_id"].iloc[0]))  # Shagbark Hills (2068)

SAR_LIN_PT = {"P_ratio": "L_CR_dB", "P_RVI": "L_RVI", "P_SPAN": "L_SPAN_dB"}
SAR_BUF = {"P_VV": "B_VV", "P_VH": "B_VH", "P_ratio": "B_CR_dB", "P_RVI": "B_RVI", "P_SPAN": "B_SPAN_dB"}
S2_MASK = {"P_S2_NDVI": "M_S2_NDVI", "P_S2_NDMI": "M_S2_NDMI", "P_S2_EVI": "M_S2_EVI"}


def sub(feats, mapping):
    return [mapping.get(f, f) for f in feats]


# step name, coordinates, feature list (with lag), row-mean fill, lag column
def ladder():
    f = list(PAPER16)
    steps = [("S0 paper as published", "paper", f, True, "SM_lag_prevS1")]
    steps.append(("S1 + corrected coordinates", "nrcs", f, True, "SM_lag_prevS1"))
    f = sub(f, SAR_LIN_PT)
    steps.append(("S2 + SAR indices in linear units", "nrcs", f, True, "SM_lag_prevS1"))
    f = sub(f, {"P_VV": "B_VV", "P_VH": "B_VH", "L_CR_dB": "B_CR_dB", "L_RVI": "B_RVI", "L_SPAN_dB": "B_SPAN_dB"})
    steps.append(("S3 + 50 m field-scale SAR mean", "nrcs", f, True, "SM_lag_prevS1"))
    f = sub(f, S2_MASK)
    steps.append(("S4 + Sentinel-2 SCL cloud mask", "nrcs", f, True, "SM_lag_prevS1"))
    steps.append(("S5 + no row-mean fill (NaN-aware)", "nrcs", f, False, "SM_lag_prevS1"))
    f = [("SM_lag1d" if x == "SM_lag_prevS1" else x) for x in f]
    steps.append(("S6 + consistent 1-day lag", "nrcs", f, False, "SM_lag1d"))
    return steps


def stations_rows(pa, coords):
    s = pa[pa["paper16"].astype(bool)]
    s = s[s["site"].str.endswith("_pc")] if coords == "paper" else s[~s["site"].str.endswith("_pc")]
    return s.dropna(subset=[TARGET]).reset_index(drop=True)


# ----------------------------------------------------------------------------
def part_a(pa):
    rows = []
    for coords in ("paper", "nrcs"):
        d = stations_rows(pa, coords)
        trn = d[d["scan_id"] != VALIDATION].reset_index(drop=True)
        val = d[d["scan_id"] == VALIDATION].reset_index(drop=True)
        tr, te = random_split(len(trn))
        for name, feats in (("paper16 (+lag prevS1)", PAPER16), ("paper16 no lag", PAPER16_NOLAG)):
            p = fit_predict_trees(trn.loc[tr, feats], trn.loc[tr, TARGET], pd.concat([trn.loc[te, feats], val[feats]]),
                                  paper=True, paper_fill=True)
            test = trn.loc[te].copy()
            vv = val.copy()
            for k, v in p.items():
                test[k] = v[: len(te)]
                vv[k] = v[len(te):]
            test["Persistence(prevS1)"] = test["SM_lag_prevS1"]
            vv["Persistence(prevS1)"] = vv["SM_lag_prevS1"]
            for ev_name, ev in (("random 20% test rows (15 stations)", test), ("Shagbark Hills (unseen station)", vv)):
                cols = ["RF", "XGB", "LGBM", "Ensemble"]
                collect(rows, ev, cols, coords=coords, features=name, evaluation=ev_name)
                if "lag" in name and "no lag" not in name:
                    e = ev.dropna(subset=["SM_lag_prevS1"])
                    collect(rows, e, ["Persistence(prevS1)"], coords=coords, features=name,
                            evaluation=ev_name + " [rows with lag]")
                    collect(rows, e, ["Ensemble"], coords=coords, features=name,
                            evaluation=ev_name + " [rows with lag]")
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/v2_1a_before_paper_protocol.csv", index=False)
    return out


def part_b(pa):
    rows = []
    for step, coords, feats, fill, lag in ladder():
        d = stations_rows(pa, coords)
        nolag = [f for f in feats if f != lag]
        for proto in ("random80/20", "leave-one-station-out"):
            if proto == "random80/20":
                folds = [random_split(len(d))]
            else:
                folds = list(group_folds(d["scan_id"].to_numpy(), n_splits=16))
            for variant, F in (("with lag", feats), ("no lag", nolag)):
                pred = np.full(len(d), np.nan)
                for tr, te in folds:
                    pred[te] = fit_predict_trees(d.loc[tr, F], d.loc[tr, TARGET], d.loc[te, F],
                                                 paper=True, paper_fill=fill)["Ensemble"]
                test_idx = np.concatenate([te for _, te in folds])
                ev = d.loc[test_idx].assign(pred=pred[test_idx], persistence=d.loc[test_idx, lag])
                rows.append(dict(step=step, protocol=proto, variant=variant, model="Ensemble", **summarize(ev, "pred")))
                if variant == "with lag":
                    e = ev.dropna(subset=[lag])
                    rows.append(dict(step=step, protocol=proto, variant=f"persistence ({lag})", model="Persistence",
                                     **summarize(e, "persistence")))
        log_progress(f"v2-1 fix ladder: {step} done")
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/v2_1b_fix_ladder.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# daily loop
# ----------------------------------------------------------------------------
def run_loop(pdaily, panchor, sites, mode, anc_models, anc_feats, anc_fill, gap_models, gap_feats, lag_name):
    """Vectorised over stations, sequential over days. mode: 'paper' (recursive on own
    predictions), 'insitu' (yesterday's in-situ value when available), 'drift' (anchor model is
    satellite-only, gap model uses the consistent 1-day lag, recursive from the last anchor)."""
    D = pdaily[pdaily["site"].isin(sites)].sort_values(["site", "date"])
    dates = np.sort(D["date"].unique())
    S, T = len(sites), len(dates)
    y = D.pivot(index="site", columns="date", values=TARGET).reindex(index=sites, columns=dates).to_numpy()
    G = {f: D.pivot(index="site", columns="date", values=f).reindex(index=sites, columns=dates).to_numpy()
         for f in gap_feats if f != lag_name}
    A = panchor[panchor["site"].isin(sites)].set_index(["site", "date"])
    A = A[~A.index.duplicated()]
    anc_at = {}
    for (s, dt), r in A.iterrows():
        anc_at.setdefault(np.datetime64(dt, "ns"), []).append((sites.index(s), r))
    # initial state: first in-situ value of each station (the paper initialises with in-situ)
    prev = np.array([row[np.isfinite(row)][0] if np.isfinite(row).any() else np.nan for row in y])
    pred = np.full((S, T), np.nan)
    for t, dt in enumerate(dates):
        Xg = np.column_stack([prev if f == lag_name else G[f][:, t] for f in gap_feats])
        p = predict_trees(gap_models, Xg)["Ensemble"]
        if dt in anc_at:
            idx = [i for i, _ in anc_at[dt]]
            Xa = np.array([[prev[i] if f in ("SM_lag_prevS1", "SM_lag1d") else r[f] for f in anc_feats]
                           for i, r in anc_at[dt]], float)
            p[idx] = predict_trees(anc_models, Xa, paper_fill=anc_fill)["Ensemble"]
        pred[:, t] = p
        if mode == "insitu":
            prev = np.where(np.isfinite(y[:, t]), y[:, t], p)
        else:
            prev = p
    out = pd.DataFrame({"site": np.repeat(sites, T), "date": np.tile(dates, S),
                        TARGET: y.ravel(), "pred": pred.ravel()})
    return out


def part_c(pa, pdaily):
    rows, series = [], []
    t = Timer()
    for coords in ("paper", "nrcs"):
        d = stations_rows(pa, coords)
        pdd = pdaily[pdaily["paper16"].astype(bool)]
        pdd = pdd[pdd["site"].str.endswith("_pc")] if coords == "paper" else pdd[~pdd["site"].str.endswith("_pc")]
        groups = d["scan_id"].to_numpy()
        if coords == "paper":
            folds = [(np.where(groups != VALIDATION)[0], np.where(groups == VALIDATION)[0])]
            evaluation = "Shagbark Hills (paper protocol)"
        else:
            folds = list(group_folds(groups, n_splits=4))
            evaluation = "4-fold leave-stations-out (16 stations)"
        for tr, te in folds:
            trn = d.loc[tr]
            test_sites = sorted(d.loc[te, "site"].unique())
            # paper models: anchor = paper16 (+prevS1 lag, row-mean fill); gap-fill = daily subset (+prevS1 lag)
            _, anc = fit_predict_trees(trn[PAPER16], trn[TARGET], trn[PAPER16].iloc[:2], paper=True, paper_fill=True,
                                       return_models=True)
            _, gap = fit_predict_trees(trn[PAPER_GAPFILL], trn[TARGET], trn[PAPER_GAPFILL].iloc[:2], paper=True,
                                       return_models=True)
            # drift-controlled: anchor without lag (satellite-only reset), gap model with consistent 1-day lag
            _, anc0 = fit_predict_trees(trn[PAPER16_NOLAG], trn[TARGET], trn[PAPER16_NOLAG].iloc[:2], paper=True,
                                        paper_fill=False, return_models=True)
            g1 = [("SM_lag1d" if f == "SM_lag_prevS1" else f) for f in PAPER_GAPFILL]
            tg = trn.dropna(subset=["SM_lag1d"])
            _, gap1 = fit_predict_trees(tg[g1], tg[TARGET], tg[g1].iloc[:2], paper=True, return_models=True)
            pa_te = pa[pa["site"].isin(test_sites)]
            for mode, A, AF, fill, Gm, GF, lag in (
                    ("paper loop (recursive, no drift control)", anc, PAPER16, True, gap, PAPER_GAPFILL, "SM_lag_prevS1"),
                    ("paper loop fed with in-situ lag", anc, PAPER16, True, gap, PAPER_GAPFILL, "SM_lag_prevS1"),
                    ("drift-controlled loop (satellite reset + 1-day lag)", anc0, PAPER16_NOLAG, False, gap1, g1,
                     "SM_lag1d")):
                m = "insitu" if "in-situ" in mode else ("drift" if "drift" in mode else "paper")
                out = run_loop(pdd, pa_te, test_sites, m, A, AF, fill, Gm, GF, lag)
                out["mode"], out["coords"], out["evaluation"] = mode, coords, evaluation
                series.append(out)
            log_progress(f"v2-1 daily loop {coords}: fold with {len(test_sites)} test stations done ({t()})")
    S = pd.concat(series, ignore_index=True)
    # persistence on the same days
    S = S.sort_values(["coords", "mode", "site", "date"])
    S["persistence_1d"] = S.groupby(["coords", "mode", "site"])[TARGET].shift(1)
    S.to_parquet(f"{TAB}/v2_1c_daily_loop_series.parquet")
    for (coords, ev, mode), g in S.groupby(["coords", "evaluation", "mode"]):
        g = g.dropna(subset=[TARGET])
        rows.append(dict(coords=coords, evaluation=ev, method=mode, **summarize(g, "pred")))
        if mode.startswith("paper loop (recursive"):
            gg = g.dropna(subset=["persistence_1d"])
            rows.append(dict(coords=coords, evaluation=ev, method="persistence (in-situ yesterday)",
                             **summarize(gg, "persistence_1d")))
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/v2_1c_daily_loop.csv", index=False)
    return out


def main():
    pa = pd.read_parquet(os.path.join(PROC, "paper_anchor.parquet"))
    pdaily = pd.read_parquet(os.path.join(PROC, "paper_daily.parquet"))
    for df in (pa, pdaily):
        df["site"] = df["site"].astype(str)
        df["scan_id"] = df["scan_id"].astype(str)
    log_progress("v2-1 started")
    a = part_a(pa)
    cols = ["coords", "features", "evaluation", "model", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]
    print(a[cols].round(3).to_string())
    b = part_b(pa)
    print(b[["step", "protocol", "variant", "n", "R2", "RMSE", "ubRMSE", "site_median_r"]].round(3).to_string())
    c = part_c(pa, pdaily)
    print(c[["coords", "evaluation", "method", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]].round(3).to_string())
    log_progress("v2-1 finished")


if __name__ == "__main__":
    main()
