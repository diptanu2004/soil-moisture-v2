"""
Experiment v2-4: few-shot adaptation to a NEW station (leave-stations-out folds of v2-3).

For each held-out station the first k labelled Sentinel-1 dates (k = 5, 10, 20) are used for
adaptation; evaluation is on all LATER dates of that station (no overlap, no look-ahead).

  none / bias / linear   statistical corrections of the out-of-station predictions
  FT-TempoSAR            pretrained TempoSAR fine-tuned with L2-SP on the k samples
  TabPFN in-context      the k samples are added to TabPFN's context set (no training)
  Hybrids                FT-TempoSAR + Trees(bias); TabPFN in-context + Trees(bias)

Writes results_v2/tables/v2_4_fewshot.csv and v2_4_fewshot_preds.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_v2 import (IMPROVED_V2, TAB, TARGET, TABPFN_MAX_CONTEXT, Timer, context_subsample, folds_for,
                       load_full_anchor, log_progress, summarize, tabpfn_fit_predict)
from deep_v2 import Prepared, fine_tune, predict, pretrain
from exp_v2_2_protocols import load_eval_rows

KS = (5, 10, 20)
BASE = ["Trees", "TempoSAR", "TabPFN", "Hybrid(TempoSAR+Trees)", "Hybrid(TabPFN+Trees)"]


def main():
    df = load_eval_rows()
    full = load_full_anchor()
    full = full[full["site"].isin(df["site"].unique())].reset_index(drop=True)
    P = Prepared(df, full, L=10)
    sites = df["site"].to_numpy()
    oof = pd.read_parquet(f"{TAB}/v2_3_oof_leave-stations-out.parquet")
    df = df.merge(oof[["site", "date"] + BASE], on=["site", "date"], how="left")
    X = df[IMPROVED_V2].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    log_progress("v2-4 few-shot started")
    t = Timer()
    out = []
    for f, (tr, te) in enumerate(folds_for("leave-stations-out", df)):
        models, sc = pretrain(P, tr, sites, seeds=(0, 1))
        ctx_full = tr[context_subsample(sites[tr], TABPFN_MAX_CONTEXT, seed=f)]
        splits = {}
        for s in np.unique(sites[te]):
            idx = np.where(sites == s)[0]
            idx = idx[np.argsort(df.loc[idx, "date"].to_numpy())]
            for k in KS:
                if len(idx) >= k + 10:
                    splits[(s, k)] = (idx[:k], idx[k:])
        # TabPFN in-context adaptation: one fit per fold and k. The context holds the training
        # stations plus the first k samples of EVERY held-out station of the fold (each station's
        # later dates are predicted; no station sees its own later data)
        icl = {}
        for k in KS:
            cals = [c for (s, kk), (c, _) in splits.items() if kk == k]
            evs = [e for (s, kk), (_, e) in splits.items() if kk == k]
            if not cals:
                continue
            cal_all, ev_all = np.concatenate(cals), np.concatenate(evs)
            ctx = np.concatenate([ctx_full[: TABPFN_MAX_CONTEXT - len(cal_all)], cal_all])
            icl[k] = dict(zip(ev_all, tabpfn_fit_predict(X[ctx], y[ctx], X[ev_all])))
        for (s, k), (cal, ev) in splits.items():
            r = df.loc[ev, ["site", "date", "region", TARGET] + BASE].copy()
            r["k"] = k
            c = df.loc[cal]
            for b in BASE:
                off = (c[TARGET] - c[b]).mean()
                r[f"{b}|none"] = r[b]
                r[f"{b}|bias"] = r[b] + off
                if k >= 10 and c[b].std() > 1e-6:
                    r[f"{b}|linear"] = (r[b] - c[b].mean()) / c[b].std() * c[TARGET].std() + c[TARGET].mean()
                else:
                    r[f"{b}|linear"] = r[f"{b}|bias"]
            Tc, Te = sc.tensors(P.subset(cal)), sc.tensors(P.subset(ev))
            r["FT-TempoSAR"] = np.mean([predict(fine_tune(m, Tc, seed=i), Te, sc) for i, m in enumerate(models)], 0)
            r["TabPFN in-context"] = [icl[k][i] for i in ev]
            r["Hybrid(FT-TempoSAR+Trees|bias)"] = 0.5 * (r["FT-TempoSAR"] + r["Trees|bias"])
            r["Hybrid(TabPFN in-context+Trees|bias)"] = 0.5 * (r["TabPFN in-context"] + r["Trees|bias"])
            out.append(r)
        print("fold", f, t(), flush=True)
        log_progress(f"v2-4 fold {f} done ({t()})")
    R = pd.concat(out, ignore_index=True)
    R.to_parquet(f"{TAB}/v2_4_fewshot_preds.parquet")
    rows = []
    methods = [f"{b}|{c}" for b in BASE for c in ("none", "bias", "linear")] + [
        "FT-TempoSAR", "TabPFN in-context", "Hybrid(FT-TempoSAR+Trees|bias)", "Hybrid(TabPFN in-context+Trees|bias)"]
    for k, g in R.groupby("k"):
        for m in methods:
            rows.append(dict(k=k, method=m, **summarize(g, m)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/v2_4_fewshot.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(res[["k", "method", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]].round(3).to_string())
    log_progress("v2-4 finished")


if __name__ == "__main__":
    main()
