"""
Experiment 2C - few-shot station adaptation (transfer to a NEW station that has
only a short in-situ record).

For every station in a leave-stations-out fold, the first k labelled Sentinel-1
dates (k = 5, 10, 20; i.e. roughly the first 2-8 months) are used for adaptation
and the model is evaluated on all LATER dates of that station.

  none          out-of-station model as is
  bias          + mean-offset correction from the k samples
  linear        + mean/std rescaling (only when k >= 10)
  FT-Deep       pretrained TempoSAR fine-tuned (L2-SP) on the k samples

Uses the out-of-fold predictions saved by exp2_transfer.py part A.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import TAB, TARGET, group_folds, summarize
from deep import Prepared, Scaler, build_model, fine_tune, load_for_deep, predict, train
from exp2_transfer import D_KW, TRAIN_KW, split_inner

KS = (5, 10, 20)


def corrections(df, base_cols, k):
    out = []
    for s, g in df.groupby("site"):
        g = g.sort_values("date")
        if len(g) < k + 10:
            continue
        cal, ev = g.iloc[:k], g.iloc[k:].copy()
        for c in base_cols:
            off = (cal[TARGET] - cal[c]).mean()
            ev[f"{c}|none"] = ev[c]
            ev[f"{c}|bias"] = ev[c] + off
            if k >= 10 and cal[c].std() > 1e-6:
                ev[f"{c}|linear"] = (ev[c] - cal[c].mean()) / cal[c].std() * cal[TARGET].std() + cal[TARGET].mean()
            else:
                ev[f"{c}|linear"] = ev[f"{c}|bias"]
        ev["k"] = k
        out.append(ev)
    return pd.concat(out, ignore_index=True)


def main():
    oof = pd.read_parquet(f"{TAB}/exp2a_loso_preds.parquet")
    base_cols = ["TreeEns", "TempoSAR", "Hybrid"]
    rows = []
    allev = []
    for k in KS:
        ev = corrections(oof, base_cols, k)
        allev.append(ev)
        for c in base_cols:
            for corr in ("none", "bias", "linear"):
                rows.append(dict(k=k, model=c, correction=corr, **summarize(ev, f"{c}|{corr}")))

    # ---- deep fine-tuning on the first k samples of each held-out station ----
    lab, full = load_for_deep()
    lab = lab.dropna(subset=["SM_lag1d"]).reset_index(drop=True)
    P = Prepared(lab, full, L=10)
    sites = lab["site"].to_numpy()
    ft_rows = []
    t0 = time.time()
    for f, (tr, te) in enumerate(group_folds(sites, 10)):
        inner, val = split_inner(tr, sites)
        sc = Scaler().fit(P.subset(inner))
        Tin, Tva = sc.tensors(P.subset(inner)), sc.tensors(P.subset(val))
        pre = [train(build_model(Tin, **D_KW), Tin, Tva, seed=s, **TRAIN_KW) for s in (0, 1)]
        for s in np.unique(sites[te]):
            idx = np.where(sites == s)[0]
            idx = idx[np.argsort(lab.loc[idx, "date"].to_numpy())]
            for k in KS:
                if len(idx) < k + 10:
                    continue
                cal, ev = idx[:k], idx[k:]
                Tc, Te = sc.tensors(P.subset(cal)), sc.tensors(P.subset(ev))
                p = np.mean([predict(fine_tune(m, Tc, mode="l2sp", epochs=40, lr=3e-4, seed=i, alpha=1e-2), Te, sc)
                             for i, m in enumerate(pre)], 0)
                ft_rows.append(pd.DataFrame({"site": s, "date": lab.loc[ev, "date"].to_numpy(), "k": k,
                                             TARGET: lab.loc[ev, TARGET].to_numpy(), "FT-Deep": p}))
        print("fold", f, round(time.time() - t0), flush=True)
    ft = pd.concat(ft_rows, ignore_index=True)
    ev_all = pd.concat(allev, ignore_index=True)
    ft = ft.merge(ev_all[["site", "date", "k", "TreeEns|bias", "TempoSAR|bias", "Hybrid|bias"]],
                  on=["site", "date", "k"], how="left")
    ft["FT-Deep+Trees(bias)"] = 0.5 * (ft["FT-Deep"] + ft["TreeEns|bias"])
    for k, g in ft.groupby("k"):
        rows.append(dict(k=k, model="TempoSAR", correction="fine-tune(L2-SP)", **summarize(g, "FT-Deep")))
        rows.append(dict(k=k, model="Hybrid", correction="fine-tune+bias", **summarize(g, "FT-Deep+Trees(bias)")))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/exp2c_calibration.csv", index=False)
    ft.to_parquet(f"{TAB}/exp2c_ft_preds.parquet")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(res[["k", "model", "correction", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r",
                   "site_median_ubRMSE"]].round(3).to_string())


if __name__ == "__main__":
    main()
