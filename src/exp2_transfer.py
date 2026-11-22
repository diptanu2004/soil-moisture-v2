"""
Experiment 2 - stronger models and transfer learning (satellite-only retrieval).

Part A  leave-stations-out (10 folds, all 84 sites):
        regularised trees vs. TempoSAR (Transformer over the Sentinel-1 history)
        vs. hybrid (mean of TempoSAR and the tree ensemble)
Part B  regional transfer learning: for each target climate region, pretrain on
        all other regions, then adapt with in-situ data from only n target
        stations and test on the remaining (unseen) target stations.

usage: python exp2_transfer.py [A|B|AB]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from common import IMPROVED_SAT, TAB, TARGET, group_folds, make_tree_models, summarize
from deep import Prepared, Scaler, build_model, fine_tune, load_for_deep, predict, train

D_KW = dict(d=32, dropout=0.2)
TRAIN_KW = dict(epochs=60, lr=1e-3, wd=1e-3)


def split_inner(idx, sites, frac=0.12, seed=0):
    s = np.unique(sites[idx])
    rng = np.random.RandomState(seed)
    vs = rng.choice(s, max(1, int(round(len(s) * frac))), replace=False)
    return idx[~np.isin(sites[idx], vs)], idx[np.isin(sites[idx], vs)]


def deep_fit_predict(P, tr, te, sites, seeds=(0, 1, 2, 3, 4), use_seq=True):
    inner, val = split_inner(tr, sites)
    sc = Scaler().fit(P.subset(inner))
    Ttr, Tva, Tte = sc.tensors(P.subset(inner)), sc.tensors(P.subset(val)), sc.tensors(P.subset(te))
    ps = []
    for s in seeds:
        m = build_model(Ttr, use_seq=use_seq, **D_KW)
        m = train(m, Ttr, Tva, seed=s, **TRAIN_KW)
        ps.append(predict(m, Tte, sc))
    return np.mean(ps, 0)


def trees_fit_predict(X, y, tr, te):
    out = {}
    med = np.nanmedian(X[tr], 0)
    med = np.where(np.isfinite(med), med, 0)
    for name, m in make_tree_models(paper=False).items():
        if name == "RF":
            m.fit(np.where(np.isfinite(X[tr]), X[tr], med), y[tr])
            out[name] = m.predict(np.where(np.isfinite(X[te]), X[te], med))
        else:
            m.fit(X[tr], y[tr])
            out[name] = m.predict(X[te])
    out["TreeEns"] = np.mean([out["RF"], out["XGB"], out["LGBM"]], 0)
    return out


# ----------------------------------------------------------------------------
def part_a(lab, full):
    lab = lab.dropna(subset=["SM_lag1d"]).reset_index(drop=True)   # same rows as exp1
    P = Prepared(lab, full, L=10)
    sites = lab["site"].to_numpy()
    X, y = lab[IMPROVED_SAT].to_numpy(float), lab[TARGET].to_numpy(float)
    cols = ["RF", "XGB", "LGBM", "TreeEns", "TempoSAR", "MLP(noSeq)", "Hybrid"]
    pred = {c: np.full(len(lab), np.nan) for c in cols}
    t0 = time.time()
    for f, (tr, te) in enumerate(group_folds(sites, 10)):
        t = trees_fit_predict(X, y, tr, te)
        for k, v in t.items():
            pred[k][te] = v
        pred["TempoSAR"][te] = deep_fit_predict(P, tr, te, sites)
        pred["MLP(noSeq)"][te] = deep_fit_predict(P, tr, te, sites, seeds=(0, 1, 2), use_seq=False)
        pred["Hybrid"][te] = 0.5 * (pred["TempoSAR"][te] + pred["TreeEns"][te])
        print("fold", f, round(time.time() - t0), "s", flush=True)
    rows = []
    for c in cols:
        lab[c] = pred[c]
        rows.append(dict(model=c, **summarize(lab, c)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/exp2a_loso_models.csv", index=False)
    print(res.round(3).to_string())
    best = res.sort_values("RMSE").iloc[0]["model"]
    lab[["site", "date", "region", TARGET] + cols].to_parquet(f"{TAB}/exp2a_loso_preds.parquet")
    lab[["site", "date"]].assign(pred=lab[best]).to_parquet(f"{TAB}/exp2_loso_satonly_best_preds.parquet")
    return res


# ----------------------------------------------------------------------------
def part_b(lab, full, regions=("Southwest", "West", "N.Rockies&Plains", "South", "Northwest"),
           n_ft_list=(1, 3, "all"), seeds=(0, 1)):
    lab = lab.reset_index(drop=True)
    P = Prepared(lab, full, L=10)
    sites = lab["site"].to_numpy()
    X, y = lab[IMPROVED_SAT].to_numpy(float), lab[TARGET].to_numpy(float)
    records = []
    t0 = time.time()
    for R in regions:
        src = np.where(lab["region"] != R)[0]
        tgt = np.where(lab["region"] == R)[0]
        # ---- pretraining on source regions (deep) + zero-shot trees ----
        inner, val = split_inner(src, sites)
        sc = Scaler().fit(P.subset(inner))
        Tin, Tva = sc.tensors(P.subset(inner)), sc.tensors(P.subset(val))
        pre = []
        for s in seeds:
            m = build_model(Tin, **D_KW)
            pre.append(train(m, Tin, Tva, seed=s, **TRAIN_KW))
        print(R, "pretrained", round(time.time() - t0), flush=True)

        zs_tree_all = pd.Series(trees_fit_predict(X, y, src, tgt)["TreeEns"], index=tgt)
        tsites = np.unique(sites[tgt])
        k = min(5, len(tsites))
        for f, (ptr, pte) in enumerate(group_folds(sites[tgt], k, seed=1)):
            pool_idx, test_idx = tgt[ptr], tgt[pte]
            Tte = sc.tensors(P.subset(test_idx))
            base = lab.iloc[test_idx][["site", "date", "region", TARGET]].copy()
            base["fold"] = f
            zs_deep = np.mean([predict(m, Tte, sc) for m in pre], 0)
            zs_tree = zs_tree_all.loc[test_idx].to_numpy()
            pool_sites = np.unique(sites[pool_idx])
            rng = np.random.RandomState(100 + f)
            for n_ft in n_ft_list:
                n = len(pool_sites) if n_ft == "all" else min(n_ft, len(pool_sites))
                chosen = rng.choice(pool_sites, n, replace=False)
                ft_idx = pool_idx[np.isin(sites[pool_idx], chosen)]
                Tft = sc.tensors(P.subset(ft_idx))
                out = base.copy()
                out["n_ft_sites"] = str(n_ft)
                out["ZeroShot-Deep"] = zs_deep
                out["ZeroShot-Trees"] = zs_tree
                # target-only (no transfer)
                out["Scratch-Trees"] = trees_fit_predict(X, y, ft_idx, test_idx)["TreeEns"]
                sc_t = Scaler().fit(P.subset(ft_idx))
                Tft_t, Tte_t = sc_t.tensors(P.subset(ft_idx)), sc_t.tensors(P.subset(test_idx))
                out["Scratch-Deep"] = np.mean([
                    predict(train(build_model(Tft_t, **D_KW), Tft_t, None, seed=s, epochs=60, lr=1e-3, wd=1e-3),
                            Tte_t, sc_t) for s in seeds], 0)
                # pooled trees = "add more stations from other regions"
                out["Pooled-Trees"] = trees_fit_predict(X, y, np.concatenate([src, ft_idx]), test_idx)["TreeEns"]
                # transfer learning
                out["FT-Deep(head)"] = np.mean([predict(fine_tune(m, Tft, mode="head", epochs=30, lr=1e-3, seed=s),
                                                        Tte, sc) for s, m in zip(seeds, pre)], 0)
                out["FT-Deep(L2-SP)"] = np.mean([predict(fine_tune(m, Tft, mode="l2sp", epochs=30, lr=3e-4, seed=s),
                                                         Tte, sc) for s, m in zip(seeds, pre)], 0)
                out["Hybrid(FT-Deep+Pooled-Trees)"] = 0.5 * (out["FT-Deep(L2-SP)"] + out["Pooled-Trees"])
                records.append(out)
            print(R, "fold", f, round(time.time() - t0), flush=True)
    P_all = pd.concat(records, ignore_index=True)
    P_all.to_parquet(f"{TAB}/exp2b_transfer_preds.parquet")
    methods = ["ZeroShot-Trees", "ZeroShot-Deep", "Scratch-Trees", "Scratch-Deep", "Pooled-Trees",
               "FT-Deep(head)", "FT-Deep(L2-SP)", "Hybrid(FT-Deep+Pooled-Trees)"]
    rows = []
    for (R, n_ft), g in P_all.groupby(["region", P_all["n_ft_sites"].astype(str)]):
        for mth in methods:
            rows.append(dict(region=R, n_ft_sites=n_ft, method=mth, **summarize(g, mth)))
    for n_ft, g in P_all.groupby(P_all["n_ft_sites"].astype(str)):
        for mth in methods:
            rows.append(dict(region="ALL", n_ft_sites=n_ft, method=mth, **summarize(g, mth)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/exp2b_transfer.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(res[res.region == "ALL"][["n_ft_sites", "method", "R2", "RMSE", "ubRMSE", "site_median_r",
                                        "site_median_ubRMSE"]].round(3).to_string())
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    lab, full = load_for_deep()
    if "A" in which:
        part_a(lab, full)
    if "B" in which:
        part_b(lab, full)
