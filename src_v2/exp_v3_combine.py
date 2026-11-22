"""
v3 combine: v2-3 OOF + TabPFN bagging (v3-1) + station-mean correction (v3-2) -> new rows, with
station-level cluster bootstrap CIs (1000 reps) and paired differences vs the v2 headline.
Rows whose inputs are missing or incomplete are skipped, never partially scored.

Writes results_v2/tables/v3_models.csv and v3_oof_<tag>.parquet
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from common_v2 import TAB, TARGET, folds_for, summarize
from exp_v2_2_protocols import load_eval_rows

PROTOS = ["leave-stations-out", "leave-one-region-out", "temporal(2022-23)"]
HEAD = "Hybrid(TabPFN+TempoSAR+Trees)"
BAGS = 5
N_BOOT = 1000


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


def add_bagging(res, proto, n_folds):
    files = glob.glob(os.path.join(TAB, "v3_tabpfn_bag", tag(proto), "f*_b*.parquet"))
    if not files:
        return res, "no bagging files"
    b = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    have = b.groupby("fold")["bag"].nunique()
    if len(have) < n_folds or (have < BAGS).any():
        return res, f"bagging incomplete: {len(have)}/{n_folds} folds, min bags {have.min()}"
    b0 = b[b.bag == 0][["site", "date", "pred"]].rename(columns={"pred": "TabPFN-bag0(rerun)"})
    bm = b.groupby(["site", "date"], as_index=False)["pred"].mean().rename(columns={"pred": "TabPFN-bag5"})
    res = res.merge(b0, on=["site", "date"], how="left").merge(bm, on=["site", "date"], how="left")
    res["Hybrid(TabPFN-bag5+TempoSAR+Trees)"] = (res["TabPFN-bag5"] + res["TempoSAR"] + res["Trees"]) / 3
    return res, "bagging ok"


def add_polaris_trees(res, proto):
    """v3-3 (exploratory): trees with POLARIS soil properties, and hybrids that use them."""
    fn = os.path.join(TAB, f"v3_polaris_trees_{tag(proto)}.parquet")
    if not os.path.exists(fn):
        return res, [], "no POLARIS trees"
    res = res.merge(pd.read_parquet(fn), on=["site", "date"], how="left")
    if res["Trees+POLARIS"].isna().any():
        return res.drop(columns="Trees+POLARIS"), [], "POLARIS trees incomplete"
    new = ["Trees+POLARIS", "Hybrid(TabPFN+TempoSAR+Trees+POLARIS)"]
    res[new[1]] = (res["TabPFN"] + res["TempoSAR"] + res["Trees+POLARIS"]) / 3
    if "TabPFN-bag5" in res:
        new.append("Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS)")
        res[new[-1]] = (res["TabPFN-bag5"] + res["TempoSAR"] + res["Trees+POLARIS"]) / 3
    pf = glob.glob(os.path.join(TAB, "v3_tabpfn_bag_polaris", tag(proto), "f*_b0.parquet"))
    if pf:
        b = pd.concat([pd.read_parquet(f) for f in pf], ignore_index=True)
        if b["fold"].nunique() == len(folds_for(proto, load_eval_rows())):
            res = res.merge(b[["site", "date", "pred"]].rename(columns={"pred": "TabPFN+POLARIS"}),
                            on=["site", "date"], how="left")
            res["Hybrid(TabPFN+POLARIS+TempoSAR+Trees+POLARIS)"] = (res["TabPFN+POLARIS"] + res["TempoSAR"]
                                                                    + res["Trees+POLARIS"]) / 3
            new += ["TabPFN+POLARIS", "Hybrid(TabPFN+POLARIS+TempoSAR+Trees+POLARIS)"]
    return res, new, "POLARIS trees ok"


REF4 = "Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS)"   # v4 reference, declared in progress.md
V4_CAND = "Hybrid(TabPFN-bag5+TempoSAR+POLARIS+Trees+POLARIS(exact, Se))"


def add_v4(res, proto):
    """v4 rows (pre-registered 2026-11-22): exact-coordinate POLARIS, Se target, TempoSAR+POLARIS."""
    new, msgs = [], []
    for pol, tgt in [("exact", "raw"), ("approx", "se"), ("exact", "se")]:
        fn = os.path.join(TAB, f"v4_trees_{pol}_{tgt}_{tag(proto)}.parquet")
        if os.path.exists(fn):
            t = pd.read_parquet(fn)
            c = [x for x in t.columns if x.startswith("Trees+")][0]
            if t[c].notna().all():
                res = res.merge(t, on=["site", "date"], how="left")
                new.append(c)
    for d, name in [("v4_temposar", "TempoSAR(rerun)"), ("v4_temposar_polaris", "TempoSAR+POLARIS")]:
        fs = glob.glob(os.path.join(TAB, d, tag(proto), "f*.parquet"))
        if fs:
            b = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            if b["fold"].nunique() == len(folds_for(proto, load_eval_rows())):
                res = res.merge(b[["site", "date", "pred"]].rename(columns={"pred": name}), on=["site", "date"],
                                how="left")
                new.append(name)
            else:
                msgs.append(f"{name} incomplete")
    if "TabPFN-bag5" in res:
        for temp in ["TempoSAR", "TempoSAR(rerun)", "TempoSAR+POLARIS"]:
            for trees in ["Trees+POLARIS", "Trees+POLARIS(exact)", "Trees+POLARIS(Se)", "Trees+POLARIS(exact, Se)"]:
                h = f"Hybrid(TabPFN-bag5+{temp}+{trees})"
                if temp in res and trees in res and h not in res:
                    res[h] = (res["TabPFN-bag5"] + res[temp] + res[trees]) / 3
                    new.append(h)
    return res, new, "; ".join(msgs) or f"v4 rows: {len(new)}"


def add_station_mean(res, proto, bases, sfx=""):
    fn = os.path.join(TAB, f"v3_station_mean{sfx}_{tag(proto)}.parquet")
    if not os.path.exists(fn):
        return res, [], f"no station-mean{sfx} file"
    sm = pd.read_parquet(fn).set_index("site")
    if not res["site"].isin(sm.index).all():
        return res, [], "station-mean file incomplete"
    new = []
    for c in bases:
        if c not in res:
            continue
        pbar = res.groupby("site")[c].transform("mean")
        m = res["site"].map(sm["m"])
        anom = res[c] - pbar
        for name, w, a in [("w=0.5", 0.5, 1.0),
                           ("w nested", res["site"].map(sm["w_only"]), 1.0),
                           ("w,a nested", res["site"].map(sm["w"]), res["site"].map(sm["a"]))]:
            col = f"{c} +StationMean{'+POLARIS' if sfx else ''}({name})"
            res[col] = a * anom + w * m + (1 - w) * pbar
            new.append(col)
    return res, new, "station-mean ok"


def bootstrap(res, cols, ref=HEAD, n=N_BOOT, seed=0):
    """Station-cluster bootstrap of pooled R2 and RMSE, and of the paired difference vs ref."""
    y = res[TARGET].to_numpy()
    codes, uniq = pd.factorize(res["site"])
    S = len(uniq)

    def sums(v):
        return np.bincount(codes, v, S)
    cnt, sy, syy = sums(np.ones_like(y)), sums(y), sums(y * y)
    sse = {c: sums((res[c].to_numpy() - y) ** 2) for c in cols + [ref]}
    W = np.random.RandomState(seed).multinomial(S, np.full(S, 1 / S), size=n).astype(float)
    N, SY, SYY = W @ cnt, W @ sy, W @ syy
    SST = SYY - SY ** 2 / N
    r2 = {c: 1 - (W @ sse[c]) / SST for c in sse}
    rm = {c: np.sqrt((W @ sse[c]) / N) for c in sse}
    out = {}
    for c in cols:
        d = r2[c] - r2[ref]
        out[c] = dict(R2_lo=np.percentile(r2[c], 2.5), R2_hi=np.percentile(r2[c], 97.5),
                      dR2_vs_head=np.mean(d), dR2_lo=np.percentile(d, 2.5), dR2_hi=np.percentile(d, 97.5),
                      dRMSE_lo=np.percentile(rm[c] - rm[ref], 2.5), dRMSE_hi=np.percentile(rm[c] - rm[ref], 97.5),
                      P_better=float(np.mean(d > 0)))
    return out


def main():
    df = load_eval_rows()
    rows = []
    for proto in PROTOS:
        res = pd.read_parquet(os.path.join(TAB, f"v2_3_oof_{tag(proto)}.parquet"))
        n_folds = len(folds_for(proto, df))
        res, msg_b = add_bagging(res, proto, n_folds)
        bases = [HEAD, "Trees", "Hybrid(TabPFN-bag5+TempoSAR+Trees)"]
        new = []
        if proto != "temporal(2022-23)":
            res, new, msg_s = add_station_mean(res, proto, bases)
            res, newp, msg_p = add_polaris_trees(res, proto)
            res, news, msg_q = add_station_mean(res, proto, [HEAD] + newp[1:], sfx="_polaris")
            new += newp + news
            res, new4, msg_4 = add_v4(res, proto)
            new += new4
            msg_s = f"{msg_s}; {msg_p}; {msg_q}; {msg_4}"
        else:
            msg_s = "station-mean / POLARIS not applied (temporal: stations seen in training)"
        print(f"{proto}: {msg_b}; {msg_s}", flush=True)
        cols = ["Trees", "TempoSAR", "TabPFN", HEAD] + [c for c in ("TabPFN-bag0(rerun)", "TabPFN-bag5",
                                                                     "Hybrid(TabPFN-bag5+TempoSAR+Trees)") if c in res] + new
        bs = bootstrap(res, cols)
        bs4 = bootstrap(res, cols, ref=REF4) if REF4 in res else {}
        for c in cols:
            extra = {f"{k.replace('head', 'v4ref')}": v for k, v in bs4.get(c, {}).items()
                     if k.startswith(("dR2", "P_better"))}
            extra = {(k if "v4ref" in k else k + "_v4ref"): v for k, v in extra.items()}
            rows.append(dict(protocol=proto, model=c, **summarize(res, c), **bs[c], **extra))
        res.to_parquet(os.path.join(TAB, f"v3_oof_{tag(proto)}.parquet"))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "v3_models.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60):
        print(out[["protocol", "model", "R2", "RMSE", "ubRMSE", "bias", "site_median_r", "R2_lo", "R2_hi",
                   "dR2_vs_head", "dR2_lo", "dR2_hi", "P_better"]].round(3).to_string())


if __name__ == "__main__":
    main()
