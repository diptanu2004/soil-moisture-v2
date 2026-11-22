"""
Experiment v3-1: TabPFN v2 context bagging (satellite-only, same rows / folds as v2-3).

Per fold, B station-stratified context sets of <= 10,000 training rows are drawn
(context_subsample with different seeds) and the TabPFN predictions are averaged.
Bag 0 uses exactly the v2-3 seeds (context seed = fold index, model seed = SEED), so bag 0 alone
is the v2-3 single-context TabPFN re-run with the installed TabPFN version; bags 1..B-1 use new seeds.
Nothing is fitted on test rows; the average is unweighted.

Each (fold, bag) prediction is written to its own file, so a crashed session resumes where it stopped.

  python exp_v3_tabpfn_bag.py --protocol leave-stations-out [--bags 5] [--folds 0,1,2] [--dry]

Writes <SM_RES>/tables/v3_tabpfn_bag/<tag>/f<fold>_b<bag>.parquet
Combine with:  python exp_v3_combine.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from common_v2 import (IMPROVED_V2, PROC, SEED, TAB, TABPFN_MAX_CONTEXT, TARGET, context_subsample, folds_for,
                       tabpfn_fit_predict)
from exp_v2_2_protocols import load_eval_rows


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


def bag_seeds(fold, bag):
    """bag 0 = the v2-3 seeds; later bags get new, deterministic seeds."""
    if bag == 0:
        return fold, SEED
    return 1000 * bag + fold, SEED + bag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--bags", type=int, default=5)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--n-est", type=int, default=4, help="kept at the v2-3 value so bagging is the only change")
    ap.add_argument("--polaris", action="store_true", help="v3-3 exploratory: add POLARIS soil columns")
    ap.add_argument("--dry", action="store_true", help="skip TabPFN (predict the context mean) to test I/O")
    a = ap.parse_args()

    df = load_eval_rows()
    feats = list(IMPROVED_V2)
    if a.polaris:
        pol = pd.read_parquet(os.environ.get("SM_POLARIS", os.path.join(os.path.dirname(PROC), "raw_v2",
                                                                         "polaris_stations.parquet")))
        df = df.merge(pol, on="site", how="left")
        feats += [c for c in pol.columns if c != "site"]
    sites = df["site"].to_numpy()
    X = df[feats].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    folds = folds_for(a.protocol, df)
    use = range(len(folds)) if a.folds == "all" else [int(f) for f in a.folds.split(",")]
    out = os.path.join(TAB, "v3_tabpfn_bag" + ("_polaris" if a.polaris else ""), tag(a.protocol))
    os.makedirs(out, exist_ok=True)
    print(f"{a.protocol}: {len(df)} rows, {len(feats)} features, {len(folds)} folds, bags={a.bags}, out={out}", flush=True)

    t0 = time.time()
    for f in use:
        tr, te = folds[f]
        bal = df.loc[tr, "region"].to_numpy() if a.protocol == "leave-one-region-out" else None
        for b in range(a.bags):
            fn = os.path.join(out, f"f{f}_b{b}.parquet")
            if os.path.exists(fn):
                continue
            cseed, mseed = bag_seeds(f, b)
            ctx = tr[context_subsample(sites[tr], TABPFN_MAX_CONTEXT, seed=cseed, balance=bal)]
            if a.dry:
                p = np.full(len(te), y[ctx].mean(), np.float32)
            else:
                p = tabpfn_fit_predict(X[ctx], y[ctx], X[te], n_estimators=a.n_est, seed=mseed)
            res = df.loc[te, ["site", "date"]].copy()
            res["pred"] = p
            res["fold"], res["bag"], res["ctx_seed"], res["model_seed"] = f, b, cseed, mseed
            res.to_parquet(fn + ".tmp")
            os.replace(fn + ".tmp", fn)
            print(f"{a.protocol} fold {f} bag {b} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"{a.protocol} finished ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
