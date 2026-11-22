"""
Experiment v4-3: TempoSAR with POLARIS static inputs (pre-registered 2026-11-22, see progress.md).

Same rows, folds, architecture, seeds (0, 1, 2) and training as v2-3 (deep_v2.deep_fit_predict). With
--polaris, the 20 POLARIS ~1 km window means (10 properties x 0-5 / 5-15 cm, the v3 table) are appended to
TempoSAR's static branch. Missing values go to 0 + missing flag, as in src/deep.py.
Without --polaris this is the control: a TempoSAR re-run on the same hardware (Kaggle T4), because GPU
training is not bit-reproducible across machines.

  python exp_v4_temposar.py --protocol leave-stations-out [--polaris] [--folds 0,1]
Writes <SM_RES>/tables/v4_temposar[_polaris]/<tag>/f<fold>.parquet  (site, date, pred)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import deep_v2 as DV
from common_v2 import PROC, TAB, folds_for, load_full_anchor
from exp_v2_2_protocols import load_eval_rows


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--polaris", action="store_true")
    ap.add_argument("--folds", default="all")
    a = ap.parse_args()

    df = load_eval_rows()
    full = load_full_anchor()
    full = full[full["site"].isin(df["site"].unique())].reset_index(drop=True)
    if a.polaris:
        pol = pd.read_parquet(os.environ.get("SM_POLARIS", os.path.join(os.path.dirname(PROC), "raw_v2",
                                                                         "polaris_stations.parquet")))
        pcols = [c for c in pol.columns if c.startswith("pol_") and c.endswith("_win")]
        df = df.merge(pol[["site"] + pcols], on="site", how="left")
        DV.D.STATIC_FEATS = DV.D.STATIC_FEATS + pcols
    P = DV.Prepared(df, full, L=10)
    sites = df["site"].to_numpy()
    folds = folds_for(a.protocol, df)
    use = range(len(folds)) if a.folds == "all" else [int(f) for f in a.folds.split(",")]
    out = os.path.join(TAB, "v4_temposar" + ("_polaris" if a.polaris else ""), tag(a.protocol))
    os.makedirs(out, exist_ok=True)
    print(f"{a.protocol}: device={DV.DEVICE}, {len(DV.D.STATIC_FEATS)} static features, {len(folds)} folds", flush=True)
    t0 = time.time()
    for f in use:
        fn = os.path.join(out, f"f{f}.parquet")
        if os.path.exists(fn):
            continue
        tr, te = folds[f]
        p = DV.deep_fit_predict(P, tr, te, sites)[0]
        res = df.loc[te, ["site", "date"]].copy()
        res["pred"], res["fold"] = p, f
        res.to_parquet(fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        print(f"{a.protocol} fold {f} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"{a.protocol} finished ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
