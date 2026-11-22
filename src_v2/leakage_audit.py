"""Leakage audit for the v2 experiments. Writes results_v2/tables/v2_leakage_audit.txt

1. no in-situ-derived column in any satellite-only feature set
2. folds: test stations / regions / years never appear in training
3. single-feature correlation with the target (a near-1 feature would signal leakage)
4. random-split vs leave-stations-out gap for the satellite-only model (station-identity leakage)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_v2 import IMPROVED_SAT, IMPROVED_V2, PROTOCOLS, TAB, TARGET, folds_for
from exp_v2_2_protocols import FIXED16, load_eval_rows

INSITU = {"sm_5cm", "SM_lag1d", "SM_lag_prevS1", "stemp_5cm_C", "Persistence(prevS1)", "Persistence(1d)"}


def main():
    lines = []
    df = load_eval_rows()
    for name, F in (("improved v1 sat-only", IMPROVED_SAT), ("improved v2 sat-only", IMPROVED_V2),
                    ("paper16 fixed no-lag", FIXED16)):
        bad = sorted(set(F) & INSITU)
        lines.append(f"[1] {name}: {len(F)} features, in-situ-derived columns: {bad if bad else 'none'}")
    for proto in PROTOCOLS:
        for tr, te in folds_for(proto, df):
            if proto == "leave-stations-out":
                assert not set(df.site.iloc[tr]) & set(df.site.iloc[te])
            if proto == "leave-one-region-out":
                assert not set(df.region.iloc[tr]) & set(df.region.iloc[te])
            if proto.startswith("temporal"):
                assert df.date.iloc[tr].max() < df.date.iloc[te].min()
        lines.append(f"[2] {proto}: folds checked (disjoint stations/regions/time where required)")
    y = df[TARGET]
    cors = {}
    for f in IMPROVED_V2:
        x = pd.to_numeric(df[f], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() > 100 and x[m].std() > 0:
            cors[f] = np.corrcoef(x[m], y[m])[0, 1]
    c = pd.Series(cors).abs().sort_values(ascending=False)
    lines.append("[3] top |r| of single satellite/reanalysis features with in-situ 5 cm SM (pooled):")
    lines += [f"      {k:22s} {v:.3f}" for k, v in c.head(10).items()]
    p = os.path.join(TAB, "v2_2_protocols.csv")
    if os.path.exists(p):
        r = pd.read_csv(p)
        r = r[(r.features == "improved v2, satellite-only") & (r.model == "Ensemble")].set_index("protocol").R2
        lines.append("[4] satellite-only trees, R2 by protocol (random split >> station split = station-identity "
                     "leakage of random splits): " + ", ".join(f"{k}: {v:.3f}" for k, v in r.items()))
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(TAB, "v2_leakage_audit.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
