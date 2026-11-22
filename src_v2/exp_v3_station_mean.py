"""
Experiment v3-2: two-stage station-mean correction (satellite-only, LOSO and LORO, same folds as v2-3).

About 58 % of the hybrid's MSE at unseen stations is a constant per-station offset. Stage 2 predicts
each station's mean 5 cm soil moisture from station-level aggregates of the satellite-only v2 features
(means of all IMPROVED_V2 features + the spread of a few dynamic ones). No in-situ input; the target of
stage 2 is the in-situ station mean of the *training* stations only.

Correction applied to any prediction p at a test station s (pbar_s = mean of p over the station's rows):
    p' = a * (p - pbar_s) + w * m_s + (1 - w) * pbar_s
  w = 0.5, a = 1      fixed (declared before results)
  w, a nested         fitted on 5 inner station folds of the training stations, with a row-level LGBM
                      as a cheap proxy for the hybrid (the hybrid's members are too costly to re-run nested)

--polaris (exploratory, v3-3) adds POLARIS 30 m soil properties to the station table.

Writes <SM_RES>/tables/v3_station_mean[_polaris]_<tag>.parquet  (one row per test station: site, fold, m, w, a, w_only)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from common_v2 import IMPROVED_V2, PROC, SEED, TAB, TARGET, Timer, folds_for, group_folds, log_progress
from exp_v2_2_protocols import load_eval_rows

DYN = ["smap_l4_sm", "smap_l3e_sm", "smap_l4_rz", "era5_swvl1", "era5_swvl2", "VV", "VH", "VV_norm40",
       "NDVI", "LST_day", "rain_30d", "API_090"]
PROTOS = ["leave-stations-out", "leave-one-region-out"]
N_INNER = 5


def tag(proto):
    return proto.replace("/", "_").replace("(", "_").replace(")", "")


POLARIS = os.environ.get("SM_POLARIS", os.path.join(os.path.dirname(PROC), "raw_v2", "polaris_stations.parquet"))


def station_table(df, polaris=False):
    g = df.groupby("site")
    st = g[IMPROVED_V2].mean()
    st = st.join(g[DYN].std().add_suffix("_sd")).join(g[DYN].quantile(0.9).add_suffix("_q90"))
    st = st.join(g[DYN].quantile(0.1).add_suffix("_q10"))
    if polaris:  # static 30 m SSURGO-based soil properties (map product, no in-situ input)
        pol = pd.read_parquet(POLARIS).set_index("site")
        st = st.join(pol, how="left")
    st["y_mean"] = g[TARGET].mean()
    return st


def fit_station_model(St, sites_tr):
    from sklearn.ensemble import RandomForestRegressor
    A = St.loc[sites_tr]
    feats = [c for c in St.columns if c != "y_mean"]
    X = A[feats].to_numpy(float)
    med = np.nanmedian(X, 0)
    med = np.where(np.isfinite(med), med, 0)
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.33, random_state=SEED,
                              n_jobs=os.cpu_count())
    m.fit(np.where(np.isfinite(X), X, med), A["y_mean"].to_numpy())

    def predict(sites):
        Z = St.loc[sites, feats].to_numpy(float)
        return pd.Series(m.predict(np.where(np.isfinite(Z), Z, med)), index=sites)
    return predict


def proxy_lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, min_child_samples=20, reg_lambda=5.0, random_state=SEED,
                         verbose=-1, n_jobs=os.cpu_count())


def fit_w_a(y, p, site, m_site):
    """Inner-fold fit of w (with a = 1) and of (w, a) jointly, by least squares; clipped to sane ranges."""
    d = pd.DataFrame({"y": y, "p": p, "s": site})
    pbar = d.groupby("s")["p"].transform("mean").to_numpy()
    m = d["s"].map(m_site).to_numpy()
    dm, dp, t = m - pbar, p - pbar, y - pbar
    w_only = float(np.clip(np.dot(dm, y - p) / np.dot(dm, dm), 0, 1))
    A = np.c_[dp, dm]
    a, w = np.linalg.lstsq(A, t, rcond=None)[0]
    return w_only, float(np.clip(w, 0, 1)), float(np.clip(a, 0.8, 1.5))


def main(protos=PROTOS, max_folds=None, polaris=False):
    sfx = "_polaris" if polaris else ""
    df = load_eval_rows()
    St = station_table(df, polaris)
    X = df[IMPROVED_V2].to_numpy(np.float32)
    y = df[TARGET].to_numpy(float)
    sites = df["site"].to_numpy()
    t = Timer()
    log_progress(f"v3-2{sfx} station-mean started ({len(St)} stations, {St.shape[1] - 1} station features)")
    for proto in protos:
        folds = folds_for(proto, df)[:max_folds]
        out = []
        for f, (tr, te) in enumerate(folds):
            s_tr, s_te = np.unique(sites[tr]), np.unique(sites[te])
            # inner station folds on the training stations -> proxy predictions + stage-2 predictions
            P, M = np.full(len(tr), np.nan), {}
            for itr, iva in group_folds(sites[tr], N_INNER, seed=SEED + f):
                a_tr, a_va = tr[itr], tr[iva]
                P[iva] = proxy_lgbm().fit(X[a_tr], y[a_tr]).predict(X[a_va])
                M.update(fit_station_model(St, np.unique(sites[a_tr]))(np.unique(sites[a_va])).to_dict())
            w_only, w, a = fit_w_a(y[tr], P, sites[tr], M)
            m_te = fit_station_model(St, s_tr)(s_te)
            out.append(pd.DataFrame({"site": s_te, "fold": f, "m": m_te.to_numpy(), "w_only": w_only,
                                     "w": w, "a": a}))
            print(f"{proto} fold {f}: w_only={w_only:.2f} w={w:.2f} a={a:.2f} ({t()})", flush=True)
        res = pd.concat(out, ignore_index=True)
        res.to_parquet(os.path.join(TAB, f"v3_station_mean{sfx}_{tag(proto)}.parquet"))
        log_progress(f"v3-2{sfx} {proto} done ({t()})")
    log_progress(f"v3-2{sfx} finished")


if __name__ == "__main__":
    mf = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--max-folds=")), None)
    main(max_folds=mf, polaris="--polaris" in sys.argv)
