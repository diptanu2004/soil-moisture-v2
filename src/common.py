"""Shared utilities: data loading, feature sets, metrics, splits, tree models."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 42
MIN_SAMPLES_PER_SITE = 20
TARGET = "sm_5cm"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def soil_hydraulics(df: pd.DataFrame) -> pd.DataFrame:
    """Saxton & Rawls (2006) pedotransfer (OM assumed 2.5 %) -> wilting point,
    field capacity and saturation (% vol) from sand/clay fractions."""
    S = df["sand_fract"].astype(float) / 100
    C = df["clay_fract"].astype(float) / 100
    OM = 2.5
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    tsat = t33 + ts33 - 0.097 * S + 0.043
    out = df.copy()
    out["theta_wp"] = 100 * t1500
    out["theta_fc"] = 100 * t33
    out["theta_sat"] = 100 * tsat
    return out


def add_site_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Site descriptors computed from the site's own satellite record (no in-situ)."""
    g = df.groupby("site")
    for c in ["precip2d", "NDVI", "NDMI", "LAI", "VH"]:
        df[f"site_mean_{c}"] = g[c].transform("mean")
    df["site_wet_frac"] = g["precip2d"].transform(lambda s: (s > 1).mean())
    df["VV_site_q05"] = g["VV"].transform(lambda s: s.quantile(0.05))
    df["VV_site_q95"] = g["VV"].transform(lambda s: s.quantile(0.95))
    return df


def load_full() -> pd.DataFrame:
    """All Sentinel-1 rows (labelled or not) with derived site descriptors."""
    a = pd.read_parquet(os.path.join(PROC, "anchor.parquet"))
    a = soil_hydraulics(a)
    a = add_site_climatology(a)
    return a


def load_anchor(min_samples: int = MIN_SAMPLES_PER_SITE) -> pd.DataFrame:
    a = load_full()
    a = a.dropna(subset=[TARGET])
    n = a.groupby("site")[TARGET].transform("size")
    a = a[n >= min_samples].reset_index(drop=True)
    # paper-style (incorrect) SAR indices computed directly on dB values
    a["ratio_dBdiv"] = a["VV"] / a["VH"]
    a["SPAN_dBsum"] = a["VV"] + a["VH"]
    a["RVI_dB"] = 4 * a["VH"] / (a["VV"] + a["VH"])
    a["lc_code"] = a["lc_group"].astype("category").cat.codes
    a["has_VH"] = a["VH"].notna().astype(int)
    return a


def load_daily() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PROC, "scan_daily.parquet"))


# ----------------------------------------------------------------------------
# feature sets
# ----------------------------------------------------------------------------
# Analogue of the paper's 16 anchor features (S2 -> Landsat-8 indices, MODIS NDVI/EVI -> MODIS LAI,
# slope unavailable). SAR indices computed the way the repo does (on dB values).
PAPER_FEATS = ["VV", "VH", "ratio_dBdiv", "RVI_dB", "SPAN_dBsum", "NDVI", "NDMI", "EVI", "LAI",
               "elev", "DOY", "Month", "SM_lag_prevS1", "precip2d"]
PAPER_FEATS_NOLAG = [f for f in PAPER_FEATS if f != "SM_lag_prevS1"]

# Improved, satellite-only feature set (no in-situ inputs at prediction time)
IMPROVED_SAT = [
    "VV", "VH", "CR_dB", "RVI", "SPAN_dB", "has_VH", "acq_hour",
    "dVV", "dVH", "days_since_prev_s1",
    "VV_cd", "VV_anom", "VH_anom", "VV_site_med", "VV_site_iqr",
    "NDVI", "NDMI", "EVI", "NDVI_anom", "LAI", "precip2d",
    "doy_sin", "doy_cos", "elev",
    "clay_fract", "sand_fract", "theta_wp", "theta_fc", "theta_sat", "lc_code",
    # site climatology from the site's own satellite record
    "site_mean_precip2d", "site_mean_NDVI", "site_mean_NDMI", "site_mean_LAI", "site_mean_VH",
    "site_wet_frac", "VV_site_q05", "VV_site_q95",
]
# Improved, in-situ-assisted (consistent daily lag in training AND application)
IMPROVED_LAG = IMPROVED_SAT + ["SM_lag1d"]


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 3:
        return dict(n=len(y), R2=np.nan, RMSE=np.nan, ubRMSE=np.nan, bias=np.nan, r=np.nan)
    err = p - y
    bias = err.mean()
    rmse = np.sqrt((err ** 2).mean())
    ubrmse = np.sqrt(max(rmse ** 2 - bias ** 2, 0))
    r = np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan
    r2 = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(y)), R2=r2, RMSE=rmse, ubRMSE=ubrmse, bias=bias, r=r)


def per_site_metrics(df: pd.DataFrame, pred_col: str, y_col: str = TARGET) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("site"):
        m = metrics(g[y_col], g[pred_col])
        m["site"] = s
        rows.append(m)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, pred_col: str, y_col: str = TARGET) -> dict:
    """Pooled metrics + median per-site r and ubRMSE (standard in SM validation)."""
    out = metrics(df[y_col], df[pred_col])
    ps = per_site_metrics(df, pred_col, y_col)
    out["site_median_r"] = ps["r"].median()
    out["site_median_ubRMSE"] = ps["ubRMSE"].median()
    out["site_median_R2"] = ps["R2"].median()
    return out


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def group_folds(sites: np.ndarray, n_splits: int = 10, seed: int = SEED):
    """Leave-stations-out folds: each site appears in exactly one test fold."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(np.unique(sites)))
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, n_splits)
    for ch in chunks:
        test = np.isin(sites, ch)
        yield np.where(~test)[0], np.where(test)[0]


def random_split(n: int, test_size: float = 0.2, seed: int = SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return tr, te


# ----------------------------------------------------------------------------
# tree models (paper configuration)
# ----------------------------------------------------------------------------
def make_tree_models(paper: bool = True, seed: int = SEED):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    if paper:
        return {
            "RF": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=2),
            "XGB": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=2),
            "LGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed,
                                  verbose=-1, n_jobs=2),
        }
    # regularised versions for cross-site generalisation
    return {
        "RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                    random_state=seed, n_jobs=2),
        "XGB": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                            colsample_bytree=0.7, min_child_weight=5, reg_lambda=5.0,
                            random_state=seed, n_jobs=2),
        "LGBM": LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=-1,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              min_child_samples=20, reg_lambda=5.0, random_state=seed,
                              verbose=-1, n_jobs=2),
    }


def fit_predict_trees(Xtr, ytr, Xte, paper=True, impute_for_rf=True):
    """Fit RF/XGB/LGBM and their mean ensemble. RF needs NaNs imputed."""
    preds = {}
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    for name, m in make_tree_models(paper).items():
        if name == "RF" and impute_for_rf:
            Xa = np.where(np.isfinite(Xtr), Xtr, med)
            Xb = np.where(np.isfinite(Xte), Xte, med)
            m.fit(Xa, ytr)
            preds[name] = m.predict(Xb)
        else:
            m.fit(Xtr, ytr)
            preds[name] = m.predict(Xte)
    preds["Ensemble"] = np.mean([preds[k] for k in ("RF", "XGB", "LGBM")], axis=0)
    return preds
