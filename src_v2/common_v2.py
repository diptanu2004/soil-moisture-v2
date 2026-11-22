"""Shared utilities for the 2019-2023 (v2) experiments: data, feature sets, splits, models.

Metrics (R2, RMSE, ubRMSE, bias, per-station median r) are the ones of src/common.py.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import metrics, per_site_metrics, soil_hydraulics, summarize  # noqa: E402,F401

PROC = os.environ.get("SM_PROC", os.path.join(ROOT, "data", "processed_v2"))
RES = os.environ.get("SM_RES", os.path.join(ROOT, "results_v2"))
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

SEED = 42
TARGET = "sm_5cm"
MIN_SAMPLES_PER_SITE = 20
TEMPORAL_SPLIT = "2022-01-01"      # train 2019-2021, test 2022-2023 (after the Sentinel-1B failure)
N_JOBS = 8

# ----------------------------------------------------------------------------
# feature sets
# ----------------------------------------------------------------------------
# The paper's 16 anchor features, built the paper's way (see gee/assemble_v2.py: paper_anchor)
PAPER16 = ["P_VV", "P_VH", "P_ratio", "P_RVI", "P_SPAN", "P_S2_NDVI", "P_S2_NDMI", "P_S2_EVI",
           "P_MODIS_NDVI", "P_MODIS_EVI", "P_elev", "P_slope", "P_DOY", "P_Month", "SM_lag_prevS1", "P_precip"]
PAPER16_NOLAG = [f for f in PAPER16 if f != "SM_lag_prevS1"]
# paper's gap-fill (daily) model: the daily-available subset of the anchor features
PAPER_GAPFILL = ["SM_lag_prevS1", "P_precip", "P_MODIS_NDVI", "P_MODIS_EVI", "P_elev", "P_slope", "P_DOY", "P_Month"]

# Satellite-only improved set of the 2015-2018 package (src/common.py IMPROVED_SAT)
IMPROVED_SAT = [
    "VV", "VH", "CR_dB", "RVI", "SPAN_dB", "has_VH", "acq_hour",
    "dVV", "dVH", "days_since_prev_s1",
    "VV_cd", "VV_anom", "VH_anom", "VV_site_med", "VV_site_iqr",
    "NDVI", "NDMI", "EVI", "NDVI_anom", "LAI", "precip2d",
    "doy_sin", "doy_cos", "elev",
    "clay_fract", "sand_fract", "theta_wp", "theta_fc", "theta_sat", "lc_code",
    "site_mean_precip2d", "site_mean_NDVI", "site_mean_NDMI", "site_mean_LAI", "site_mean_VH",
    "site_wet_frac", "VV_site_q05", "VV_site_q95",
]
# v2 additions: incidence-angle normalisation / orbit, antecedent rain, ERA5-Land, SMAP, MODIS NBAR + LST,
# extra S2 indices, terrain wetness and SoilGrids
V2_EXTRA = [
    "angle", "VV_norm40", "pass_asc", "relorb_n_obs",
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "API_085", "API_090", "API_095", "days_since_rain",
    "era5_swvl1", "era5_swvl2", "era5_t2m", "era5_stl1", "era5_pev", "era5_evap", "era5_ssr", "era5_snow",
    "smap_l4_sm", "smap_l4_rz", "smap_l4_wetness", "smap_l3e_sm",
    "LST_day", "LST_night", "dLST", "MODIS_NDVI", "MODIS_NDMI", "MODIS_EVI",
    "S2_BSI", "S2_NDWI", "s2_age_days", "TWI", "slope", "bdod", "soc", "hnd",
]
IMPROVED_V2 = IMPROVED_SAT + V2_EXTRA


def with_lag(feats, lag="SM_lag1d"):
    return feats + [lag]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _finish(a: pd.DataFrame) -> pd.DataFrame:
    a = soil_hydraulics(a)
    a["lc_code"] = a["lc_group"].astype("category").cat.codes
    a["site"] = a["site"].astype(str)
    return a


def load_anchor(min_samples=MIN_SAMPLES_PER_SITE, labelled=True) -> pd.DataFrame:
    a = pd.read_parquet(os.path.join(PROC, "anchor.parquet"))
    a = _finish(a)
    if labelled:
        a = a.dropna(subset=[TARGET])
        a = a[a.groupby("site")[TARGET].transform("size") >= min_samples]
    return a.reset_index(drop=True)


def load_full_anchor() -> pd.DataFrame:
    """All S1 rows (labelled or not), used for TempoSAR context."""
    return load_anchor(labelled=False)


def load_daily() -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(PROC, "daily.parquet"))
    return _finish(d)


def load_paper_anchor() -> pd.DataFrame:
    a = pd.read_parquet(os.path.join(PROC, "paper_anchor.parquet"))
    a["site"] = a["site"].astype(str)
    return a


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def random_split(n, test_size=0.2, seed=SEED):
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(n), test_size=test_size, random_state=seed)


def group_folds(groups, n_splits=10, seed=SEED):
    """Leave-stations-out: each station in exactly one test fold."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(np.unique(groups)))
    rng.shuffle(uniq)
    for ch in np.array_split(uniq, min(n_splits, len(uniq))):
        test = np.isin(groups, ch)
        yield np.where(~test)[0], np.where(test)[0]


def region_folds(regions):
    for r in sorted(pd.unique(regions)):
        yield np.where(regions != r)[0], np.where(regions == r)[0]


def temporal_fold(dates, split=TEMPORAL_SPLIT):
    dates = pd.to_datetime(dates)
    yield np.where(dates < split)[0], np.where(dates >= split)[0]


def folds_for(protocol, df, n_splits=10):
    if protocol == "random80/20":
        tr, te = random_split(len(df))
        return [(tr, te)]
    if protocol == "leave-stations-out":
        return list(group_folds(df["site"].to_numpy(), n_splits))
    if protocol == "leave-one-region-out":
        return list(region_folds(df["region"].to_numpy()))
    if protocol == "temporal(2022-23)":
        return list(temporal_fold(df["date"]))
    raise ValueError(protocol)


PROTOCOLS = ["random80/20", "leave-stations-out", "leave-one-region-out", "temporal(2022-23)"]


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def paper_row_mean_fill(X):
    """The paper's preprocessing bug: row.fillna(row.mean()) - a missing feature is replaced by the
    mean of the OTHER features of the same row (dB, elevation, day of year ... averaged together)."""
    X = np.array(X, float)
    rm = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1, keepdims=True)
    return np.where(np.isfinite(X), X, rm)


def make_tree_models(paper=True, seed=SEED):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    if paper:
        return {
            "RF": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=N_JOBS),
            "XGB": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=N_JOBS),
            "LGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed,
                                  verbose=-1, n_jobs=N_JOBS),
        }
    return {
        "RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                    random_state=seed, n_jobs=N_JOBS),
        "XGB": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                            colsample_bytree=0.7, min_child_weight=5, reg_lambda=5.0,
                            random_state=seed, n_jobs=N_JOBS),
        "LGBM": LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=-1,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              min_child_samples=20, reg_lambda=5.0, random_state=seed,
                              verbose=-1, n_jobs=N_JOBS),
    }


def fit_predict_trees(Xtr, ytr, Xte, paper=True, paper_fill=False, return_models=False):
    """RF/XGB/LGBM + mean ensemble. paper_fill=True reproduces row.fillna(row.mean());
    otherwise XGB/LGBM use NaN natively and RF gets training-set medians."""
    Xtr, Xte = np.array(Xtr, float), np.array(Xte, float)
    if paper_fill:
        Xtr, Xte = paper_row_mean_fill(Xtr), paper_row_mean_fill(Xte)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    preds, models = {}, {}
    for name, m in make_tree_models(paper).items():
        if name == "RF":
            m.fit(np.where(np.isfinite(Xtr), Xtr, med), ytr)
            preds[name] = m.predict(np.where(np.isfinite(Xte), Xte, med))
        else:
            m.fit(Xtr, ytr)
            preds[name] = m.predict(Xte)
        models[name] = m
    preds["Ensemble"] = np.mean([preds[k] for k in ("RF", "XGB", "LGBM")], axis=0)
    if return_models:
        return preds, (models, med)
    return preds


def predict_trees(models_med, X, paper_fill=False):
    models, med = models_med
    X = np.array(X, float)
    if paper_fill:
        X = paper_row_mean_fill(X)
    p = {}
    for name, m in models.items():
        p[name] = m.predict(np.where(np.isfinite(X), X, med) if name == "RF" else X)
    p["Ensemble"] = np.mean([p[k] for k in ("RF", "XGB", "LGBM")], axis=0)
    return p


# ----------------------------------------------------------------------------
# TabPFN v2 (pretrained tabular foundation model)
# ----------------------------------------------------------------------------
TABPFN_MAX_CONTEXT = 10_000   # TabPFN v2 was pretrained on <= 10k training samples


def context_subsample(groups, max_n=TABPFN_MAX_CONTEXT, seed=SEED, balance=None):
    """Indices (into groups) for the TabPFN context set, at most max_n rows. Sampling is stratified
    so that every station (or every region if balance is given) contributes equally (capped)."""
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups if balance is None else balance)
    if len(groups) <= max_n:
        return np.arange(len(groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    per = max_n // len(uniq)
    take = []
    leftovers = []
    for g in range(len(uniq)):
        idx = np.where(inv == g)[0]
        rng.shuffle(idx)
        take.append(idx[:per])
        leftovers.append(idx[per:])
    take = np.concatenate(take)
    rest = np.concatenate(leftovers)
    if len(take) < max_n and len(rest):
        take = np.concatenate([take, rng.choice(rest, min(max_n - len(take), len(rest)), replace=False)])
    return np.sort(take)


def tabpfn_fit_predict(Xtr, ytr, Xte, n_estimators=4, seed=SEED, batch=2000):
    """n_estimators=4 (library default 8) to fit the compute budget of a 6 GB laptop GPU."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=dev, n_estimators=n_estimators,
                                                   random_state=seed, ignore_pretraining_limits=False,
                                                   memory_saving_mode=True)
    m.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.float32))
    Xte = np.asarray(Xte, np.float32)
    out = [m.predict(Xte[i:i + batch]) for i in range(0, len(Xte), batch)]
    return np.concatenate(out) if out else np.array([])


# ----------------------------------------------------------------------------
def collect(rows, df, pred_cols, **tags):
    for c in pred_cols:
        rows.append(dict(**tags, model=c, **summarize(df, c)))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return f"{(time.time() - self.t0) / 60:.1f} min"


def log_progress(msg):
    line = f"- {time.strftime('%Y-%m-%d %H:%M')} {msg}"
    print(line, flush=True)
    with open(os.path.join(RES, "progress.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
