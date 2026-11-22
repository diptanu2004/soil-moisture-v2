"""
Daily (gap-fill) models for the v2 tables produced by gee/assemble_v2.py.

Not runnable on the public 2015-2018 dataset used for the experiments in this
package (it has no daily drivers). Tested for code paths on a synthetic table.

Models
  * LGBM-daily     gradient boosting on the daily drivers (+ latest S1 state)
  * DailyLSTM      LSTM over the last W days of drivers -> sm_5cm today.
                   Transfer learning: pretrain on a reanalysis/satellite soil
                   moisture target that exists everywhere (ERA5-Land layer 1 or
                   SMAP L4 surface SM) at many sites, then fine-tune on in-situ.

Modes
  * satellite-only     no in-situ input (the product the project plan aims for)
  * +lag1d             previous-day in-situ value, defined the same way in
                       training and application (fixes the v1 mismatch)

usage:
  python daily_v2.py --daily ../data/processed_v2/daily.parquet --out ../results/v2
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import group_folds, summarize

TARGET = "sm_5cm"
DAILY_DRIVERS = [
    "precip", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "API_085", "API_090", "API_095",
    "days_since_rain", "era5_swvl1", "era5_swvl2", "era5_t2m", "era5_pev", "era5_evap", "era5_ssr",
    "era5_snow", "smap_l4_sm", "smap_l4_wetness", "smap_l3e_sm", "MODIS_NDVI", "MODIS_NDMI",
    "MODIS_EVI", "LAI", "LST_day", "LST_night", "dLST", "NDVI", "NDMI", "s2_age_days",
    "last_VV", "last_VV_cd", "last_VV_anom", "last_CR_dB", "s1_age_days", "doy_sin", "doy_cos",
]
STATIC = ["elev", "slope", "TWI", "sand_fract", "clay_fract", "silt_fract", "bdod", "soc", "lat", "lon"]


def available(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().any()]


# ----------------------------------------------------------------------------
# gradient boosting baseline
# ----------------------------------------------------------------------------
def lgbm_loso(df, feats, n_splits=10):
    from lightgbm import LGBMRegressor
    lab = df.dropna(subset=[TARGET]).reset_index(drop=True)
    pred = np.full(len(lab), np.nan)
    for tr, te in group_folds(lab["site"].to_numpy(), n_splits):
        m = LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, min_child_samples=40, reg_lambda=5, verbose=-1)
        m.fit(lab.iloc[tr][feats], lab.iloc[tr][TARGET])
        pred[te] = m.predict(lab.iloc[te][feats])
    return lab.assign(pred=pred)


# ----------------------------------------------------------------------------
# DailyLSTM with pretraining
# ----------------------------------------------------------------------------
class Windows:
    def __init__(self, df, drivers, static, W=60):
        df = df.sort_values(["site", "date"]).reset_index(drop=True)
        self.df = df
        X = df[drivers].to_numpy(np.float32)
        self.mu = np.nanmean(X, 0)
        self.sd = np.where(np.nanstd(X, 0) > 1e-6, np.nanstd(X, 0), 1)
        S = df[static].to_numpy(np.float32) if static else np.zeros((len(df), 0), np.float32)
        self.smu = np.nanmean(S, 0) if S.shape[1] else S[:0].mean(0)
        self.ssd = np.where(np.nanstd(S, 0) > 1e-6, np.nanstd(S, 0), 1) if S.shape[1] else self.smu
        self.X, self.S, self.W = X, S, W
        self.start = df.groupby("site").cumcount().to_numpy()

    def norm(self, mu=None, sd=None, smu=None, ssd=None):
        mu = self.mu if mu is None else mu
        sd = self.sd if sd is None else sd
        X = (self.X - mu) / sd
        miss = ~np.isfinite(X)
        self.Xn = np.concatenate([np.where(miss, 0, X), miss], 1).astype(np.float32)
        if self.S.shape[1]:
            smu = self.smu if smu is None else smu
            ssd = self.ssd if ssd is None else ssd
            S = (self.S - smu) / ssd
            self.Sn = np.where(np.isfinite(S), S, 0).astype(np.float32)
        else:
            self.Sn = self.S
        return self

    def batch(self, rows):
        W = self.W
        x = np.zeros((len(rows), W, self.Xn.shape[1]), np.float32)
        for i, r in enumerate(rows):
            k = min(W, self.start[r] + 1)
            x[i, W - k:] = self.Xn[r - k + 1:r + 1]
        return torch.tensor(x), torch.tensor(self.Sn[rows])


class DailyLSTM(nn.Module):
    def __init__(self, n_in, n_static, h=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, h, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.Linear(h + n_static, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, x, s):
        o, _ = self.lstm(x)
        return self.head(torch.cat([o[:, -1], s], -1)).squeeze(-1)


def fit_lstm(model, win, rows, y, epochs=10, lr=1e-3, bs=256, seed=0, ref=None, alpha=0.0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.HuberLoss()
    rows, y = np.asarray(rows), np.asarray(y, np.float32)
    ok = np.isfinite(y)
    rows, y = rows[ok], y[ok]
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(rows))
        for i in range(0, len(rows), bs):
            b = perm[i:i + bs]
            x, s = win.batch(rows[b])
            loss = lf(model(x, s), torch.tensor(y[b]))
            if ref is not None and alpha > 0:
                loss = loss + alpha * sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict_lstm(model, win, rows, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            x, s = win.batch(np.asarray(rows[i:i + bs]))
            out.append(model(x, s).numpy())
    return np.concatenate(out) if out else np.array([])


def lstm_loso(df, drivers, static, pretrain_target=None, W=60, n_splits=5, epochs=10, ft_epochs=5):
    """Leave-stations-out DailyLSTM. If pretrain_target (e.g. 'era5_swvl1') is given, the
    network is first trained on that target at ALL training-site days (dense labels), then
    fine-tuned on the in-situ target with an L2-SP penalty."""
    win = Windows(df, drivers, static, W).norm()
    d = win.df
    y = d[TARGET].to_numpy(np.float32)
    ym, ys = np.nanmean(y), np.nanstd(y)
    pred = np.full(len(d), np.nan)
    for tr, te in group_folds(d["site"].to_numpy(), n_splits):
        m = DailyLSTM(win.Xn.shape[1], win.Sn.shape[1])
        if pretrain_target:
            yp = d[pretrain_target].to_numpy(np.float32)
            m = fit_lstm(m, win, tr, (yp[tr] - ym) / ys, epochs=epochs)
            ref = {n: p.detach().clone() for n, p in m.named_parameters()}
            m = fit_lstm(m, win, tr, (y[tr] - ym) / ys, epochs=ft_epochs, lr=3e-4, ref=ref, alpha=1e-3)
        else:
            m = fit_lstm(m, win, tr, (y[tr] - ym) / ys, epochs=epochs)
        pred[te] = predict_lstm(m, win, te) * ys + ym
    return d.assign(pred=pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--out", default="../results/v2")
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(a.daily)
    df = df[df.groupby("site")[TARGET].transform(lambda s: s.notna().sum()) >= 30]
    drivers = available(df, DAILY_DRIVERS)
    static = available(df, STATIC)
    rows = []
    lab = df.dropna(subset=[TARGET, "SM_lag1d"])
    rows.append(dict(model="Persistence(1d)", **summarize(lab.assign(p=lab["SM_lag1d"]), "p")))
    for name, feats in [("LGBM-daily satOnly", drivers + static), ("LGBM-daily +lag1d", drivers + static + ["SM_lag1d"])]:
        r = lgbm_loso(df, feats)
        rows.append(dict(model=name, **summarize(r, "pred")))
    r = lstm_loso(df, drivers, static, epochs=a.epochs)
    rows.append(dict(model="DailyLSTM satOnly (scratch)", **summarize(r.dropna(subset=[TARGET]), "pred")))
    for pt in ("era5_swvl1", "smap_l4_sm"):
        if pt in df:
            r = lstm_loso(df, drivers, static, pretrain_target=pt, epochs=a.epochs)
            rows.append(dict(model=f"DailyLSTM satOnly (pretrained on {pt})", **summarize(r.dropna(subset=[TARGET]), "pred")))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "daily_v2_results.csv"), index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
