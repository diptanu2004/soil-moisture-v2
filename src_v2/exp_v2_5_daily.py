"""
Experiment v2-5: daily reconstruction at unseen stations (5-fold leave-stations-out, every
labelled station-day 2019-2023).

  Persistence(1d)               yesterday's in-situ value (needs in-situ, upper reference)
  LGBM-daily satOnly / +lag1d   gradient boosting on the daily drivers (src/daily_v2.py config)
  DailyLSTM satOnly (scratch)   LSTM over the last 60 days of drivers (src/daily_v2.py architecture)
  DailyLSTM satOnly pretrained  first trained on a dense, everywhere-available target (ERA5-Land
                                layer-1 soil water or SMAP L4 surface soil moisture) at the training
                                stations, then fine-tuned on in-situ with L2-SP
  Anchor interpolation          satellite-only anchor estimates (v2-3 best model) held / linearly
                                interpolated between Sentinel-1 dates

Writes results_v2/tables/v2_5_daily.csv and v2_5_daily_preds.parquet
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common_v2 import ROOT, TAB, TARGET, Timer, group_folds, load_daily, log_progress, summarize

sys.path.insert(0, os.path.join(ROOT, "src"))
from daily_v2 import DAILY_DRIVERS, STATIC, DailyLSTM, available  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W = 60
N_FOLDS = 5


class GPUWindows:
    """All driver rows on the GPU; a batch of windows is one gather (rows r-W+1..r of the same site)."""

    def __init__(self, df, drivers, static, train_rows):
        X = df[drivers].to_numpy(np.float32)
        mu, sd = np.nanmean(X[train_rows], 0), np.nanstd(X[train_rows], 0)
        sd = np.where(sd > 1e-6, sd, 1)
        Z = (X - mu) / sd
        miss = ~np.isfinite(Z)
        self.X = torch.tensor(np.concatenate([np.where(miss, 0, Z), miss], 1), dtype=torch.float32, device=DEVICE)
        S = df[static].to_numpy(np.float32)
        smu, ssd = np.nanmean(S[train_rows], 0), np.nanstd(S[train_rows], 0)
        S = (S - smu) / np.where(ssd > 1e-6, ssd, 1)
        self.S = torch.tensor(np.where(np.isfinite(S), S, 0), dtype=torch.float32, device=DEVICE)
        first = df.groupby("site").cumcount().to_numpy()
        self.start = torch.tensor(np.arange(len(df)) - first, device=DEVICE)
        self.off = torch.arange(-W + 1, 1, device=DEVICE)

    def batch(self, rows):
        rows = torch.as_tensor(rows, device=DEVICE)
        idx = rows[:, None] + self.off[None, :]
        valid = idx >= self.start[rows][:, None]
        x = self.X[idx.clamp(min=0)] * valid[..., None]
        return x, self.S[rows]


def fit(model, win, rows, y, epochs, lr=1e-3, bs=512, seed=0, ref=None, alpha=0.0):
    torch.manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.HuberLoss()
    rows, y = np.asarray(rows), np.asarray(y, np.float32)
    ok = np.isfinite(y)
    rows, yt = rows[ok], torch.tensor(y[ok], device=DEVICE)
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(rows))
        for i in range(0, len(rows), bs):
            b = perm[i:i + bs]
            x, s = win.batch(rows[b])
            loss = lf(model(x, s), yt[b])
            if ref is not None and alpha > 0:
                loss = loss + alpha * sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict(model, win, rows, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            x, s = win.batch(rows[i:i + bs])
            out.append(model(x, s).cpu().numpy())
    return np.concatenate(out)


def lgbm(Xtr, ytr, Xte):
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, min_child_samples=40, reg_lambda=5, verbose=-1, n_jobs=8)
    return m.fit(Xtr, ytr).predict(Xte)


def anchor_interp(d, ap):
    """hold-last / linear interpolation of satellite-only anchor estimates onto every day."""
    hold, lin = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for s, g in d.groupby("site"):
        a = ap[ap.site == s].set_index("date")["pred"].sort_index()
        a = a[~a.index.duplicated()]
        if len(a) < 5:
            continue
        idx = pd.DatetimeIndex(g["date"])
        hold[g.index] = a.reindex(idx, method="ffill").to_numpy()
        lin[g.index] = a.reindex(idx.union(a.index)).interpolate("time", limit_area="inside").reindex(idx).to_numpy()
    return hold, lin


def main(epochs=10, ft_epochs=5, interp_only=False):
    cols = ["LGBM-daily satOnly", "LGBM-daily +lag1d", "DailyLSTM satOnly (scratch)",
            "DailyLSTM satOnly (pretrained ERA5-Land)", "DailyLSTM satOnly (pretrained SMAP L4)"]
    if interp_only:                                   # reuse trained predictions, add anchor interpolation
        R = pd.read_parquet(f"{TAB}/v2_5_daily_preds.parquet")
        R = R[["site", "date", "region", TARGET, "SM_lag1d", "has_s1"] + cols + ["Persistence(1d)"]]
        return finish(R, cols)
    d = load_daily()
    d = d[d.groupby("site")[TARGET].transform(lambda s: s.notna().sum()) >= 30]
    d = d.sort_values(["site", "date"]).reset_index(drop=True)
    drivers, static = available(d, DAILY_DRIVERS), available(d, STATIC)
    sites = d["site"].to_numpy()
    y = d[TARGET].to_numpy(np.float32)
    lab = np.where(np.isfinite(y))[0]
    log_progress(f"v2-5 daily started: {d.site.nunique()} stations, {len(lab)} labelled days, "
                 f"{len(drivers)} drivers, device {DEVICE}")
    t = Timer()
    pred = {c: np.full(len(d), np.nan) for c in cols}
    Xs = d[drivers + static].to_numpy(np.float32)
    Xl = d[drivers + static + ["SM_lag1d"]].to_numpy(np.float32)
    for f, (trs, tes) in enumerate(group_folds(sites[lab], N_FOLDS)):
        tr, te = lab[trs], lab[tes]
        pred["LGBM-daily satOnly"][te] = lgbm(Xs[tr], y[tr], Xs[te])
        pred["LGBM-daily +lag1d"][te] = lgbm(Xl[tr], y[tr], Xl[te])
        train_sites = np.unique(sites[tr])
        all_tr = np.where(np.isin(sites, train_sites))[0]          # every day at training stations
        win = GPUWindows(d, drivers, static, all_tr)
        ym, ys = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))
        m = fit(DailyLSTM(win.X.shape[1], win.S.shape[1]), win, tr, (y[tr] - ym) / ys, epochs)
        pred["DailyLSTM satOnly (scratch)"][te] = predict(m, win, te) * ys + ym
        for name, pt in (("ERA5-Land", "era5_swvl1"), ("SMAP L4", "smap_l4_sm")):
            yp = d[pt].to_numpy(np.float32)
            m = fit(DailyLSTM(win.X.shape[1], win.S.shape[1]), win, all_tr, (yp[all_tr] - ym) / ys, epochs)
            ref = {n: p.detach().clone() for n, p in m.named_parameters()}
            m = fit(m, win, tr, (y[tr] - ym) / ys, ft_epochs, lr=3e-4, ref=ref, alpha=1e-3)
            pred[f"DailyLSTM satOnly (pretrained {name})"][te] = predict(m, win, te) * ys + ym
        print("fold", f, t(), flush=True)
        log_progress(f"v2-5 fold {f} done ({t()})")
    R = d.loc[lab, ["site", "date", "region", TARGET, "SM_lag1d", "has_s1"]].copy()
    for c in cols:
        R[c] = pred[c][lab]
    R["Persistence(1d)"] = R["SM_lag1d"]
    return finish(R, cols)


def finish(R, cols):
    cols = list(cols)
    p = f"{TAB}/v2_3_oof_leave-stations-out.parquet"
    if os.path.exists(p):
        oof = pd.read_parquet(p)
        best = "Hybrid(TabPFN+TempoSAR+Trees)" if "Hybrid(TabPFN+TempoSAR+Trees)" in oof else "Hybrid(TabPFN+Trees)"
        ap = oof[["site", "date"]].assign(pred=oof[best])
        hold, lin = anchor_interp(R.reset_index(drop=True), ap)
        R["Anchor hold-last (" + best + ")"] = hold
        R["Anchor linear interp (" + best + ")"] = lin
        cols += ["Anchor hold-last (" + best + ")", "Anchor linear interp (" + best + ")"]
    R.to_parquet(f"{TAB}/v2_5_daily_preds.parquet")
    rows = []
    common = R.dropna(subset=["Persistence(1d)"] + cols)
    for c in cols + ["Persistence(1d)"]:
        rows.append(dict(eval_set=f"common days (n={len(common)})", model=c, **summarize(common, c)))
    for c in cols:
        rows.append(dict(eval_set="all labelled days with a prediction", model=c, **summarize(R.dropna(subset=[c]), c)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{TAB}/v2_5_daily.csv", index=False)
    with pd.option_context("display.width", 220):
        print(res[["eval_set", "model", "n", "R2", "RMSE", "ubRMSE", "bias", "site_median_r"]].round(3).to_string())
    log_progress("v2-5 finished")


if __name__ == "__main__":
    main(interp_only="--interp-only" in sys.argv)
