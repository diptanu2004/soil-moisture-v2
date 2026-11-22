"""
Deep models for station-scale soil moisture retrieval + transfer learning.

TempoSAR  : a Transformer that encodes the last L Sentinel-1 acquisitions of a
            site (VV/VH/cross-ratio, optical indices, LAI, rain, time gaps) and
            fuses them with static site descriptors (soil hydraulics, land
            cover, climatology) to predict 5 cm soil moisture at the current
            acquisition. Satellite-only: no in-situ value is used as input
            unless `use_lag=True`.

Transfer learning:
  pretrain(source) -> fine_tune(target pool) with
    * 'head'   : only the prediction head + static encoder are updated
    * 'l2sp'   : all weights updated, L2-SP penalty pulls them to the
                 pretrained solution (Li et al., ICML 2018)
"""
from __future__ import annotations

import copy
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import TARGET, load_anchor, load_full

torch.set_num_threads(2)
DEVICE = "cpu"

SEQ_FEATS = ["VV", "VH", "CR_dB", "RVI", "NDVI", "NDMI", "LAI", "precip2d", "doy_sin", "doy_cos",
             "acq_hour"]
CUR_FEATS = SEQ_FEATS + ["dVV", "dVH", "days_since_prev_s1", "VV_cd", "VV_anom", "VH_anom",
                         "NDVI_anom", "EVI", "SPAN_dB"]
STATIC_FEATS = ["elev", "clay_fract", "sand_fract", "theta_wp", "theta_fc", "theta_sat",
                "VV_site_med", "VV_site_iqr", "site_mean_precip2d", "site_mean_NDVI",
                "site_mean_NDMI", "site_mean_LAI", "site_mean_VH", "site_wet_frac",
                "VV_site_q05", "VV_site_q95"]
LC_GROUPS = ["barren", "cropland", "developed", "forest", "grassland", "other", "pasture", "shrub",
             "water", "wetland"]
CLIM_FEATS = ["site_mean_precip2d", "site_mean_NDVI", "site_mean_NDMI", "site_mean_LAI",
              "site_mean_VH", "site_wet_frac", "VV_site_q05", "VV_site_q95"]




# ----------------------------------------------------------------------------
# tensors
# ----------------------------------------------------------------------------
class Prepared:
    """Holds numpy arrays for a dataframe (labelled rows) with their S1 context."""

    def __init__(self, lab: pd.DataFrame, full: pd.DataFrame, L: int = 10, use_lag: bool = False):
        self.df = lab.reset_index(drop=True)
        self.L = L
        full = full.sort_values(["site", "date"]).reset_index(drop=True)
        key = {(s, d): i for i, (s, d) in enumerate(zip(full["site"], full["date"]))}
        site_first = full.groupby("site").cumcount().to_numpy()
        seq_raw = full[SEQ_FEATS].to_numpy(np.float32)
        dates = full["date"].to_numpy("datetime64[D]").astype(np.int64)

        n = len(self.df)
        self.seq = np.zeros((n, L, len(SEQ_FEATS) + 1), np.float32)   # + log gap
        self.seq_mask = np.ones((n, L), bool)                         # True = padding
        for r, (s, d) in enumerate(zip(self.df["site"], self.df["date"])):
            i = key[(s, d)]
            k = min(L, site_first[i])          # number of previous acquisitions available
            if k == 0:
                continue
            idx = np.arange(i - k, i)
            self.seq[r, L - k:, :-1] = seq_raw[idx]
            self.seq[r, L - k:, -1] = np.log1p(dates[i] - dates[idx])
            self.seq_mask[r, L - k:] = False
        self.cur = self.df[CUR_FEATS + (["SM_lag1d"] if use_lag else [])].to_numpy(np.float32)
        self.stat = self.df[STATIC_FEATS].to_numpy(np.float32)
        self.lc = self.df["lc_group"].map({g: i for i, g in enumerate(LC_GROUPS)}).fillna(5).to_numpy(np.int64)
        self.y = self.df[TARGET].to_numpy(np.float32)

    def subset(self, idx):
        new = copy.copy(self)
        new.df = self.df.iloc[idx].reset_index(drop=True)
        for a in ("seq", "seq_mask", "cur", "stat", "lc", "y"):
            setattr(new, a, getattr(self, a)[idx])
        return new


class Scaler:
    """Standardise with stats of the (pre)training set; NaN -> 0 + missing flag."""

    def fit(self, P: Prepared):
        def st(x):
            m = np.nanmean(x.reshape(-1, x.shape[-1]), 0)
            s = np.nanstd(x.reshape(-1, x.shape[-1]), 0)
            return np.nan_to_num(m), np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
        seq_valid = P.seq[~P.seq_mask]
        self.sm, self.ss = st(seq_valid if len(seq_valid) else P.seq)
        self.cm, self.cs = st(P.cur)
        self.tm, self.ts = st(P.stat)
        self.ym, self.ys = float(np.nanmean(P.y)), float(np.nanstd(P.y))
        return self

    def tensors(self, P: Prepared):
        def tr(x, m, s):
            z = (x - m) / s
            miss = ~np.isfinite(z)
            z = np.where(miss, 0, z)
            return np.concatenate([z, miss.astype(np.float32)], -1).astype(np.float32)
        seq = tr(P.seq, self.sm, self.ss)
        seq[P.seq_mask] = 0
        return dict(
            seq=torch.tensor(seq), mask=torch.tensor(P.seq_mask),
            cur=torch.tensor(tr(P.cur, self.cm, self.cs)),
            stat=torch.tensor(tr(P.stat, self.tm, self.ts)),
            lc=torch.tensor(P.lc),
            y=torch.tensor((P.y - self.ym) / self.ys),
        )


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class TempoSAR(nn.Module):
    def __init__(self, n_seq, n_cur, n_stat, d=64, heads=4, layers=2, dropout=0.15, use_seq=True):
        super().__init__()
        self.use_seq = use_seq
        self.seq_in = nn.Linear(n_seq, d)
        self.pos = nn.Parameter(torch.zeros(1, 32, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.cur_in = nn.Sequential(nn.Linear(n_cur, d), nn.GELU(), nn.LayerNorm(d))
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.lc_emb = nn.Embedding(len(LC_GROUPS), 8)
        self.static = nn.Sequential(nn.Linear(n_stat + 8, d), nn.GELU(), nn.LayerNorm(d))
        self.head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, seq, mask, cur, stat, lc):
        q = self.cur_in(cur)
        if self.use_seq:
            h = self.seq_in(seq) + self.pos[:, : seq.shape[1]]
            all_pad = mask.all(1)
            m = mask.clone()
            m[all_pad, -1] = False                      # avoid NaN for sites' first acquisition
            h = self.encoder(h, src_key_padding_mask=m)
            ctx, _ = self.attn(q.unsqueeze(1), h, h, key_padding_mask=m)
            ctx = ctx.squeeze(1) * (~all_pad).float().unsqueeze(1)
        else:
            ctx = torch.zeros_like(q)
        s = self.static(torch.cat([stat, self.lc_emb(lc)], -1))
        return self.head(torch.cat([q, ctx, s], -1)).squeeze(-1)


def build_model(T, use_seq=True, **kw):
    return TempoSAR(T["seq"].shape[-1], T["cur"].shape[-1], T["stat"].shape[-1], use_seq=use_seq, **kw)


def _batches(n, bs, shuffle, gen):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def predict(model, T, scaler, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(T["y"]), bs, False, None):
            out.append(model(T["seq"][b], T["mask"][b], T["cur"][b], T["stat"][b], T["lc"][b]))
    return torch.cat(out).numpy() * scaler.ys + scaler.ym


def train(model, T, Tval=None, epochs=60, lr=1e-3, wd=1e-4, bs=256, seed=0, patience=10,
          params=None, l2sp_ref=None, l2sp_alpha=0.0, verbose=False):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, math.ceil(len(T["y"]) / bs)))
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_state, bad = np.inf, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        for b in _batches(len(T["y"]), bs, True, gen):
            opt.zero_grad()
            p = model(T["seq"][b], T["mask"][b], T["cur"][b], T["stat"][b], T["lc"][b])
            loss = loss_fn(p, T["y"][b])
            if l2sp_ref is not None and l2sp_alpha > 0:
                reg = sum(((w - l2sp_ref[n]) ** 2).sum() for n, w in model.named_parameters() if n in l2sp_ref)
                loss = loss + l2sp_alpha * reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if Tval is not None:
            model.eval()
            with torch.no_grad():
                pv = model(Tval["seq"], Tval["mask"], Tval["cur"], Tval["stat"], Tval["lc"])
                vl = loss_fn(pv, Tval["y"]).item()
            if verbose:
                print(ep, round(vl, 4))
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if Tval is not None:
        model.load_state_dict(best_state)
    return model


def fine_tune(pre_model, T, mode="l2sp", epochs=40, lr=3e-4, seed=0, alpha=1e-3, Tval=None):
    m = copy.deepcopy(pre_model)
    if mode == "head":
        for n, p in m.named_parameters():
            p.requires_grad = n.startswith(("head", "static", "lc_emb"))
        params = [p for p in m.parameters() if p.requires_grad]
        return train(m, T, Tval, epochs=epochs, lr=lr, seed=seed, params=params, patience=8)
    ref = {n: p.detach().clone() for n, p in pre_model.named_parameters()}
    return train(m, T, Tval, epochs=epochs, lr=lr, seed=seed, l2sp_ref=ref, l2sp_alpha=alpha, patience=8)


# ----------------------------------------------------------------------------
# data loading helper
# ----------------------------------------------------------------------------
def load_for_deep(min_samples=20):
    lab = load_anchor(min_samples)
    full = load_full()
    full = full[full["site"].isin(lab["site"].unique())]
    return lab.reset_index(drop=True), full.reset_index(drop=True)
