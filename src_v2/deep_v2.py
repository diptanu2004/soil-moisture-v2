"""GPU wrappers around src/deep.py (TempoSAR) with the v2 feature lists."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

import common_v2  # noqa: F401  (puts src/ on sys.path)
import deep as D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

# v2 feature lists (the module-level names in deep.py are what Prepared reads)
D.SEQ_FEATS = D.SEQ_FEATS + ["VV_norm40", "angle", "smap_l4_sm", "era5_swvl1", "API_090", "LST_day"]
D.CUR_FEATS = D.SEQ_FEATS + ["dVV", "dVH", "days_since_prev_s1", "VV_cd", "VV_anom", "VH_anom", "NDVI_anom", "EVI",
                             "SPAN_dB", "pass_asc", "rain_7d", "rain_30d", "API_085", "days_since_rain",
                             "era5_swvl2", "era5_t2m", "era5_stl1", "smap_l4_rz", "smap_l3e_sm", "MODIS_NDVI",
                             "MODIS_NDMI", "LST_night", "dLST", "S2_BSI", "S2_NDWI", "s2_age_days"]
D.STATIC_FEATS = D.STATIC_FEATS + ["TWI", "slope", "bdod", "soc", "hnd"]
D.LC_GROUPS = ["barren", "cropland", "developed", "forest", "grassland", "other", "pasture", "shrub", "water",
               "wetland"]

Prepared, Scaler, build_model = D.Prepared, D.Scaler, D.build_model


def to_dev(T):
    return {k: v.to(DEVICE) for k, v in T.items()}


def predict(model, T, scaler, bs=4096):
    model.eval()
    T = to_dev(T)
    out = []
    with torch.no_grad():
        for i in range(0, len(T["y"]), bs):
            sl = slice(i, i + bs)
            out.append(model(T["seq"][sl], T["mask"][sl], T["cur"][sl], T["stat"][sl], T["lc"][sl]).cpu())
    return torch.cat(out).numpy() * scaler.ys + scaler.ym


def train(model, T, Tval=None, epochs=60, lr=1e-3, wd=1e-4, bs=256, seed=0, patience=10, params=None,
          l2sp_ref=None, l2sp_alpha=0.0):
    """Same procedure as deep.train (AdamW + OneCycle, Huber, early stopping on validation stations)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = model.to(DEVICE)
    T = to_dev(T)
    Tval = to_dev(Tval) if Tval is not None else None
    if l2sp_ref is not None:
        l2sp_ref = {k: v.to(DEVICE) for k, v in l2sp_ref.items()}
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd)
    n = len(T["y"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, math.ceil(n / bs)))
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_state, bad = np.inf, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            p = model(T["seq"][b], T["mask"][b], T["cur"][b], T["stat"][b], T["lc"][b])
            loss = loss_fn(p, T["y"][b])
            if l2sp_ref is not None and l2sp_alpha > 0:
                loss = loss + l2sp_alpha * sum(((w - l2sp_ref[k]) ** 2).sum()
                                               for k, w in model.named_parameters() if k in l2sp_ref)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if Tval is not None:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Tval["seq"], Tval["mask"], Tval["cur"], Tval["stat"], Tval["lc"]), Tval["y"]).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if Tval is not None:
        model.load_state_dict(best_state)
    return model


def fine_tune(pre_model, T, epochs=40, lr=3e-4, seed=0, alpha=1e-2):
    """L2-SP fine-tuning (Li et al., 2018) on a handful of target samples."""
    m = copy.deepcopy(pre_model)
    ref = {k: p.detach().clone() for k, p in pre_model.named_parameters()}
    return train(m, T, None, epochs=epochs, lr=lr, seed=seed, l2sp_ref=ref, l2sp_alpha=alpha, bs=32)


def split_inner(idx, sites, frac=0.12, seed=0):
    s = np.unique(sites[idx])
    rng = np.random.RandomState(seed)
    vs = rng.choice(s, max(1, int(round(len(s) * frac))), replace=False)
    return idx[~np.isin(sites[idx], vs)], idx[np.isin(sites[idx], vs)]


D_KW = dict(d=48, dropout=0.2)
TRAIN_KW = dict(epochs=60, lr=1e-3, wd=1e-3)


def pretrain(P, tr, sites, seeds=(0, 1, 2)):
    inner, val = split_inner(tr, sites)
    sc = Scaler().fit(P.subset(inner))
    Tin, Tva = sc.tensors(P.subset(inner)), sc.tensors(P.subset(val))
    models = [train(build_model(Tin, **D_KW), Tin, Tva, seed=s, **TRAIN_KW) for s in seeds]
    return models, sc


def deep_fit_predict(P, tr, te, sites, seeds=(0, 1, 2)):
    models, sc = pretrain(P, tr, sites, seeds)
    Tte = sc.tensors(P.subset(te))
    return np.mean([predict(m, Tte, sc) for m in models], 0), models, sc
