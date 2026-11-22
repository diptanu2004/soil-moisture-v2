"""
Experiment v2-6: where does the satellite-only skill come from? (same rows as v2-2)

  raw products      SMAP L4 / SMAP L3E / ERA5-Land layer-1 soil moisture used directly as the
                    estimate (no training), plus a per-fold linear calibration to in-situ
  ablations         regularised trees on the improved v2 set minus one feature group
                    (no Sentinel-1 / no SMAP+ERA5 / no optical) and Sentinel-1 + static only

Protocols: leave-stations-out, leave-one-region-out.  Writes results_v2/tables/v2_6_ablation.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_v2 import (IMPROVED_V2, TAB, TARGET, Timer, collect, fit_predict_trees, folds_for, log_progress,
                       summarize)
from exp_v2_2_protocols import load_eval_rows

S1 = ["VV", "VH", "CR_dB", "RVI", "SPAN_dB", "has_VH", "acq_hour", "dVV", "dVH", "days_since_prev_s1", "VV_cd",
      "VV_anom", "VH_anom", "VV_site_med", "VV_site_iqr", "site_mean_VH", "VV_site_q05", "VV_site_q95", "angle",
      "VV_norm40", "pass_asc", "relorb_n_obs"]
REANALYSIS = ["era5_swvl1", "era5_swvl2", "era5_t2m", "era5_stl1", "era5_pev", "era5_evap", "era5_ssr", "era5_snow",
              "smap_l4_sm", "smap_l4_rz", "smap_l4_wetness", "smap_l3e_sm"]
OPTICAL = ["NDVI", "NDMI", "EVI", "NDVI_anom", "LAI", "MODIS_NDVI", "MODIS_NDMI", "MODIS_EVI", "S2_BSI", "S2_NDWI",
           "s2_age_days", "LST_day", "LST_night", "dLST", "site_mean_NDVI", "site_mean_NDMI", "site_mean_LAI"]
STATIC = ["elev", "clay_fract", "sand_fract", "theta_wp", "theta_fc", "theta_sat", "lc_code", "TWI", "slope", "bdod",
          "soc", "hnd", "doy_sin", "doy_cos"]
SETS = [
    ("all improved v2 features", IMPROVED_V2),
    ("without Sentinel-1", [f for f in IMPROVED_V2 if f not in S1]),
    ("without SMAP + ERA5-Land", [f for f in IMPROVED_V2 if f not in REANALYSIS]),
    ("without optical / LST", [f for f in IMPROVED_V2 if f not in OPTICAL]),
    ("Sentinel-1 + static only", S1 + STATIC),
    ("SMAP + ERA5-Land + static only", REANALYSIS + STATIC),
]
RAW = [("SMAP L4 surface SM (raw)", "smap_l4_sm"), ("SMAP L3E SM (raw)", "smap_l3e_sm"),
       ("ERA5-Land layer-1 SM (raw)", "era5_swvl1")]


def main():
    df = load_eval_rows()
    log_progress("v2-6 ablation started")
    t = Timer()
    rows = []
    for proto in ("leave-stations-out", "leave-one-region-out"):
        folds = folds_for(proto, df)
        te_all = np.concatenate([te for _, te in folds])
        ev = df.loc[te_all, ["site", "date", "region", TARGET] + [c for _, c in RAW]].copy()
        for name, col in RAW:
            ev[name] = ev[col]
            cal = np.full(len(df), np.nan)
            for tr, te in folds:       # linear calibration fitted on the training stations only
                ok = df.loc[tr, [col, TARGET]].dropna()
                a, b = np.polyfit(ok[col], ok[TARGET], 1)
                cal[te] = a * df.loc[te, col].to_numpy() + b
            ev[name + " + linear calibration"] = cal[te_all]
            for c in (name, name + " + linear calibration"):
                rows.append(dict(protocol=proto, model=c, **summarize(ev.dropna(subset=[c]), c)))
        for name, feats in SETS:
            pred = np.full(len(df), np.nan)
            X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
            for tr, te in folds:
                pred[te] = fit_predict_trees(X[tr], y[tr], X[te], paper=False)["Ensemble"]
            ev[name] = pred[te_all]
            collect(rows, ev, [name], protocol=proto)
            print(proto, name, t(), flush=True)
        pd.DataFrame(rows).to_csv(f"{TAB}/v2_6_ablation.csv", index=False)
        log_progress(f"v2-6 {proto} done ({t()})")
    out = pd.DataFrame(rows)
    print(out[["protocol", "model", "n", "R2", "RMSE", "ubRMSE", "bias", "r", "site_median_r"]].round(3).to_string())
    log_progress("v2-6 finished")


if __name__ == "__main__":
    main()
