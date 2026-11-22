"""
Build the multi-region experiment datasets from public, real data.

Source: github.com/kashingtonDC/RZSM (SCAN hourly soil moisture downloaded from
the NRCS Report Generator, and Sentinel-1 / Landsat-8 / PRISM / NLCD values
extracted with Google Earth Engine at each SCAN site, 2015-01 .. 2019-01).

Outputs (data/processed/):
  scan_daily.parquet   daily 5 cm (2 in) soil moisture for every SCAN site with data
  anchor.parquet       one row per (site, Sentinel-1 date) with features + target
  sites.parquet        site metadata (lat/lon/elev/texture/landcover/region)
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")

# NOAA NCEI U.S. climate regions (Karl & Koss 1984)
CLIMATE_REGION = {
    **dict.fromkeys(["WA", "OR", "ID"], "Northwest"),
    **dict.fromkeys(["CA", "NV"], "West"),
    **dict.fromkeys(["UT", "CO", "AZ", "NM"], "Southwest"),
    **dict.fromkeys(["MT", "WY", "ND", "SD", "NE"], "N.Rockies&Plains"),
    **dict.fromkeys(["MN", "WI", "MI", "IA"], "UpperMidwest"),
    **dict.fromkeys(["MO", "IL", "IN", "OH", "KY", "TN", "WV"], "OhioValley"),
    **dict.fromkeys(["KS", "OK", "TX", "AR", "LA", "MS"], "South"),
    **dict.fromkeys(["AL", "GA", "FL", "SC", "NC", "VA"], "Southeast"),
    **dict.fromkeys(["ME", "NH", "VT", "MA", "RI", "CT", "NY", "PA", "NJ", "DE", "MD"], "Northeast"),
}

NLCD_GROUP = {  # RZSM lc_type codes -> coarse group
    0: "water", 2: "developed", 3: "developed", 4: "developed", 5: "developed",
    6: "barren", 7: "forest", 8: "forest", 9: "forest", 11: "shrub",
    12: "grassland", 16: "pasture", 17: "cropland", 18: "wetland", 19: "wetland",
}


# ----------------------------------------------------------------------------
# 1. SCAN hourly -> daily 5 cm soil moisture
# ----------------------------------------------------------------------------
def read_scan_hourly(path: str) -> pd.DataFrame:
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    header = lines[0].strip().split(",")
    rows = [l.strip().split(",") for l in lines[1:] if l.strip()]
    df = pd.DataFrame(rows, columns=header[: len(rows[0])] if rows else header)
    out = pd.DataFrame({"datetime": pd.to_datetime(df.iloc[:, 0], errors="coerce")})
    for depth in (2, 4, 8):
        col = [c for c in df.columns if f"-{depth}in" in c]
        out[f"sm_{depth}in"] = pd.to_numeric(df[col[0]], errors="coerce") if col else np.nan
    return out.dropna(subset=["datetime"])


def qc_hourly(s: pd.Series) -> pd.Series:
    """Basic QC on an hourly volumetric SM series (percent)."""
    s = s.where((s > 0.5) & (s <= 60))
    # remove isolated spikes: |x - rolling median(25h)| > 8 % vol
    med = s.rolling(25, center=True, min_periods=5).median()
    s = s.where((s - med).abs() <= 8)
    return s


def build_scan_daily() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "rzsm_data", "*.txt"))):
        site = os.path.basename(path)[:-4]
        if not site.isdigit():
            continue
        h = read_scan_hourly(path)
        if h.empty:
            continue
        h = h.set_index("datetime").sort_index()
        for c in ["sm_2in", "sm_4in", "sm_8in"]:
            h[c] = qc_hourly(h[c])
        d = h.resample("D").agg(["mean", "count"])
        daily = pd.DataFrame({
            "sm_5cm": d[("sm_2in", "mean")].where(d[("sm_2in", "count")] >= 12),
            "sm_10cm": d[("sm_4in", "mean")].where(d[("sm_4in", "count")] >= 12),
            "sm_20cm": d[("sm_8in", "mean")].where(d[("sm_8in", "count")] >= 12),
        })
        # flat-line detection: 10+ days with identical daily value -> sensor stuck / frozen
        roll_std = daily["sm_5cm"].rolling(10, min_periods=10).std()
        stuck = roll_std.fillna(1) < 1e-6
        stuck = stuck | stuck.shift(-9, fill_value=False)
        daily.loc[stuck, "sm_5cm"] = np.nan
        daily["site"] = int(site)
        frames.append(daily.reset_index(names="date"))
    daily = pd.concat(frames, ignore_index=True)
    return daily


# ----------------------------------------------------------------------------
# 2. Site metadata
# ----------------------------------------------------------------------------
def build_sites() -> pd.DataFrame:
    import geopandas as gpd

    s = gpd.read_file(os.path.join(RAW, "scan_sites.shp"))
    s = pd.DataFrame(s.drop(columns="geometry"))
    s["site"] = s["id"].astype(int)
    s["name"] = s["site_name"].str.replace(r"\s*\(\d+\)", "", regex=True)
    s["region"] = s["state"].map(CLIMATE_REGION)
    s = s[["site", "name", "state", "region", "lat", "lon", "elev",
           "clay_fract", "silt_fract", "sand_fract"]].copy()
    s["elev"] = pd.to_numeric(s["elev"], errors="coerce") * 0.3048  # ft -> m
    return s


# ----------------------------------------------------------------------------
# 3. Sentinel-1 anchor table
# ----------------------------------------------------------------------------
def _parse_pixels(x):
    if not isinstance(x, str):
        return np.nan
    vals = [float(v) for v in re.sub(r"[\[\]]", " ", x).split()]
    vals = [v for v in vals if v != 0.0 and v > -40]
    if not vals:
        return np.nan
    # average in linear power, return dB
    return 10 * np.log10(np.mean(10 ** (np.array(vals) / 10)))


def build_anchor(scan_daily: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_csv(os.path.join(RAW, "rzsm_data", "all_dat_f.csv"))
    a = a.rename(columns={"Unnamed: 0": "site", "Unnamed: 1": "acq_time"})
    a["acq_time"] = pd.to_datetime(a["acq_time"])
    a["date"] = a["acq_time"].dt.normalize()
    a["VV"] = a["vv"].apply(_parse_pixels)
    a["VH"] = a["hv"].apply(_parse_pixels)
    a = a.dropna(subset=["VV"])
    a = a.sort_values(["site", "acq_time"]).drop_duplicates(["site", "date"], keep="first")

    # LAI (MODIS) from the companion table, matched on site+date
    lai = pd.read_csv(os.path.join(RAW, "rzsm_data", "all_dat_desc.csv"))
    lai = lai.rename(columns={"Unnamed: 0": "site"})
    lai["date"] = pd.to_datetime(lai["date"]).dt.normalize()
    lai = lai.groupby(["site", "date"], as_index=False)["LAI"].mean()
    a = a.merge(lai, on=["site", "date"], how="left")

    # Optical (Landsat-8, temporally filtered) indices -> analogues of the S2 features
    b2, b4, b5, b6 = (a[f"B{i}_filt"] for i in (2, 4, 5, 6))
    a["NDVI"] = (b5 - b4) / (b5 + b4)
    a["NDMI"] = (b5 - b6) / (b5 + b6)
    a["EVI"] = 2.5 * (b5 - b4) / (b5 + 6 * b4 - 7.5 * b2 + 1)
    for c in ["NDVI", "NDMI", "EVI"]:
        a[c] = a[c].where(a[c].between(-1, 1))

    # SAR features computed correctly in LINEAR power (paper computed them on dB values)
    vv_lin, vh_lin = 10 ** (a["VV"] / 10), 10 ** (a["VH"] / 10)
    a["CR_dB"] = a["VH"] - a["VV"]                       # cross-pol ratio (dB) = 10log10(VH/VV)
    a["RVI"] = 4 * vh_lin / (vv_lin + vh_lin)            # 0..~1 for natural surfaces
    a["SPAN_dB"] = 10 * np.log10(vv_lin + vh_lin)
    a["acq_hour"] = a["acq_time"].dt.hour + a["acq_time"].dt.minute / 60  # orbit-track proxy

    a["precip2d"] = a["precip"].clip(upper=300)
    a["lc_type"] = a["lc_type"].fillna(-1).astype(int)
    a["lc_group"] = a["lc_type"].map(NLCD_GROUP).fillna("other")

    # ---- target and lags from the QC'd daily SCAN series ----
    sd = scan_daily[["site", "date", "sm_5cm"]].copy()
    a = a.merge(sd, on=["site", "date"], how="left")
    prev = sd.copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    a = a.merge(prev.rename(columns={"sm_5cm": "SM_lag1d"}), on=["site", "date"], how="left")

    a = a.merge(sites, on="site", how="left")
    a = a[a["region"].notna()]  # CONUS only

    # ---- temporal context per site (satellite-only, no in-situ needed) ----
    a = a.sort_values(["site", "date"]).reset_index(drop=True)
    g = a.groupby("site")
    a["days_since_prev_s1"] = g["date"].diff().dt.days
    a["dVV"] = g["VV"].diff()
    a["dVH"] = g["VH"].diff()
    # paper-style lag: SCAN value at the preceding S1 sample of the same site
    a["SM_lag_prevS1"] = g["sm_5cm"].shift(1)

    # change-detection normalisation using the full S1 history of the site
    q05 = g["VV"].transform(lambda s: s.quantile(0.05))
    q95 = g["VV"].transform(lambda s: s.quantile(0.95))
    a["VV_cd"] = (a["VV"] - q05) / (q95 - q05)
    a["VV_anom"] = a["VV"] - g["VV"].transform("median")
    a["VH_anom"] = a["VH"] - g["VH"].transform("median")
    a["VV_site_med"] = g["VV"].transform("median")
    a["VV_site_iqr"] = g["VV"].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    a["NDVI_anom"] = a["NDVI"] - g["NDVI"].transform("median")

    a["DOY"] = a["date"].dt.dayofyear
    a["Month"] = a["date"].dt.month
    a["doy_sin"] = np.sin(2 * np.pi * a["DOY"] / 365.25)
    a["doy_cos"] = np.cos(2 * np.pi * a["DOY"] / 365.25)

    keep = ["site", "name", "state", "region", "date", "acq_time", "lat", "lon", "elev",
            "clay_fract", "silt_fract", "sand_fract", "lc_type", "lc_group",
            "VV", "VH", "CR_dB", "RVI", "SPAN_dB", "acq_hour", "dVV", "dVH",
            "days_since_prev_s1", "VV_cd", "VV_anom", "VH_anom", "VV_site_med", "VV_site_iqr",
            "NDVI", "NDMI", "EVI", "NDVI_anom", "LAI", "precip2d",
            "DOY", "Month", "doy_sin", "doy_cos",
            "SM_lag1d", "SM_lag_prevS1", "sm_5cm"]
    return a[keep]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("SCAN hourly -> daily ...")
    daily = build_scan_daily()
    sites = build_sites()
    daily = daily[daily["site"].isin(sites.loc[sites.region.notna(), "site"])]
    daily.to_parquet(os.path.join(OUT, "scan_daily.parquet"), index=False)
    sites.to_parquet(os.path.join(OUT, "sites.parquet"), index=False)
    print("daily rows", len(daily), "sites with any 5cm data", daily.dropna(subset=["sm_5cm"]).site.nunique())

    anchor = build_anchor(daily, sites)
    anchor.to_parquet(os.path.join(OUT, "anchor.parquet"), index=False)
    lab = anchor.dropna(subset=["sm_5cm"])
    print("anchor rows", len(anchor), "labelled", len(lab), "sites", lab.site.nunique())
    print(lab.groupby("region").agg(sites=("site", "nunique"), n=("site", "size")))


if __name__ == "__main__":
    main()
