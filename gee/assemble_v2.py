"""
Assemble the cached Earth Engine pulls (gee_extract_v2.py --cache) + SCAN hourly in-situ data
(insitu_v2.py scan) into the model tables:

  insitu_daily.parquet  sid, date, sm_5cm, stemp_5cm_C (daily mean of QC'd hourly values)
  daily.parquet         one row per station-day: daily drivers, latest S1 state (with age),
                        interpolated optical indices, static descriptors, sm_5cm, lags
  anchor.parquet        the station-days with a Sentinel-1 acquisition (S1 features of that day)
  paper_anchor.parquet  the paper's 16 anchor features, built the way the paper builds them
                        (single-pixel dB SAR, SAR indices on dB numbers, unmasked Sentinel-2,
                        MODIS MOD13Q1 NDVI/EVI, lag = in-situ at the previous S1 date), for every
                        station incl. the 16 paper stations at the paper's own coordinates
                        (sid "<scan_id>_pc"); plus the columns needed for the fix ladder

Column names of daily/anchor match src/common.py (VV, VH, CR_dB, RVI, ..., sm_5cm, SM_lag1d,
site, region). The extra v2 drivers are listed in V2_EXTRA_FEATS.

usage:
    python assemble_v2.py --cache ../data/raw_v2/ee_cache --scan ../data/raw_v2/scan_hourly \
                          --stations ../data/raw_v2/stations_v2.csv --out ../data/processed_v2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insitu_v2 import hourly_to_daily  # noqa: E402

V2_EXTRA_FEATS = [
    "angle", "VV_norm40", "pass_asc", "relorb_n_obs",
    "smap_l4_sm", "smap_l4_rz", "smap_l4_wetness", "smap_l3e_sm", "era5_swvl1", "era5_swvl2", "era5_t2m",
    "era5_stl1", "era5_pev", "era5_evap", "era5_ssr", "era5_snow",
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "API_085", "API_090", "API_095",
    "days_since_rain", "LST_day", "LST_night", "dLST", "MODIS_NDVI", "MODIS_NDMI", "MODIS_EVI",
    "S2_BSI", "S2_NDWI", "s2_age_days", "TWI", "slope", "bdod", "soc", "hnd",
]

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
WORLDCOVER_GROUP = {10: "forest", 20: "shrub", 30: "grassland", 40: "cropland", 50: "developed",
                    60: "barren", 70: "other", 80: "water", 90: "wetland", 95: "wetland", 100: "other"}

PAPER_S2_MAX_CLOUD = 20      # scene-level cloud filter only (no pixel mask), as in the paper
PAPER_S2_WINDOW_DAYS = 5     # nearest Sentinel-2 scene within +-5 days of the S1 date
PAPER_MODIS_WINDOW_DAYS = 16  # latest MOD13Q1 16-day composite


# ----------------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------------
def read_product(cache: str, product: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(cache, product, "*.parquet")))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in ("geo", "id") if c in df])
    df["sid"] = df["sid"].astype(str)
    # a station with no valid value of a band yields an all-null object column -> numeric
    for c in df.columns:
        if c not in ("sid", "date", "datetime", "time", "pass", "platform", "tile") and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "datetime" in df:
        df["date"] = df["datetime"].dt.normalize()
    elif "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    if "date" in df:
        df["date"] = df["date"].astype("datetime64[ns]")     # pandas 3 keeps ms/us units otherwise
    return df


def _daily_mean(df, cols):
    return df.groupby(["sid", "date"], as_index=False)[cols].mean()


def _interp_per_site(daily, obs, cols, max_gap_days, age_col=None):
    """Linear interpolation of sparse observations onto the daily grid (per site),
    limited to gaps <= max_gap_days; optional 'age' (days to nearest observation)."""
    out = []
    for sid, g in daily.groupby("sid"):
        o = obs[obs.sid == sid].set_index("date")[cols].sort_index()
        o = o[~o.index.duplicated()].dropna(how="all")
        idx = pd.DatetimeIndex(g["date"])
        if o.empty:
            res = pd.DataFrame(np.nan, index=idx, columns=cols)
            age = pd.Series(np.nan, index=idx)
        else:
            u = o.reindex(o.index.union(idx))
            res = u.interpolate("time", limit=max_gap_days, limit_area="inside").reindex(idx)
            pos = np.searchsorted(o.index.values, idx.values)
            prev = o.index.values[np.clip(pos - 1, 0, len(o) - 1)]
            nxt = o.index.values[np.clip(pos, 0, len(o) - 1)]
            age = np.minimum(np.abs((idx.values - prev).astype("timedelta64[D]").astype(float)),
                             np.abs((nxt - idx.values).astype("timedelta64[D]").astype(float)))
            age = pd.Series(age, index=idx)
        res["sid"] = sid
        res["date"] = idx
        if age_col:
            res[age_col] = np.asarray(age)
        out.append(res.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# product tables (QC + units)
# ----------------------------------------------------------------------------
def prep_products(cache: str) -> dict:
    P = {}
    ch = read_product(cache, "chirps")
    P["chirps"] = _daily_mean(ch.rename(columns={"precipitation": "precip"}), ["precip"])

    e5 = read_product(cache, "era5l").rename(columns={
        "volumetric_soil_water_layer_1": "era5_swvl1", "volumetric_soil_water_layer_2": "era5_swvl2",
        "temperature_2m": "era5_t2m", "temperature_2m_max": "era5_tmax", "skin_temperature": "era5_skt",
        "soil_temperature_level_1": "era5_stl1", "total_precipitation_sum": "era5_tp",
        "potential_evaporation_sum": "era5_pev", "total_evaporation_sum": "era5_evap",
        "surface_net_solar_radiation_sum": "era5_ssr", "snow_cover": "era5_snow"})
    for c in ("era5_swvl1", "era5_swvl2"):
        e5[c] *= 100                                  # m3/m3 -> % vol
    for c in ("era5_t2m", "era5_tmax", "era5_skt", "era5_stl1"):
        e5[c] -= 273.15
    for c in ("era5_tp", "era5_pev", "era5_evap"):
        e5[c] *= 1000                                 # m -> mm (evaporation negative = upward)
    e5["era5_ssr"] /= 1e6                             # J m-2 -> MJ m-2
    P["era5l"] = _daily_mean(e5, [c for c in e5.columns if c.startswith("era5_")])

    l4 = read_product(cache, "smapl4").rename(columns={"sm_surface": "smap_l4_sm", "sm_rootzone": "smap_l4_rz",
                                                       "sm_surface_wetness": "smap_l4_wetness"})
    l4["smap_l4_sm"] *= 100
    l4["smap_l4_rz"] *= 100
    P["smapl4"] = _daily_mean(l4, ["smap_l4_sm", "smap_l4_rz", "smap_l4_wetness"])

    # SMAP L3E: only ~20 % of retrievals at these sites carry the "recommended quality" bit, so all
    # retrievals are used as a predictor (it is an input, not a target)
    l3 = read_product(cache, "smapl3e")
    l3["smap_l3e_sm"] = 100 * l3[["soil_moisture_am", "soil_moisture_pm"]].mean(axis=1)
    P["smapl3e"] = _daily_mean(l3, ["smap_l3e_sm"])

    nb = read_product(cache, "nbar")
    q = lambda b: nb[f"BRDF_Albedo_Band_Mandatory_Quality_Band{b}"] <= 1   # 0 full / 1 magnitude inversion
    R = (nb["Nadir_Reflectance_Band1"] * 1e-4).where(q(1))
    N = (nb["Nadir_Reflectance_Band2"] * 1e-4).where(q(2))
    B = (nb["Nadir_Reflectance_Band3"] * 1e-4).where(q(1) & q(2))
    S = (nb["Nadir_Reflectance_Band6"] * 1e-4).where(q(6))
    nb["MODIS_NDVI"] = (N - R) / (N + R)
    nb["MODIS_NDMI"] = (N - S) / (N + S)
    nb["MODIS_EVI"] = (2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)).where(lambda x: x.between(-1, 1))
    P["nbar"] = _daily_mean(nb, ["MODIS_NDVI", "MODIS_NDMI", "MODIS_EVI"])

    la = read_product(cache, "lai")
    la["LAI"] = (la["Lai"] * 0.1).where((la["Lai"] <= 100) & ((la["FparLai_QC"].fillna(1).astype(int) & 1) == 0))
    P["lai"] = _daily_mean(la, ["LAI"])

    lst = []
    for prod in ("lstT", "lstA"):
        d = read_product(cache, prod)
        if d is None:
            continue
        # bits 0-1 == 00 (good). getRegion returns QC_Night = 0 as null (masked), so a null QC on a
        # pixel with an LST value means "good quality"
        good = lambda qc: (d[qc].fillna(0).astype(int) & 3) == 0
        d["LST_day"] = (d["LST_Day_1km"] * 0.02 - 273.15).where(good("QC_Day"))
        d["LST_night"] = (d["LST_Night_1km"] * 0.02 - 273.15).where(good("QC_Night"))
        lst.append(d[["sid", "date", "LST_day", "LST_night"]])
    lst = pd.concat(lst).groupby(["sid", "date"], as_index=False).mean()
    lst["dLST"] = lst["LST_day"] - lst["LST_night"]
    P["lst"] = lst

    s2 = read_product(cache, "s2")
    P["s2raw"] = s2
    m = s2[s2["S2_valid"] >= 0.5].rename(columns={"S2_NDVI": "NDVI", "S2_NDMI": "NDMI", "S2_EVI": "EVI"})
    P["s2masked"] = _daily_mean(m, ["NDVI", "NDMI", "EVI", "S2_BSI", "S2_NDWI"])

    P["mod13"] = read_product(cache, "mod13")
    P["s1raw"] = read_product(cache, "s1")

    st = read_product(cache, "static").rename(columns={"elevation": "elev"})
    for c in ("sand", "clay", "silt"):
        st[c] = st[c] / 10                            # g/kg -> %
    st["bdod"] = st["bdod"] / 100                     # cg/cm3 -> g/cm3
    st["soc"] = st["soc"] / 10                        # dg/kg -> g/kg
    st["lc_group"] = st["worldcover"].round().map(WORLDCOVER_GROUP).fillna("other")
    P["static"] = st.rename(columns={"sand": "sand_fract", "clay": "clay_fract", "silt": "silt_fract"})[
        ["sid", "elev", "slope", "aspect", "TWI", "hnd", "sand_fract", "clay_fract", "silt_fract", "bdod",
         "soc", "worldcover", "lc_group"]]
    return P


def insitu_daily(scan_dir: str) -> pd.DataFrame:
    out = []
    files = sorted(glob.glob(os.path.join(scan_dir, "*.parquet")))
    by_sid = {}
    for f in files:
        by_sid.setdefault(os.path.basename(f).split("_")[0], []).append(f)
    for sid, fs in by_sid.items():
        h = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        if h.empty or "sms_2in" not in h or h["sms_2in"].notna().sum() == 0:
            continue
        d = hourly_to_daily(h)
        d["date"] = d["date"].astype("datetime64[ns]")
        d["sid"] = sid
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# Sentinel-1
# ----------------------------------------------------------------------------
def s1_features(s1: pd.DataFrame) -> pd.DataFrame:
    s1 = s1.rename(columns={"VV_mean": "VV_lin", "VH_mean": "VH_lin", "angle_mean": "angle",
                            "VV_count": "n_pix"})
    s1 = s1.dropna(subset=["VV_lin"])
    s1 = s1[s1["VV_lin"] > 0]
    # one acquisition per station-day (keep the one with most valid pixels)
    s1 = s1.sort_values(["sid", "date", "n_pix"], ascending=[True, True, False]).drop_duplicates(["sid", "date"])
    s1["VV"] = 10 * np.log10(s1["VV_lin"])
    s1["VH"] = 10 * np.log10(s1["VH_lin"].where(s1["VH_lin"] > 0))
    s1["CR_dB"] = s1["VH"] - s1["VV"]
    s1["RVI"] = 4 * s1["VH_lin"] / (s1["VV_lin"] + s1["VH_lin"])
    s1["SPAN_dB"] = 10 * np.log10(s1["VV_lin"] + s1["VH_lin"])
    hh = s1["time"].astype(str).str.slice(0, 2).astype(float)
    mm = s1["time"].astype(str).str.slice(3, 5).astype(float)
    s1["acq_hour"] = hh + mm / 60
    s1["pass_asc"] = (s1["pass"] == "ASCENDING").astype(int)
    s1["has_VH"] = s1["VH"].notna().astype(int)
    s1 = s1.sort_values(["sid", "date"]).reset_index(drop=True)

    # incidence-angle normalisation to 40 deg with a per-station slope
    def _beta(g):
        g = g.dropna(subset=["angle", "VV"])
        if len(g) < 10 or g["angle"].std() < 2:
            return 0.0
        return float(np.polyfit(g["angle"], g["VV"], 1)[0])
    beta = s1.groupby("sid")[["angle", "VV"]].apply(_beta).rename("beta")
    s1 = s1.merge(beta, left_on="sid", right_index=True)
    s1["VV_norm40"] = s1["VV"] - s1["beta"] * (s1["angle"] - 40)

    # change detection per station AND relative orbit (same geometry)
    g = s1.groupby(["sid", "relorb"])
    q05 = g["VV_norm40"].transform(lambda s: s.quantile(0.05))
    q95 = g["VV_norm40"].transform(lambda s: s.quantile(0.95))
    s1["VV_cd"] = (s1["VV_norm40"] - q05) / (q95 - q05).replace(0, np.nan)
    s1["VV_anom"] = s1["VV_norm40"] - g["VV_norm40"].transform("median")
    s1["VH_anom"] = s1["VH"] - g["VH"].transform("median")
    s1["relorb_n_obs"] = g["VV"].transform("size")
    gs = s1.groupby("sid")
    s1["dVV"] = gs["VV"].diff()
    s1["dVH"] = gs["VH"].diff()
    s1["days_since_prev_s1"] = gs["date"].diff().dt.days
    s1["VV_site_med"] = gs["VV_norm40"].transform("median")
    s1["VV_site_iqr"] = gs["VV_norm40"].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    s1["VV_site_q05"] = gs["VV_norm40"].transform(lambda s: s.quantile(0.05))
    s1["VV_site_q95"] = gs["VV_norm40"].transform(lambda s: s.quantile(0.95))
    s1["site_mean_VH"] = gs["VH"].transform("mean")
    keep = ["sid", "date", "platform", "VV", "VH", "CR_dB", "RVI", "SPAN_dB", "has_VH", "angle", "VV_norm40",
            "pass_asc", "relorb", "relorb_n_obs", "acq_hour", "dVV", "dVH", "days_since_prev_s1", "VV_cd",
            "VV_anom", "VH_anom", "VV_site_med", "VV_site_iqr", "VV_site_q05", "VV_site_q95", "site_mean_VH"]
    return s1[keep]


def rain_features(d: pd.DataFrame, col="precip") -> pd.DataFrame:
    d = d.sort_values(["sid", "date"])
    g = d.groupby("sid")[col]
    for n in (1, 3, 7, 14, 30):
        d[f"rain_{n}d"] = g.transform(lambda s: s.rolling(n, min_periods=1).sum())
    d["precip2d"] = g.transform(lambda s: s.rolling(2, min_periods=1).sum())
    for k in (0.85, 0.90, 0.95):
        def api(s, k=k):
            out, a = np.empty(len(s)), 0.0
            for i, p in enumerate(np.nan_to_num(s.to_numpy())):
                a = k * a + p
                out[i] = a
            return pd.Series(out, index=s.index)
        d[f"API_{int(k * 100):03d}"] = g.transform(api)

    def dsr(s):
        out, c = np.empty(len(s)), 99.0
        for i, p in enumerate(np.nan_to_num(s.to_numpy())):
            c = 0.0 if p >= 2 else c + 1
            out[i] = min(c, 99)
        return pd.Series(out, index=s.index)
    d["days_since_rain"] = g.transform(dsr)
    return d


# ----------------------------------------------------------------------------
# paper-style anchor table (the "before" model) + fix-ladder columns
# ----------------------------------------------------------------------------
def _s2_indices(B2, B4, B8, B11):
    ndvi = (B8 - B4) / (B8 + B4)
    ndmi = (B8 - B11) / (B8 + B11)
    evi = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1)
    return ndvi.where(ndvi.between(-1, 1)), ndmi.where(ndmi.between(-1, 1)), evi.where(evi.between(-1, 1))


def _asof(left, right, cols, tol_days, direction):
    left = left.sort_values("date")
    right = right.dropna(subset=cols, how="all").sort_values("date")[["sid", "date"] + cols]
    right = right.rename(columns={"date": "_obs_date"})
    out = pd.merge_asof(left, right, left_on="date", right_on="_obs_date", by="sid",
                        tolerance=pd.Timedelta(days=tol_days), direction=direction)
    return out.drop(columns="_obs_date")


def paper_anchor(P: dict, ins: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    s1 = P["s1raw"].copy()
    s1 = s1.sort_values(["sid", "date", "time"]).drop_duplicates(["sid", "date"])
    s1 = s1[s1["VV_pt"] > 0]
    a = s1[["sid", "date", "VV_pt", "VH_pt", "VV_mean", "VH_mean"]].copy()
    # paper: dB values of one 10 m pixel, indices computed directly on the dB numbers
    a["P_VV"] = 10 * np.log10(a["VV_pt"])
    a["P_VH"] = 10 * np.log10(a["VH_pt"].where(a["VH_pt"] > 0))
    a["P_ratio"] = a["P_VV"] / a["P_VH"]
    a["P_RVI"] = 4 * a["P_VH"] / (a["P_VV"] + a["P_VH"])
    a["P_SPAN"] = a["P_VV"] + a["P_VH"]
    # fix: the same pixel, indices in linear power
    a["L_CR_dB"] = a["P_VH"] - a["P_VV"]
    a["L_RVI"] = 4 * a["VH_pt"] / (a["VV_pt"] + a["VH_pt"])
    a["L_SPAN_dB"] = 10 * np.log10(a["VV_pt"] + a["VH_pt"])
    # fix: 50 m field-scale buffer mean in linear power
    a["B_VV"] = 10 * np.log10(a["VV_mean"])
    a["B_VH"] = 10 * np.log10(a["VH_mean"].where(a["VH_mean"] > 0))
    a["B_CR_dB"] = a["B_VH"] - a["B_VV"]
    a["B_RVI"] = 4 * a["VH_mean"] / (a["VV_mean"] + a["VH_mean"])
    a["B_SPAN_dB"] = 10 * np.log10(a["VV_mean"] + a["VH_mean"])

    # Sentinel-2, paper style: scene cloud filter only, single pixel, nearest scene within +-5 d
    s2 = P["s2raw"]
    raw = s2[s2["cloud_pct"] < PAPER_S2_MAX_CLOUD].copy()
    b = {k: raw[f"pt_{k}"] / 1e4 for k in ("B2", "B4", "B8", "B11")}
    raw["P_S2_NDVI"], raw["P_S2_NDMI"], raw["P_S2_EVI"] = _s2_indices(b["B2"], b["B4"], b["B8"], b["B11"])
    raw = raw.groupby(["sid", "date"], as_index=False)[["P_S2_NDVI", "P_S2_NDMI", "P_S2_EVI"]].mean()
    a = _asof(a, raw, ["P_S2_NDVI", "P_S2_NDMI", "P_S2_EVI"], PAPER_S2_WINDOW_DAYS, "nearest")
    # fix: SCL pixel mask over the 50 m buffer (>= 50 % clear), same matching rule
    m = P["s2masked"].rename(columns={"NDVI": "M_S2_NDVI", "NDMI": "M_S2_NDMI", "EVI": "M_S2_EVI"})
    a = _asof(a, m, ["M_S2_NDVI", "M_S2_NDMI", "M_S2_EVI"], PAPER_S2_WINDOW_DAYS, "nearest")

    # MODIS MOD13Q1 NDVI / EVI (latest 16-day composite, no QA filter as in the paper)
    md = P["mod13"].copy()
    md["P_MODIS_NDVI"] = md["NDVI"] * 1e-4
    md["P_MODIS_EVI"] = md["EVI"] * 1e-4
    a = _asof(a, md.groupby(["sid", "date"], as_index=False)[["P_MODIS_NDVI", "P_MODIS_EVI"]].mean(),
              ["P_MODIS_NDVI", "P_MODIS_EVI"], PAPER_MODIS_WINDOW_DAYS, "backward")

    ch = P["chirps"].sort_values(["sid", "date"]).copy()
    ch["P_precip"] = ch.groupby("sid")["precip"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    a = a.merge(ch[["sid", "date", "P_precip"]], on=["sid", "date"], how="left")
    a = a.merge(P["static"][["sid", "elev", "slope"]].rename(columns={"elev": "P_elev", "slope": "P_slope"}),
                on="sid", how="left")
    a["P_DOY"] = a["date"].dt.dayofyear
    a["P_Month"] = a["date"].dt.month

    st = stations[["sid", "scan_id", "name", "state", "region", "set", "paper16"]].copy()
    st["scan_id"] = st["scan_id"].astype(str)
    a = a.merge(st, on="sid", how="inner")
    t = ins[["sid", "date", "sm_5cm"]].rename(columns={"sid": "scan_id"})
    a = a.merge(t, on=["scan_id", "date"], how="left")
    prev = t.copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    a = a.merge(prev.rename(columns={"sm_5cm": "SM_lag1d"}), on=["scan_id", "date"], how="left")
    a = a.sort_values(["sid", "date"]).reset_index(drop=True)
    # paper's lag: in-situ value at the preceding Sentinel-1 sample of the same station
    a["SM_lag_prevS1"] = a.groupby("sid")["sm_5cm"].shift(1)
    return a.drop(columns=["VV_pt", "VH_pt", "VV_mean", "VH_mean"]).rename(columns={"sid": "site"})


def paper_daily(P: dict, ins: pd.DataFrame, stations: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Daily inputs of the paper's gap-fill model for every station (incl. the *_pc copies):
    CHIRPS 2-day rain, latest MOD13Q1 NDVI/EVI, elevation, slope, DOY, month, in-situ target."""
    st = stations[["sid", "scan_id", "region", "set", "paper16"]].copy()
    st["scan_id"] = st["scan_id"].astype(str)
    grid = pd.MultiIndex.from_product([st["sid"].unique(), pd.date_range(start, end, freq="D")],
                                      names=["sid", "date"]).to_frame(index=False)
    grid["date"] = grid["date"].astype("datetime64[ns]")
    ch = P["chirps"].sort_values(["sid", "date"]).copy()
    ch["P_precip"] = ch.groupby("sid")["precip"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    grid = grid.merge(ch[["sid", "date", "P_precip"]], on=["sid", "date"], how="left")
    md = P["mod13"].copy()
    md["P_MODIS_NDVI"] = md["NDVI"] * 1e-4
    md["P_MODIS_EVI"] = md["EVI"] * 1e-4
    md = md.groupby(["sid", "date"], as_index=False)[["P_MODIS_NDVI", "P_MODIS_EVI"]].mean()
    grid = _asof(grid, md, ["P_MODIS_NDVI", "P_MODIS_EVI"], PAPER_MODIS_WINDOW_DAYS, "backward")
    grid = grid.merge(P["static"][["sid", "elev", "slope"]].rename(columns={"elev": "P_elev", "slope": "P_slope"}),
                      on="sid", how="left")
    grid["P_DOY"] = grid["date"].dt.dayofyear
    grid["P_Month"] = grid["date"].dt.month
    grid = grid.merge(st, on="sid", how="inner")
    t = ins[["sid", "date", "sm_5cm"]].rename(columns={"sid": "scan_id"})
    grid = grid.merge(t, on=["scan_id", "date"], how="left")
    return grid.sort_values(["sid", "date"]).rename(columns={"sid": "site"}).reset_index(drop=True)


# ----------------------------------------------------------------------------
def assemble(P: dict, ins: pd.DataFrame, stations: pd.DataFrame, start: str, end: str):
    stations = stations[stations["set"] == "scan"].copy()
    sids = stations["sid"].astype(str).unique()
    grid = pd.MultiIndex.from_product([sids, pd.date_range(start, end, freq="D")],
                                      names=["sid", "date"]).to_frame(index=False)
    grid["date"] = grid["date"].astype("datetime64[ns]")
    for k in ("chirps", "era5l", "smapl4"):
        grid = grid.merge(P[k], on=["sid", "date"], how="left")
    grid = rain_features(grid, "precip")
    base = grid[["sid", "date"]]
    grid = grid.merge(_interp_per_site(base, P["smapl3e"], ["smap_l3e_sm"], 3), on=["sid", "date"], how="left")
    grid = grid.merge(_interp_per_site(base, P["nbar"], ["MODIS_NDVI", "MODIS_NDMI", "MODIS_EVI"], 16),
                      on=["sid", "date"], how="left")
    grid = grid.merge(_interp_per_site(base, P["lai"], ["LAI"], 12), on=["sid", "date"], how="left")
    grid = grid.merge(_interp_per_site(base, P["lst"], ["LST_day", "LST_night", "dLST"], 5),
                      on=["sid", "date"], how="left")
    grid = grid.merge(_interp_per_site(base, P["s2masked"], ["NDVI", "NDMI", "EVI", "S2_BSI", "S2_NDWI"], 30,
                                       age_col="s2_age_days"), on=["sid", "date"], how="left")
    grid = grid.merge(P["static"], on="sid", how="left")
    meta = stations[["sid", "name", "state", "region", "lat", "lon", "network", "paper16"]].copy()
    meta["region"] = meta["region"].fillna(meta["state"].map(CLIMATE_REGION))
    grid = grid.merge(meta, on="sid", how="left")

    # ---- Sentinel-1: anchor rows + carried-forward state on gap days ----
    s1 = s1_features(P["s1raw"])
    s1["has_s1"] = 1
    daily = grid.merge(s1, on=["sid", "date"], how="left")
    daily["has_s1"] = daily["has_s1"].fillna(0).astype(int)
    daily = daily.sort_values(["sid", "date"]).reset_index(drop=True)
    for c in ("VV", "VH", "VV_cd", "VV_anom", "CR_dB", "VV_norm40"):
        daily[f"last_{c}"] = daily.groupby("sid")[c].ffill()
    last_date = daily["date"].where(daily["has_s1"] == 1)
    daily["s1_age_days"] = (daily["date"] - last_date.groupby(daily["sid"]).ffill()).dt.days

    # ---- calendar + site climatology (from satellite/reanalysis, no in-situ) ----
    daily["DOY"] = daily["date"].dt.dayofyear
    daily["Month"] = daily["date"].dt.month
    daily["doy_sin"] = np.sin(2 * np.pi * daily["DOY"] / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["DOY"] / 365.25)
    g = daily.groupby("sid")
    for c in ("precip2d", "NDVI", "NDMI", "LAI"):
        daily[f"site_mean_{c}"] = g[c].transform("mean")
    daily["site_wet_frac"] = g["precip2d"].transform(lambda s: (s > 1).mean())
    daily["NDVI_anom"] = daily["NDVI"] - g["NDVI"].transform("median")
    for c in ("VV_site_med", "VV_site_iqr", "VV_site_q05", "VV_site_q95", "site_mean_VH"):
        daily[c] = daily.groupby("sid")[c].transform(lambda s: s.ffill().bfill())

    # ---- in-situ target + lags ----
    t = ins[["sid", "date", "sm_5cm"]].copy()
    t["sid"] = t["sid"].astype(str)
    daily = daily.merge(t, on=["sid", "date"], how="left")
    daily = daily.sort_values(["sid", "date"]).reset_index(drop=True)
    daily["SM_lag1d"] = daily.groupby("sid")["sm_5cm"].shift(1)
    daily = daily.rename(columns={"sid": "site"})
    anchor = daily[daily["has_s1"] == 1].copy()
    anchor["SM_lag_prevS1"] = anchor.groupby("site")["sm_5cm"].shift(1)
    return daily, anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scan", required=True, help="folder of AWDB hourly parquet files (insitu_v2.py scan)")
    ap.add_argument("--stations", required=True, help="stations_v2.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--out", default="../data/processed_v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = pd.read_csv(a.stations, dtype={"sid": str, "scan_id": str})
    ins = insitu_daily(a.scan)
    ins = ins[(ins.date >= a.start) & (ins.date <= a.end)]
    ins.to_parquet(os.path.join(a.out, "insitu_daily.parquet"), index=False)
    print("in-situ daily", ins.shape, "stations with 5 cm data", ins.dropna(subset=["sm_5cm"]).sid.nunique())
    P = prep_products(a.cache)
    pa = paper_anchor(P, ins, st)
    pa.to_parquet(os.path.join(a.out, "paper_anchor.parquet"), index=False)
    print("paper anchor", pa.shape, "labelled", pa["sm_5cm"].notna().sum())
    paper_daily(P, ins, st, a.start, a.end).to_parquet(os.path.join(a.out, "paper_daily.parquet"), index=False)
    daily, anchor = assemble(P, ins, st, a.start, a.end)
    daily.to_parquet(os.path.join(a.out, "daily.parquet"), index=False)
    anchor.to_parquet(os.path.join(a.out, "anchor.parquet"), index=False)
    st.to_csv(os.path.join(a.out, "stations.csv"), index=False)
    print("daily", daily.shape, "anchor", anchor.shape, "labelled anchors", anchor["sm_5cm"].notna().sum(),
          "labelled days", daily["sm_5cm"].notna().sum())


if __name__ == "__main__":
    main()
