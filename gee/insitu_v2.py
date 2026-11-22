"""
In-situ soil moisture loaders for the v2 pipeline (5 cm, daily, % volumetric).

  scan_daily(triplet, start, end)   NRCS SCAN via the Report Generator CSV service
  uscrn_daily(path_or_url)          NOAA USCRN daily01 files (CRND0103-YYYY-*.txt)
  ismn_daily(archive_dir)           any ISMN download (SCAN, USCRN, SOILSCAPE, REMEDHUS,
                                    SMOSMANIA, TERENO, OZNET, MAQU, CTP_SMTMN, ...)

All return columns: sid, date, sm_5cm  (+ precip/tair when the network provides them).

The SCAN/USCRN URL formats follow the public services as documented by NRCS / NCEI;
they could not be called from the environment where this file was written, so
verify on first use. The ISMN loader was tested on the pytesmo/ISMN test archive.
"""
from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# NRCS SCAN
# ----------------------------------------------------------------------------
SCAN_URL = ("https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/"
            "{triplet}%7Cid=%22%22%7Cname/{start},{end}/"
            "SMS:-2:value,SMS:-4:value,PRCP::value,TAVG::value,STO:-2:value")


def scan_daily(triplet: str, start: str, end: str) -> pd.DataFrame:
    """triplet like '2068:IA:SCAN'. Returns daily 2 in (5 cm) SM + precip + air T + soil T."""
    url = SCAN_URL.format(triplet=triplet, start=start, end=end)
    txt = urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "ignore")
    lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = ["date", "sm_5cm", "sm_10cm", "prcp_in", "tavg_F", "stemp_5cm_F"][: df.shape[1]]
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["insitu_precip_mm"] = df.pop("prcp_in") * 25.4 if "prcp_in" in df else np.nan
    if "stemp_5cm_F" in df:
        df["stemp_5cm_C"] = (df.pop("stemp_5cm_F") - 32) / 1.8
        df.loc[df["stemp_5cm_C"] <= 0.5, "sm_5cm"] = np.nan          # frozen soil -> invalid
    df["sid"] = triplet.split(":")[0]
    df.loc[~df["sm_5cm"].between(0.5, 60), "sm_5cm"] = np.nan
    return df


# The daily SMS value of the Report Generator / AWDB is the start-of-day (00:00) reading, not a
# daily mean. For a daily mean (as in the 2015-2018 package data) pull hourly values from the
# AWDB REST API (~3x faster than the hourly Report Generator) and aggregate with hourly_to_daily.
AWDB_URL = ("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets={triplet}"
            "&elements=SMS:-2,STO:-2&duration=HOURLY&beginDate={start}&endDate={end}")


def scan_hourly_awdb(triplet: str, start: str, end: str) -> pd.DataFrame:
    """Hourly 2 in (5 cm) soil moisture (% vol) and soil temperature (deg F) from AWDB."""
    import json
    import requests
    r = requests.get(AWDB_URL.format(triplet=triplet, start=start, end=end), timeout=300)
    r.raise_for_status()           # AWDB answers intermittent 403s under load; callers retry
    js = json.loads(r.content)
    cols = {}
    for el in (js[0]["data"] if js else []):
        name = {"SMS": "sms_2in", "STO": "sto_2in_F"}[el["stationElement"]["elementCode"]]
        v = pd.DataFrame(el.get("values", []))
        if not v.empty and "value" in v:
            cols[name] = pd.Series(pd.to_numeric(v["value"], errors="coerce").to_numpy(),
                                   index=pd.to_datetime(v["date"]))
    df = pd.DataFrame(cols)
    df.index.name = "datetime"
    return df.reset_index()


def hourly_to_daily(h: pd.DataFrame) -> pd.DataFrame:
    """QC as in src/build_dataset.py (range, spike filter, >= 12 h per day, flat-line) plus a
    frozen-soil mask (daily mean 5 cm soil temperature <= 0.5 C)."""
    h = h.assign(datetime=pd.to_datetime(h["datetime"])).set_index("datetime").sort_index()
    h = h[~h.index.duplicated()]
    s = h["sms_2in"].where((h["sms_2in"] > 0.5) & (h["sms_2in"] <= 60)) if "sms_2in" in h else pd.Series(dtype=float)
    med = s.rolling(25, center=True, min_periods=5).median()
    s = s.where((s - med).abs() <= 8)
    d = s.resample("D").agg(["mean", "count"])
    daily = pd.DataFrame({"sm_5cm": d["mean"].where(d["count"] >= 12)})
    roll_std = daily["sm_5cm"].rolling(10, min_periods=10).std()
    stuck = roll_std.fillna(1) < 1e-6
    stuck = stuck | stuck.shift(-9, fill_value=False)
    daily.loc[stuck, "sm_5cm"] = np.nan
    if "sto_2in_F" in h:
        t = h["sto_2in_F"].where(h["sto_2in_F"].between(-40, 140))
        daily["stemp_5cm_C"] = ((t - 32) / 1.8).resample("D").mean()
        daily.loc[daily["stemp_5cm_C"] <= 0.5, "sm_5cm"] = np.nan
    return daily.reset_index(names="date")


# ----------------------------------------------------------------------------
# NOAA USCRN daily01 (fixed column order from products/daily01/HEADERS.txt)
# ----------------------------------------------------------------------------
USCRN_COLS = ["WBANNO", "LST_DATE", "CRX_VN", "LONGITUDE", "LATITUDE", "T_DAILY_MAX", "T_DAILY_MIN",
              "T_DAILY_MEAN", "T_DAILY_AVG", "P_DAILY_CALC", "SOLARAD_DAILY", "SUR_TEMP_DAILY_TYPE",
              "SUR_TEMP_DAILY_MAX", "SUR_TEMP_DAILY_MIN", "SUR_TEMP_DAILY_AVG", "RH_DAILY_MAX",
              "RH_DAILY_MIN", "RH_DAILY_AVG", "SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY",
              "SOIL_MOISTURE_20_DAILY", "SOIL_MOISTURE_50_DAILY", "SOIL_MOISTURE_100_DAILY",
              "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY", "SOIL_TEMP_20_DAILY", "SOIL_TEMP_50_DAILY",
              "SOIL_TEMP_100_DAILY"]
USCRN_BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/{year}/{fname}"
USCRN_STATIONS = "https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv"


def uscrn_daily(path_or_url: str) -> pd.DataFrame:
    src = urllib.request.urlopen(path_or_url) if path_or_url.startswith("http") else open(path_or_url, "rb")
    df = pd.read_csv(src, sep=r"\s+", header=None, names=USCRN_COLS, dtype={"LST_DATE": str})
    df = df.replace([-9999.0, -99.0, -99.000], np.nan)
    out = pd.DataFrame({
        "sid": "CRN" + df["WBANNO"].astype(str),
        "date": pd.to_datetime(df["LST_DATE"], format="%Y%m%d"),
        "sm_5cm": 100 * df["SOIL_MOISTURE_5_DAILY"],          # m3/m3 -> % vol
        "sm_10cm": 100 * df["SOIL_MOISTURE_10_DAILY"],
        "insitu_precip_mm": df["P_DAILY_CALC"],
        "tair_C": df["T_DAILY_MEAN"],
        "stemp_5cm_C": df["SOIL_TEMP_5_DAILY"],
        "lat": df["LATITUDE"], "lon": df["LONGITUDE"],
    })
    out.loc[out["stemp_5cm_C"] <= 0.5, "sm_5cm"] = np.nan
    out.loc[~out["sm_5cm"].between(0.5, 60), "sm_5cm"] = np.nan
    return out


# ----------------------------------------------------------------------------
# ISMN (tested with ismn 1.5)
# ----------------------------------------------------------------------------
def ismn_daily(archive_dir: str, max_depth_m: float = 0.05, networks=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily, stations). Drops values flagged by ISMN as dubious (D*),
    outside plausible range (C*) or missing (M); keeps G (good) and U (unchecked)."""
    from ismn.interface import ISMN_Interface

    ds = ISMN_Interface(archive_dir, parallel=False, network=networks)
    rows, meta = [], []
    for network, station, sensor in ds.collection.iter_sensors(variable="soil_moisture",
                                                                depth=[0.0, max_depth_m]):
        df = sensor.read_data()
        if "soil_moisture_flag" in df:
            fl = df["soil_moisture_flag"].astype(str)
            df = df[~(fl.str.contains("D") | fl.str.contains("C") | (fl == "M"))]
        if df.empty:
            continue
        sid = f"{network.name}:{station.name}"
        d = (df["soil_moisture"].resample("D").agg(["mean", "count"]))
        d = d[d["count"] >= 12]
        rows.append(pd.DataFrame({"sid": sid, "date": d.index, "sm_5cm": 100 * d["mean"].to_numpy()}))
        meta.append(dict(sid=sid, network=network.name, name=station.name, lat=station.lat, lon=station.lon,
                         sensor=str(sensor.instrument)))
    daily = (pd.concat(rows).groupby(["sid", "date"], as_index=False)["sm_5cm"].mean()
             if rows else pd.DataFrame(columns=["sid", "date", "sm_5cm"]))
    stations = pd.DataFrame(meta).drop_duplicates("sid")
    return daily, stations


def fetch_scan_all(stations_csv: str, cache: str, start: str = "2019-01-01", end: str = "2023-12-31",
                   workers: int = 3):
    """Hourly AWDB pull for every SCAN station (one request per station-year), cached per
    station-year so an interrupted run resumes."""
    import concurrent.futures as cf
    import time
    st = pd.read_csv(stations_csv, dtype={"sid": str})
    st = st[st["set"] == "scan"] if "set" in st else st
    os.makedirs(cache, exist_ok=True)
    jobs = [(r.sid, r.station_triplet, y) for r in st.itertuples()
            for y in range(int(start[:4]), int(end[:4]) + 1)]

    def one(job):
        sid, trip, y = job
        out = os.path.join(cache, f"{sid}_{y}.parquet")
        if os.path.exists(out):
            return "cached"
        for k in range(4):
            try:
                h = scan_hourly_awdb(trip, f"{y}-01-01", f"{y}-12-31")
                h.to_parquet(out, index=False)
                return f"ok {len(h)}"
            except Exception as e:
                err = str(e)[:150]
                time.sleep(15 * 2 ** k)
        return f"FAILED {err}"

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (job, res) in enumerate(zip(jobs, ex.map(one, jobs))):
            if (i + 1) % 50 == 0 or res.startswith("FAILED"):
                print(f"[{i + 1}/{len(jobs)}] {job} {res} {(time.time() - t0) / 60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        fetch_scan_all(sys.argv[2], sys.argv[3])
        sys.exit()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        d, s = ismn_daily(sys.argv[1])
        print(s)
        print(d.groupby("sid").sm_5cm.describe())
