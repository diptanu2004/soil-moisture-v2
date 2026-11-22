"""
POLARIS v1.0 (Chaney et al. 2019, 30 m, SSURGO-based) soil properties at the SCAN stations, read over
HTTP from the 1x1 degree tiles at hydrology.cee.duke.edu (no download of whole tiles; GDAL range reads).

Station coordinates in stations.csv have 2 decimals (~ +/- 500 m), so besides the point value a
~1 km window mean (33 x 33 pixels) is extracted. Static soil properties only -> satellite/map-derived,
no in-situ input.

  python polaris_extract.py ../data/processed_v2/stations.csv ../data/raw_v2/polaris_stations.parquet
Resumable: finished (station, variable, depth) cells are cached in <out>.partial.parquet
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

BASE = "/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
VARS = ["theta_s", "theta_r", "clay", "sand", "silt", "bd", "om", "ksat", "n", "alpha"]
DEPTHS = ["0_5", "5_15"]
HALF = 16  # 33 x 33 pixels ~ 1 km


def tile(lat, lon):
    a, o = math.floor(lat), math.floor(lon)
    return f"lat{a}{a + 1}_lon{o}{o + 1}.tif"


def read(lat, lon, var, depth):
    u = f"{BASE}/{var}/mean/{depth}/{tile(lat, lon)}"
    with rasterio.open(u) as r:
        row, col = r.index(lon, lat)
        # points within half a pixel of a tile edge can index one past the last row/column: snap (<= 30 m)
        row, col = min(max(row, 0), r.height - 1), min(max(col, 0), r.width - 1)
        r0, c0 = max(row - HALF, 0), max(col - HALF, 0)
        w = r.read(1, window=Window(c0, r0, min(2 * HALF + 1, r.width - c0), min(2 * HALF + 1, r.height - r0)))
        w = np.where(w == r.nodata, np.nan, w).astype(float)
        pt = w[row - r0, col - c0]
        return float(pt), float(np.nanmean(w)) if np.isfinite(w).any() else np.nan


def main(stations_csv, out):
    st = pd.read_csv(stations_csv)
    st["site"] = st["sid"].astype(str)
    part = out + ".partial.parquet"
    done = pd.read_parquet(part) if os.path.exists(part) else pd.DataFrame(columns=["site", "var", "depth", "pt", "win"])
    have = set(zip(done.site, done["var"], done.depth))
    jobs = [(s.site, s.lat, s.lon, v, d) for s in st.itertuples() for v in VARS for d in DEPTHS
            if (s.site, v, d) not in have]
    print(f"{len(st)} stations, {len(jobs)} reads to do", flush=True)
    rows, fails = [], 0
    with ThreadPoolExecutor(16) as ex:
        fut = {ex.submit(read, lat, lon, v, d): (site, v, d) for site, lat, lon, v, d in jobs}
        for i, f in enumerate(as_completed(fut), 1):
            site, v, d = fut[f]
            try:
                pt, win = f.result()
                rows.append(dict(site=site, var=v, depth=d, pt=pt, win=win))
            except Exception as e:  # noqa: BLE001
                fails += 1
                if fails <= 5:
                    print("fail", site, v, d, type(e).__name__, str(e)[:120], flush=True)
            if i % 400 == 0 or i == len(jobs):
                pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(part)
                print(f"{i}/{len(jobs)} reads, {fails} failed", flush=True)
    allr = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    allr.to_parquet(part)
    wide = allr.pivot_table(index="site", columns=["var", "depth"], values=["pt", "win"])
    wide.columns = [f"pol_{v}_{d}_{k}" for k, v, d in wide.columns]
    wide.reset_index().to_parquet(out)
    print(f"wrote {out}: {wide.shape}, {fails} failed reads, NaN share {wide.isna().mean().mean():.1%}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
