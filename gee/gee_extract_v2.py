"""
Upgraded Google Earth Engine extraction for station-scale soil moisture (v2).

v2.1 (Oct 2026, run against real Earth Engine): use the per-station cached pull at the
bottom of this file (`run_cached`, CLI: python gee_extract_v2.py --stations ... --cache ...).
Fixes found on first real use of the export path below:
  * reduceRegions over the bounding box of all stations returned every S1 scene of the box
    for every station (2233 mostly-empty rows per station-year) -> per-station filterBounds
  * coarse products (CHIRPS, ERA5-Land, MODIS LST) were 100 % empty with a 50 m buffer
    (buffer smaller than one pixel, no pixel centre inside) -> point sampling (getRegion)
  * SMAP L3E/L4 hit "User memory limit exceeded" -> one request per year, L4 averaged to
    daily on the server
  * Drive exports are not available (account authorised without the Drive scope) ->
    computeFeatures / getRegion, one parquet per station x product, resumable

Original usage (Drive export path, kept for reference):

    import gee_extract_v2 as G
    G.init("your-gee-project")
    st = G.load_stations("stations_paper16_corrected.csv")          # or the expanded lists
    G.export_all(st, "2019-01-01", "2023-12-31", folder="sm_v2")      # -> Google Drive CSVs
    # ...wait for the Drive exports (Tasks tab) then run assemble_v2.py

NOTE: written against the public Earth Engine catalogue (collection IDs and band
names checked against the catalogue pages, Oct 2026) but NOT executed in the
environment where it was written (no Earth Engine access there). Check the first
export of each product before scaling up.

What changed versus the original notebook
  * station coordinates come from the NRCS/ISMN metadata, never typed by hand
  * features are averaged over a field-scale buffer (default 50 m radius) instead
    of one 10 m pixel, which reduces speckle and coordinate-error sensitivity
  * Sentinel-1 is read in LINEAR power (S1_GRD_FLOAT); VV/VH ratio, RVI and SPAN
    are computed in linear units (the v1 code computed them on dB numbers);
    incidence angle, pass direction and relative orbit are kept for
    per-orbit change-detection normalisation
  * Sentinel-2 uses the Scene Classification Layer for pixel-level cloud,
    shadow and snow masking (v1 used only the scene-level cloud percentage)
  * MODIS uses the BRDF-corrected daily NBAR product (MCD43A4) with its quality
    band, plus MODIS LST day/night (thermal-inertia proxy)
  * daily drivers for the gap days: CHIRPS, ERA5-Land (soil water layer 1,
    temperature, evaporation, PET, radiation, snow) and SMAP L4 / L3-enhanced
    soil moisture (the passive-microwave prior from the project plan)
  * static descriptors: SoilGrids texture/bulk density/SOC (0-5 cm), SRTM,
    MERIT-Hydro upstream area (topographic wetness index), ESA WorldCover
  * everything is exported with reduceRegions over whole collections (one task
    per product) instead of one getInfo() call per date, so hundreds of
    stations x several years are feasible
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import ee
import pandas as pd

# ----------------------------------------------------------------------------
# Earth Engine collection IDs (public catalogue)
# ----------------------------------------------------------------------------
C = dict(
    S1="COPERNICUS/S1_GRD_FLOAT",          # linear power; bands VV, VH, angle
    S2="COPERNICUS/S2_SR_HARMONIZED",      # L2A surface reflectance + SCL
    NBAR="MODIS/061/MCD43A4",              # daily BRDF-adjusted reflectance, 500 m
    LAI="MODIS/061/MCD15A3H",              # 4-day LAI/FPAR 500 m
    LST_TERRA="MODIS/061/MOD11A1",         # daily LST 1 km (10:30 / 22:30)
    LST_AQUA="MODIS/061/MYD11A1",          # daily LST 1 km (13:30 / 01:30)
    CHIRPS="UCSB-CHG/CHIRPS/DAILY",
    ERA5L="ECMWF/ERA5_LAND/DAILY_AGGR",
    SMAP_L4="NASA/SMAP/SPL4SMGP/008",      # 3-hourly, ~9-11 km, from 2015-03-31
    SMAP_L3E_OLD="NASA/SMAP/SPL3SMP_E/005",  # daily 9 km, up to 2023-12
    SMAP_L3E_NEW="NASA/SMAP/SPL3SMP_E/006",  # daily 9 km, from 2023-12-04
    SRTM="USGS/SRTMGL1_003",
    MERIT="MERIT/Hydro/v1_0_1",
    WORLDCOVER="ESA/WorldCover/v200",
    SAND="projects/soilgrids-isric/sand_mean",
    CLAY="projects/soilgrids-isric/clay_mean",
    SILT="projects/soilgrids-isric/silt_mean",
    BDOD="projects/soilgrids-isric/bdod_mean",
    SOC="projects/soilgrids-isric/soc_mean",
)


def init(project: str):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print("Earth Engine initialised")


# ----------------------------------------------------------------------------
# stations
# ----------------------------------------------------------------------------
def load_stations(csv_path: str, lat_col: Optional[str] = None, lon_col: Optional[str] = None) -> pd.DataFrame:
    """Read a station table with columns sid, lat, lon (other columns kept).
    Accepts the CSVs shipped with this package (stations_*.csv)."""
    st = pd.read_csv(csv_path)
    lat_col = lat_col or next(c for c in ("nrcs_lat", "lat", "latitude") if c in st)
    lon_col = lon_col or next(c for c in ("nrcs_lon", "lon", "longitude") if c in st)
    if "sid" not in st:
        st["sid"] = st["scan_id"].astype(str) if "scan_id" in st else st.index.astype(str)
    st = st.rename(columns={lat_col: "lat", lon_col: "lon"})
    return st[["sid", "lat", "lon"] + [c for c in st.columns if c not in ("sid", "lat", "lon")]]


def stations_fc(st: pd.DataFrame, buffer_m: float = 50) -> ee.FeatureCollection:
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(buffer_m),
                        {"sid": str(r.sid)}) for r in st.itertuples()]
    return ee.FeatureCollection(feats)


# ----------------------------------------------------------------------------
# per-image preprocessing
# ----------------------------------------------------------------------------
def _s1_prep(img):
    # border-noise / no-data guard: linear power below 1e-3 (-30 dB) is masked
    vv = img.select("VV")
    vh = img.select("VH")
    m = vv.gt(1e-3).And(vh.gt(1e-4))
    return (img.select(["VV", "VH", "angle"]).updateMask(m)
            .set("pass", img.get("orbitProperties_pass"),
                 "relorb", img.get("relativeOrbitNumber_start"),
                 "platform", img.get("platform_number")))


def _s2_prep(img):
    scl = img.select("SCL")
    # keep: 4 vegetation, 5 bare soil, 6 water, 7 unclassified (drop shadow 3, cloud 8-10, snow 11)
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    b = img.select(["B2", "B3", "B4", "B8", "B11", "B12"]).divide(10000).updateMask(good)
    ndvi = b.normalizedDifference(["B8", "B4"]).rename("S2_NDVI")
    ndmi = b.normalizedDifference(["B8", "B11"]).rename("S2_NDMI")
    ndwi = b.normalizedDifference(["B3", "B8"]).rename("S2_NDWI")
    evi = b.expression("2.5*(N-R)/(N+6*R-7.5*B+1)",
                       {"N": b.select("B8"), "R": b.select("B4"), "B": b.select("B2")}).rename("S2_EVI")
    bsi = b.expression("((S+R)-(N+B))/((S+R)+(N+B))",
                       {"S": b.select("B11"), "R": b.select("B4"), "N": b.select("B8"),
                        "B": b.select("B2")}).rename("S2_BSI")
    valid = good.rename("S2_valid")
    return ee.Image.cat([ndvi, ndmi, ndwi, evi, bsi, b.select(["B11", "B12"]).rename(["S2_B11", "S2_B12"]),
                         valid]).copyProperties(img, ["system:time_start"])


def _nbar_prep(img):
    q1 = img.select("BRDF_Albedo_Band_Mandatory_Quality_Band1")
    q2 = img.select("BRDF_Albedo_Band_Mandatory_Quality_Band2")
    good = q1.lte(1).And(q2.lte(1))       # 0 full inversion, 1 magnitude inversion
    r = img.select(["Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band3",
                    "Nadir_Reflectance_Band6"], ["R", "N", "B", "S"]).multiply(0.0001).updateMask(good)
    ndvi = r.normalizedDifference(["N", "R"]).rename("MODIS_NDVI")
    ndmi = r.normalizedDifference(["N", "S"]).rename("MODIS_NDMI")
    evi = r.expression("2.5*(N-R)/(N+6*R-7.5*B+1)",
                       {"N": r.select("N"), "R": r.select("R"), "B": r.select("B")}).rename("MODIS_EVI")
    return ee.Image.cat([ndvi, ndmi, evi]).copyProperties(img, ["system:time_start"])


def _lst_prep(prefix):
    def f(img):
        qd = img.select("QC_Day").bitwiseAnd(3).eq(0)       # bits 0-1 == 00: good quality
        qn = img.select("QC_Night").bitwiseAnd(3).eq(0)
        d = img.select("LST_Day_1km").multiply(0.02).updateMask(qd).rename(f"{prefix}_LSTday")
        n = img.select("LST_Night_1km").multiply(0.02).updateMask(qn).rename(f"{prefix}_LSTnight")
        return d.addBands(n).copyProperties(img, ["system:time_start"])
    return f


def _smap_l3e_prep(img):
    am = img.select("soil_moisture_am").updateMask(img.select("retrieval_qual_flag_am").eq(0))
    pm = img.select("soil_moisture_pm").updateMask(img.select("retrieval_qual_flag_pm").eq(0))
    return am.rename("SMAP_L3E_am").addBands(pm.rename("SMAP_L3E_pm")).copyProperties(img, ["system:time_start"])


# ----------------------------------------------------------------------------
# generic collection -> table
# ----------------------------------------------------------------------------
def collection_to_table(ic: ee.ImageCollection, fc: ee.FeatureCollection, scale: float,
                        extra_props: Iterable[str] = (), reducer: Optional[ee.Reducer] = None,
                        tile_scale: int = 2) -> ee.FeatureCollection:
    """reduceRegions every image over every station buffer -> flat table.
    Rows without a value (fully masked) are dropped."""
    reducer = reducer or ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    extra_props = list(extra_props)

    def per_image(img):
        props = {"date": img.date().format("YYYY-MM-dd"), "time": img.date().format("HH:mm")}
        for p in extra_props:
            props[p] = img.get(p)
        out = img.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=tile_scale)
        return out.map(lambda f: ee.Feature(None, f.toDictionary()).set(props))

    return ic.map(per_image).flatten()


def export_table(table: ee.FeatureCollection, name: str, folder: str, wait: bool = False):
    task = ee.batch.Export.table.toDrive(collection=table, description=name, folder=folder,
                                         fileNamePrefix=name, fileFormat="CSV")
    task.start()
    print("started export", name)
    if wait:
        while task.active():
            time.sleep(30)
        print(name, task.status()["state"])
    return task


def table_to_df(table: ee.FeatureCollection) -> pd.DataFrame:
    """Small extractions only (a few stations / one year): pull directly with getInfo."""
    feats = table.getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in feats])


# ----------------------------------------------------------------------------
# product builders
# ----------------------------------------------------------------------------
def s1_table(fc, region, start, end):
    ic = (ee.ImageCollection(C["S1"]).filterDate(start, end).filterBounds(region)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .map(_s1_prep))
    red = (ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
           .combine(ee.Reducer.count(), sharedInputs=True))
    return collection_to_table(ic, fc, 10, ["pass", "relorb", "platform"], reducer=red)


def s2_table(fc, region, start, end, max_scene_cloud=70):
    ic = (ee.ImageCollection(C["S2"]).filterDate(start, end).filterBounds(region)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud)).map(_s2_prep))
    return collection_to_table(ic, fc, 10, reducer=ee.Reducer.mean())


def nbar_table(fc, region, start, end):
    ic = ee.ImageCollection(C["NBAR"]).filterDate(start, end).map(_nbar_prep)
    return collection_to_table(ic, fc, 500, reducer=ee.Reducer.mean())


def _lai_prep(img):
    good = img.select("FparLai_QC").bitwiseAnd(1).eq(0)      # bit 0 = 0: good quality main algorithm
    return img.select("Lai").multiply(0.1).updateMask(good).rename("LAI").copyProperties(img, ["system:time_start"])


def lai_table(fc, region, start, end):
    ic = ee.ImageCollection(C["LAI"]).filterDate(start, end).map(_lai_prep)
    return collection_to_table(ic, fc, 500, reducer=ee.Reducer.mean())


def lst_table(fc, region, start, end):
    t = ee.ImageCollection(C["LST_TERRA"]).filterDate(start, end).map(_lst_prep("T"))
    a = ee.ImageCollection(C["LST_AQUA"]).filterDate(start, end).map(_lst_prep("A"))
    return (collection_to_table(t, fc, 1000, reducer=ee.Reducer.mean()),
            collection_to_table(a, fc, 1000, reducer=ee.Reducer.mean()))


def chirps_table(fc, region, start, end):
    ic = ee.ImageCollection(C["CHIRPS"]).filterDate(start, end).select("precipitation")
    return collection_to_table(ic, fc, 5566, reducer=ee.Reducer.mean())


def era5l_table(fc, region, start, end):
    bands = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "temperature_2m",
             "temperature_2m_max", "skin_temperature", "total_precipitation_sum",
             "total_evaporation_sum", "potential_evaporation_sum",
             "surface_net_solar_radiation_sum", "snow_cover"]
    ic = ee.ImageCollection(C["ERA5L"]).filterDate(start, end).select(bands)
    return collection_to_table(ic, fc, 11132, reducer=ee.Reducer.mean())


def smap_l4_table(fc, region, start, end):
    ic = ee.ImageCollection(C["SMAP_L4"]).filterDate(start, end).select(
        ["sm_surface", "sm_rootzone", "sm_surface_wetness", "surface_temp"])
    return collection_to_table(ic, fc, 11000, reducer=ee.Reducer.mean())   # 3-hourly rows


def smap_l3e_table(fc, region, start, end):
    ic = (ee.ImageCollection(C["SMAP_L3E_OLD"]).merge(ee.ImageCollection(C["SMAP_L3E_NEW"]))
          .filterDate(start, end).map(_smap_l3e_prep))
    return collection_to_table(ic, fc, 9000, reducer=ee.Reducer.mean())


def static_table(fc):
    srtm = ee.Image(C["SRTM"])
    slope = ee.Terrain.slope(srtm)
    aspect = ee.Terrain.aspect(srtm)
    upa = ee.Image(C["MERIT"]).select("upa")                      # km2
    # TWI = ln(a / tan(beta)); a = upstream area per unit contour width (m), beta slope
    a = upa.multiply(1e6).divide(90)
    twi = a.divide(slope.multiply(3.14159265 / 180).tan().max(0.001)).log().rename("TWI")
    soil = ee.Image.cat([
        ee.Image(C[k]).select(".*0-5cm_mean").rename(k.lower()) for k in ("SAND", "CLAY", "SILT", "BDOD", "SOC")
    ])
    wc = ee.Image(ee.ImageCollection(C["WORLDCOVER"]).first()).select("Map").rename("worldcover")
    img = ee.Image.cat([srtm.rename("elevation"), slope.rename("slope"), aspect.rename("aspect"),
                        twi, soil])
    t1 = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
    t2 = wc.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=10)
    return t1, t2


# ----------------------------------------------------------------------------
def export_all(st: pd.DataFrame, start: str, end: str, folder: str = "sm_v2", buffer_m: float = 50,
               products: Optional[List[str]] = None, prefix: str = "") -> Dict[str, ee.batch.Task]:
    """Start one Drive export per product. Split long periods / many stations into
    several calls (e.g. one per year) if a task fails with 'too many concurrent
    aggregations' or runs out of memory."""
    fc = stations_fc(st, buffer_m)
    region = fc.geometry().bounds()
    products = products or ["s1", "s2", "nbar", "lai", "lst", "chirps", "era5l", "smap_l4", "smap_l3e", "static"]
    tasks = {}
    tag = f"{prefix}{start[:4]}_{end[:4]}"
    if "s1" in products:
        tasks["s1"] = export_table(s1_table(fc, region, start, end), f"S1_{tag}", folder)
    if "s2" in products:
        tasks["s2"] = export_table(s2_table(fc, region, start, end), f"S2_{tag}", folder)
    if "nbar" in products:
        tasks["nbar"] = export_table(nbar_table(fc, region, start, end), f"NBAR_{tag}", folder)
    if "lai" in products:
        tasks["lai"] = export_table(lai_table(fc, region, start, end), f"LAI_{tag}", folder)
    if "lst" in products:
        t, a = lst_table(fc, region, start, end)
        tasks["lst_t"] = export_table(t, f"LSTT_{tag}", folder)
        tasks["lst_a"] = export_table(a, f"LSTA_{tag}", folder)
    if "chirps" in products:
        tasks["chirps"] = export_table(chirps_table(fc, region, start, end), f"CHIRPS_{tag}", folder)
    if "era5l" in products:
        tasks["era5l"] = export_table(era5l_table(fc, region, start, end), f"ERA5L_{tag}", folder)
    if "smap_l4" in products:
        tasks["smap_l4"] = export_table(smap_l4_table(fc, region, start, end), f"SMAPL4_{tag}", folder)
    if "smap_l3e" in products:
        tasks["smap_l3e"] = export_table(smap_l3e_table(fc, region, start, end), f"SMAPL3E_{tag}", folder)
    if "static" in products:
        t1, t2 = static_table(fc)
        tasks["static"] = export_table(t1.map(lambda f: ee.Feature(None, f.toDictionary())), f"STATIC_{prefix}", folder)
        tasks["worldcover"] = export_table(t2.map(lambda f: ee.Feature(None, f.toDictionary())), f"WORLDCOVER_{prefix}", folder)
    return tasks


# ============================================================================
# v2.1: per-station pull with an on-disk cache (no Drive needed)
# ============================================================================
import argparse
import concurrent.futures as cf
import os
import random
import threading

import numpy as np

S2_PT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]
POINT_PRODUCTS = {
    # name: (collection, bands, scale)    QC / scaling is applied in assemble_v2.py
    "mod13": ("MODIS/061/MOD13Q1", ["NDVI", "EVI", "SummaryQA"], 250),
    "nbar": (C["NBAR"], ["Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band3",
                         "Nadir_Reflectance_Band6", "BRDF_Albedo_Band_Mandatory_Quality_Band1",
                         "BRDF_Albedo_Band_Mandatory_Quality_Band2", "BRDF_Albedo_Band_Mandatory_Quality_Band6"], 500),
    "lai": (C["LAI"], ["Lai", "FparLai_QC"], 500),
    "lstT": (C["LST_TERRA"], ["LST_Day_1km", "QC_Day", "LST_Night_1km", "QC_Night"], 1000),
    "lstA": (C["LST_AQUA"], ["LST_Day_1km", "QC_Day", "LST_Night_1km", "QC_Night"], 1000),
    "chirps": (C["CHIRPS"], ["precipitation"], 5566),
    "era5l": (C["ERA5L"], ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "temperature_2m",
                           "temperature_2m_max", "skin_temperature", "soil_temperature_level_1",
                           "total_precipitation_sum", "total_evaporation_sum", "potential_evaporation_sum",
                           "surface_net_solar_radiation_sum", "snow_cover"], 11132),
    "smapl3e": (C["SMAP_L3E_OLD"], ["soil_moisture_am", "retrieval_qual_flag_am",
                                    "soil_moisture_pm", "retrieval_qual_flag_pm"], 9000),
}
ALL_PRODUCTS = ["static", "s1", "s2", "chirps", "mod13", "nbar", "lai", "lstT", "lstA", "era5l", "smapl3e", "smapl4"]
PAPER_PRODUCTS = ["static", "s1", "s2", "chirps", "mod13"]


def _features_df(fc: ee.FeatureCollection) -> pd.DataFrame:
    return ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})


def _region_df(ic: ee.ImageCollection, pt, scale: float) -> pd.DataFrame:
    v = ic.getRegion(pt, scale).getInfo()
    d = pd.DataFrame(v[1:], columns=v[0])
    d["datetime"] = pd.to_datetime(d["time"], unit="ms")
    return d.drop(columns=["longitude", "latitude", "time"])


def _years(start, end):
    y0, y1 = int(start[:4]), int(end[:4])
    spans = [(f"{y}-01-01" if y > y0 else start, f"{y + 1}-01-01" if y < y1 else end) for y in range(y0, y1 + 1)]
    return [(a, b) for a, b in spans if a < b]


def pull_s1(lon, lat, start, end, buffer_m=50):
    """Linear-power buffer statistics (v2) + the single pixel at the point (paper-style)."""
    pt = ee.Geometry.Point([lon, lat])
    geom = pt.buffer(buffer_m)
    ic = (ee.ImageCollection(C["S1"]).filterDate(start, end).filterBounds(geom)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    red = (ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
           .combine(ee.Reducer.count(), sharedInputs=True))

    def f(img):
        p = _s1_prep(img)
        buf = p.reduceRegion(red, geom, 10)
        one = img.select(["VV", "VH"], ["VV_pt", "VH_pt"]).reduceRegion(ee.Reducer.first(), pt, 10)
        return ee.Feature(None, buf.combine(one)).set({
            "date": img.date().format("YYYY-MM-dd"), "time": img.date().format("HH:mm"),
            "pass": img.get("orbitProperties_pass"), "relorb": img.get("relativeOrbitNumber_start"),
            "platform": img.get("platform_number")})
    return _features_df(ee.FeatureCollection(ic.map(f)))


def pull_s2(lon, lat, start, end, buffer_m=50, max_scene_cloud=70):
    """SCL-masked buffer means (v2) + unmasked point reflectances and scene cloud % (paper-style)."""
    pt = ee.Geometry.Point([lon, lat])
    geom = pt.buffer(buffer_m)
    ic = (ee.ImageCollection(C["S2"]).filterBounds(geom)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud)))

    def f(img):
        m = ee.Image(_s2_prep(img)).reduceRegion(ee.Reducer.mean(), geom, 10)
        one = (img.select(S2_PT_BANDS, [f"pt_{b}" for b in S2_PT_BANDS])
               .reduceRegion(ee.Reducer.first(), pt, 10))
        return ee.Feature(None, m.combine(one)).set({
            "date": img.date().format("YYYY-MM-dd"), "time": img.date().format("HH:mm"),
            "cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"), "tile": img.get("MGRS_TILE")})
    return pd.concat([_features_df(ee.FeatureCollection(ic.filterDate(a, b).map(f))) for a, b in _years(start, end)],
                     ignore_index=True)


def pull_point(product, lon, lat, start, end):
    cid, bands, scale = POINT_PRODUCTS[product]
    pt = ee.Geometry.Point([lon, lat])
    ic = ee.ImageCollection(cid).select(bands)
    return pd.concat([_region_df(ic.filterDate(a, b), pt, scale) for a, b in _years(start, end)],
                     ignore_index=True)


def pull_smapl4(lon, lat, start, end):
    """SMAP L4 3-hourly -> daily mean on the server, one request per year."""
    pt = ee.Geometry.Point([lon, lat])
    l4 = ee.ImageCollection(C["SMAP_L4"]).select(["sm_surface", "sm_rootzone", "sm_surface_wetness",
                                                  "surface_temp"])
    out = []
    for a, b in _years(start, end):
        s = ee.Date(a)
        n = ee.Date(b).difference(s, "day").round()
        days = ee.List.sequence(0, n.subtract(1))
        daily = ee.ImageCollection(days.map(lambda i: l4.filterDate(s.advance(i, "day"),
                                                                    s.advance(ee.Number(i).add(1), "day"))
                                            .mean().set("system:time_start", s.advance(i, "day").millis())))
        out.append(_region_df(daily, pt, 11000))
    return pd.concat(out, ignore_index=True)


def pull_static(lon, lat, buffer_m=50):
    pt = ee.Geometry.Point([lon, lat])
    srtm = ee.Image(C["SRTM"])
    img = ee.Image.cat([
        srtm.rename("elevation"), ee.Terrain.slope(srtm).rename("slope"), ee.Terrain.aspect(srtm).rename("aspect"),
        ee.Image(C["MERIT"]).select(["upa", "hnd"]),
        *[ee.Image(C[k]).select(f"{k.lower()}_0-5cm_mean").rename(k.lower())
          for k in ("SAND", "CLAY", "SILT", "BDOD", "SOC")],
    ])
    d = img.reduceRegion(ee.Reducer.first(), pt, 30).getInfo()
    wc = (ee.Image(ee.ImageCollection(C["WORLDCOVER"]).first()).select("Map")
          .reduceRegion(ee.Reducer.mode(), pt.buffer(buffer_m), 10).getInfo())
    d["worldcover"] = wc.get("Map")
    # topographic wetness index ln(a / tan b): a = upstream area per unit contour width (90 m cells)
    if d.get("upa") is not None and d.get("slope") is not None:
        a = d["upa"] * 1e6 / 90
        d["TWI"] = float(np.log(a / max(np.tan(np.radians(d["slope"])), 1e-3)))
    return pd.DataFrame([d])


def pull(product, lon, lat, start, end, buffer_m=50):
    if product == "s1":
        return pull_s1(lon, lat, start, end, buffer_m)
    if product == "s2":
        return pull_s2(lon, lat, start, end, buffer_m)
    if product == "smapl4":
        return pull_smapl4(lon, lat, start, end)
    if product == "static":
        return pull_static(lon, lat, buffer_m)
    return pull_point(product, lon, lat, start, end)


_log_lock = threading.Lock()


def _log(path, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with _log_lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _one(cache, sid, product, lon, lat, start, end, buffer_m, log, retries=5):
    out = os.path.join(cache, product, f"{sid}.parquet")
    if os.path.exists(out):
        return "cached"
    for k in range(retries):
        try:
            t = time.time()
            df = pull(product, lon, lat, start, end, buffer_m)
            df["sid"] = str(sid)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df.to_parquet(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            return f"ok {len(df)} rows {time.time() - t:.0f}s"
        except Exception as e:  # rate limits / transient server errors -> back off and retry
            _log(log, f"retry {k + 1}/{retries} {sid} {product}: {str(e)[:200]}")
            time.sleep(min(300, 10 * 2 ** k) + random.random() * 5)
    return "FAILED"


def run_cached(st: pd.DataFrame, cache: str, start: str, end: str, products=None, buffer_m=50, workers=8):
    """One parquet per station x product under cache/<product>/<sid>.parquet. Re-running skips
    what is already on disk, so an interrupted run resumes where it stopped."""
    os.makedirs(cache, exist_ok=True)
    log = os.path.join(cache, "extract.log")
    jobs = []
    for r in st.itertuples():
        prods = products or (PAPER_PRODUCTS if str(getattr(r, "set", "")) == "paper_coords" else ALL_PRODUCTS)
        jobs += [(str(r.sid), p, float(r.lon), float(r.lat)) for p in prods]
    todo = [j for j in jobs if not os.path.exists(os.path.join(cache, j[1], f"{j[0]}.parquet"))]
    _log(log, f"{len(jobs)} jobs, {len(todo)} to do, {workers} workers")
    done, failed, t0 = 0, [], time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_one, cache, sid, p, lon, lat, start, end, buffer_m, log): (sid, p)
                for sid, p, lon, lat in todo}
        for fu in cf.as_completed(futs):
            sid, p = futs[fu]
            res = fu.result()
            done += 1
            if res == "FAILED":
                failed.append((sid, p))
            if done % 25 == 0 or res == "FAILED":
                el = time.time() - t0
                _log(log, f"[{done}/{len(todo)}] {sid} {p} {res} | elapsed {el / 60:.0f} min, "
                          f"ETA {el / done * (len(todo) - done) / 60:.0f} min")
    _log(log, f"finished: {done - len(failed)} ok, {len(failed)} failed {failed[:20]}")
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True, help="csv with sid, lat, lon[, set]")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-01-01", help="exclusive")
    ap.add_argument("--products", default=None, help="comma list; default depends on station set")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="soil-moisture-iiita")
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    stations = pd.read_csv(a.stations, dtype={"sid": str})
    run_cached(stations, a.cache, a.start, a.end, a.products.split(",") if a.products else None,
               workers=a.workers)
