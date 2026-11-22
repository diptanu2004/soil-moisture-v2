"""Generates the three notebooks shipped with the package."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

# ---------------------------------------------------------------------------
# 01 - experiments on the public multi-region dataset (executed)
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# 01 - Improvement experiments (multi-region SCAN + Sentinel-1, 2015-2018)

This notebook reproduces every number in the improvement report. It reads the processed data in
`../data/processed/` and the result tables in `../results/tables/`.

* Set `RERUN = True` to recompute everything from the raw data. That takes about 3-4 h on 2 CPU cores.
  The deep-learning and transfer-learning parts take most of that time.
* With `RERUN = False` it only loads the saved tables and draws the figures. That takes about 1 minute.

**Data.** Public, real data from github.com/kashingtonDC/RZSM:
* SCAN hourly soil moisture, downloaded from the NRCS Report Generator, for about 230 sites
* Sentinel-1 VV/VH, Landsat-8, MODIS LAI and PRISM rain extracted with Earth Engine at each SCAN site

The period is 2015-01 to 2019-01. 84 CONUS sites have at least 20 Sentinel-1 dates with in-situ data,
and 188 sites have daily 5 cm data."""),
    code("""import sys, os
sys.path.insert(0, '../src')
import pandas as pd
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
RERUN = False
TAB = '../results/tables'"""),
    md("## 0. Build the datasets (raw -> processed)"),
    code("""if RERUN:
    import build_dataset; build_dataset.main()
from common import load_anchor
a = load_anchor()
print(a.shape, a.site.nunique(), 'sites')
a.groupby('region').agg(sites=('site','nunique'), samples=('site','size'), mean_sm=('sm_5cm','mean')).round(2)"""),
    md("""## 1. Paper protocol vs. honest evaluation (exp1)
The paper-analogue features use the paper's 16-feature design. Landsat-8 indices stand in for Sentinel-2, MODIS LAI stands in for MODIS NDVI/EVI, and slope is not available.
The SAR indices are computed on dB values, as the repo does. Persistence = the lag value itself."""),
    code("""if RERUN:
    import exp1_baselines; exp1_baselines.main()
e1 = pd.read_csv(f'{TAB}/exp1_baselines.csv')
e1[e1.model.isin(['Ensemble','Persistence(prevS1)','Persistence(1d)'])][['protocol','features','model','n','R2','RMSE','ubRMSE','site_median_r','site_median_ubRMSE']].round(3)"""),
    md("## 2A. Stronger models: regularised trees vs TempoSAR (Transformer over the S1 history) vs hybrid"),
    code("""if RERUN:
    import exp2_transfer as e2
    lab, full = e2.load_for_deep(); e2.part_a(lab, full)
pd.read_csv(f'{TAB}/exp2a_loso_models.csv')[['model','n','R2','RMSE','ubRMSE','bias','r','site_median_r','site_median_ubRMSE']].round(3)"""),
    md("## 2B. Transfer learning to a new climate region (pretrain on other regions, fine-tune on 1 / 3 / all target stations)"),
    code("""if RERUN:
    e2.part_b(lab, full)
t = pd.read_csv(f'{TAB}/exp2b_transfer.csv')
t[t.region=='ALL'][['n_ft_sites','method','R2','RMSE','ubRMSE','site_median_r','site_median_ubRMSE']].round(3)"""),
    code("""t.pivot_table(index=['region','method'], columns='n_ft_sites', values='RMSE').round(2)"""),
    md("## 2C. Few-shot adaptation to a new station (first k in-situ samples)"),
    code("""if RERUN:
    import exp2c_calibration; exp2c_calibration.main()
pd.read_csv(f'{TAB}/exp2c_calibration.csv')[['k','model','correction','n','R2','RMSE','ubRMSE','bias','site_median_r']].round(3)"""),
    md("## 3. More stations and more regions"),
    code("""if RERUN:
    import exp3_stations_regions as e3
    df = load_anchor().dropna(subset=['SM_lag1d']).reset_index(drop=True)
    e3.part_b(df); e3.part_c(df); e3.part_a(df)
print(pd.read_csv(f'{TAB}/exp3b_more_regions.csv')[['training','test_sites','n','R2','RMSE','site_median_r','site_median_ubRMSE']].round(3))
lc = pd.read_csv(f'{TAB}/exp3a_learning_curve_raw.csv')
lc.groupby(['features','n_train_sites'], sort=False)[['R2','RMSE','site_median_r']].mean().round(3)"""),
    code("""cr = pd.read_csv(f'{TAB}/exp3c_cross_region.csv')
cr.pivot(index='train_region', columns='test_region', values='site_median_r').round(2)"""),
    md("## 4. Daily reconstruction (gap-fill): persistence, lag mismatch, satellite-only"),
    code("""if RERUN:
    import exp4_daily; exp4_daily.main()
for f in ['exp4_g1_persistence','exp4_g1_paper_stations','exp4_g2_persistence_horizon','exp4_g3_lag_mismatch','exp4_g4_satellite_only_daily']:
    print(f); display(pd.read_csv(f'{TAB}/{f}.csv').round(3))"""),
    md("## Figures"),
    code("""import make_figures as mf
for f in [mf.fig_coordinates, mf.fig_coverage, mf.fig_exp1, mf.fig_exp2a, mf.fig_exp2b, mf.fig_exp2c, mf.fig_exp3, mf.fig_exp4]:
    f()
from IPython.display import Image, display
import glob
for p in sorted(glob.glob('../results/figures/*.png')):
    display(Image(p, width=900))"""),
]
nbf.write(nb, "01_improvement_experiments.ipynb")

# ---------------------------------------------------------------------------
# 02 - GEE extraction v2 (Colab)
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# 02 - Upgraded Earth Engine extraction (v2): more stations, more regions, daily drivers

Run this in **Google Colab** with your Earth Engine project. Upload the `gee/` folder, or clone your repo.

It was **not executed** where it was written, because there was no Earth Engine access there. The collection IDs
and band names were checked against the public catalogue. The assembly step (`assemble_v2.py`) was tested on
synthetic exports with the same column layout. Start with the 16 paper stations and 1 year, check the CSVs,
then scale up.

**What it fixes or adds versus v1**
1. Station coordinates now come from NRCS/ISMN metadata. In v1, all 16 stations were 2-360 km off.
2. Features are averaged over a 50 m buffer instead of a single 10 m pixel.
3. S1 is read in linear power. Ratio, RVI and SPAN are computed correctly. Incidence angle and orbit are kept for change detection.
4. S2 uses the SCL pixel cloud mask. MODIS uses BRDF-corrected NBAR with QA, plus LAI and day/night LST.
5. Daily drivers are added: CHIRPS, ERA5-Land, and SMAP L4 / L3-enhanced (the passive-microwave prior in your project plan).
6. Static descriptors are added: SoilGrids, SRTM, MERIT-Hydro TWI and ESA WorldCover.
7. Collection-level `reduceRegions` exports scale to hundreds of stations."""),
    code("""!pip -q install earthengine-api ismn pandas pyarrow
import sys; sys.path.insert(0, 'gee')   # folder with gee_extract_v2.py, insitu_v2.py, assemble_v2.py
import pandas as pd
import gee_extract_v2 as G
G.init('YOUR-GEE-PROJECT')"""),
    md("""## 1. Stations
* `stations_paper16_corrected.csv`: your 16 stations with the NRCS coordinates.
* `stations_scan_conus_candidates.csv`: all 192 CONUS SCAN sites, with a climate region for each.
* USCRN: about 115 stations. They have 5 cm soil moisture plus precipitation and temperature measured at the same site.
* ISMN: download networks from ismn.earth. Examples are REMEDHUS (ES), SMOSMANIA (FR), TERENO (DE), OZNET (AU), MAQU / CTP_SMTMN (CN) and SOILSCAPE (US). Then use `insitu_v2.ismn_daily()`."""),
    code("""st16  = G.load_stations('gee/stations_paper16_corrected.csv')
scan  = pd.read_csv('gee/stations_scan_conus_candidates.csv').rename(columns={'scan_id':'sid'})
print(scan.region.value_counts())
# e.g. all SCAN stations that started before 2019:
scan_sel = scan[pd.to_datetime(scan.start, format='%Y-%B', errors='coerce') < '2019-01-01']
stations = pd.concat([st16[['sid','lat','lon']], scan_sel[['sid','lat','lon']]]).drop_duplicates('sid')
len(stations)"""),
    md("## 2. Start the exports (one Drive task per product). Run one call per year for many stations."),
    code("""tasks = {}
for y in range(2019, 2024):
    tasks[y] = G.export_all(stations, f'{y}-01-01', f'{y+1}-01-01', folder='sm_v2', prefix=f'y{y}_',
                            products=['s1','s2','nbar','lai','lst','chirps','era5l','smap_l4','smap_l3e'])
G.export_all(stations, '2019-01-01', '2019-01-02', folder='sm_v2', products=['static'])"""),
    md("""## 3. In-situ data
SCAN uses the Report Generator CSV service, USCRN uses the daily01 files, and ISMN uses an archive folder."""),
    code("""import insitu_v2 as I
frames = []
for r in pd.read_csv('gee/stations_scan_conus_candidates.csv').itertuples():
    try:
        frames.append(I.scan_daily(r.station_triplet, '2019-01-01', '2023-12-31'))
    except Exception as e:
        print(r.station_triplet, e)
insitu = pd.concat(frames)
insitu.to_parquet('insitu_daily.parquet')"""),
    md("## 4. Assemble the model tables (after the Drive exports finish, mount Drive)"),
    code("""from google.colab import drive; drive.mount('/content/drive')
!python gee/assemble_v2.py --exports /content/drive/MyDrive/sm_v2 --insitu insitu_daily.parquet \\
        --stations gee/stations_scan_conus_candidates.csv --start 2019-01-01 --end 2023-12-31 --out data/processed_v2"""),
]
nbf.write(nb, "02_gee_extraction_v2_colab.ipynb")

# ---------------------------------------------------------------------------
# 03 - run the improved models on your regenerated data
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# 03 - Train and evaluate the improved models on your regenerated (v2) data

The tables from notebook 02 use the same column names as the experiment dataset. Point `PROC` at them and
the experiment scripts run unchanged. The extra v2 drivers (ERA5-Land, SMAP, API, LST, ...) are added to the
feature list automatically.

**Recommended reporting for the revised paper**
* Leave-stations-out and leave-one-region-out evaluation, not only random 80/20.
* Pooled R², RMSE and ubRMSE, plus median per-station r and ubRMSE.
* A persistence baseline whenever an in-situ lag is used.
* A satellite-only model (no in-situ input) as the main product."""),
    code("""import sys; sys.path.insert(0, '../src'); sys.path.insert(0, '../gee')
import common
common.PROC = '../data/processed_v2'          # <- tables from assemble_v2.py
from assemble_v2 import V2_EXTRA_FEATS
import pandas as pd
a = pd.read_parquet(f'{common.PROC}/anchor.parquet')
extra = [c for c in V2_EXTRA_FEATS if c in a and a[c].notna().mean() > 0.5]
common.IMPROVED_SAT[:] = common.IMPROVED_SAT + [c for c in extra if c not in common.IMPROVED_SAT]
common.IMPROVED_LAG[:] = common.IMPROVED_SAT + ['SM_lag1d']
print(len(common.IMPROVED_SAT), 'satellite-only features')"""),
    md("## Anchor (Sentinel-1 date) models: protocols, persistence, trees vs TempoSAR vs hybrid, transfer learning"),
    code("""import exp1_baselines, exp2_transfer as e2
exp1_baselines.main()
lab, full = e2.load_for_deep()
e2.part_a(lab, full)
e2.part_b(lab, full, regions=[r for r, n in lab.groupby('region').site.nunique().items() if n >= 5])"""),
    md("""## Daily (gap-fill) models with daily drivers
LGBM and a DailyLSTM are compared. The DailyLSTM is pretrained on ERA5-Land / SMAP soil moisture and then fine-tuned on in-situ data."""),
    code("""!cd ../src && python daily_v2.py --daily ../data/processed_v2/daily.parquet --out ../results/v2 --epochs 15"""),
]
nbf.write(nb, "03_train_on_v2_data.ipynb")
print("notebooks written")
