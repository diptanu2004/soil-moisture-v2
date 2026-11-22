# CLAUDE.md — soil_moisture_improvements

Context for Claude Code sessions in this folder. Read this first.

## What this project is
Improvement experiments for the paper *"Multi-Sensor Machine Learning Framework for Daily Station-Scale
Soil Moisture Reconstruction at Selected SCAN Sites"*. The v2 pipeline re-runs everything on 2019–2023 data
(192 CONUS SCAN stations, 186 with 5 cm data) with Sentinel-1/2, MODIS, CHIRPS, ERA5-Land, SMAP, SoilGrids,
SRTM, MERIT and WorldCover pulled through Earth Engine. Target: `sm_5cm` (% volumetric soil moisture).

## Layout
- `src/` — original 2015–2018 package (`common.py`, `deep.py`, `exp1..4`, `make_figures.py`).
- `src_v2/` — the 2019–2023 pipeline. Entry point `run_v2.sh`. Shared code in `common_v2.py`
  (feature sets, splits, tree models, TabPFN wrapper, `log_progress`). `deep_v2.py` wraps TempoSAR from `src/deep.py`.
  - `exp_v2_1_before` paper model + fix ladder + daily loop
  - `exp_v2_2_protocols` all features x 4 protocols (writes the `v2_2_oof_*.parquet` files)
  - `exp_v2_3_models` trees / TempoSAR / TabPFN v2 / hybrids (writes `v2_3_oof_*.parquet`)
  - `exp_v2_4_fewshot`, `exp_v2_5_daily`, `exp_v2_6_ablation`, `leakage_audit`, `summary_v2`, `make_figures_v2`
- `gee/` — Earth Engine extraction, in-situ loaders, assembly. `data/raw_v2/` is cached and resumable.
- `data/processed_v2/` — `anchor.parquet`, `daily.parquet`, `paper_anchor.parquet`.
- `results_v2/` — `tables/`, `figures/`, `RESULTS_SUMMARY.md`, `progress.md`, `paper_text_draft.md`.

## Environment
- Windows. Activate with `.venv\Scripts\activate`, run from the `soil_moisture_improvements` folder.
- Hardware: the v2 run used an i7-13620H (10C/16T), 15.7 GB RAM and one RTX 4050 6 GB. **Only one GPU job at a time**
  on that machine (TabPFN and TempoSAR do not fit together). CPU-only work (pandas, trees) can run alongside a GPU job.
- The current copy (since 2026-11-22) runs on a Ryzen 5 5600H (6C/12T), 7.4 GB RAM and **no NVIDIA GPU**.
  The `.venv` is Python 3.12 and CPU-only (no torch/TabPFN). Trees take about 1.4 min per fold here.
  **Run at most one tree script at a time**: two or three in parallel run out of the 7.4 GB RAM (ArrayMemoryError).
- GPU work runs on **Kaggle (GPU T4 x2)**. Use one process per GPU with `CUDA_VISIBLE_DEVICES`, and write per-fold output files.
  Token: `kaggle.json` holds `api-token: KGAT_...`. Strip the label and pass it as `KAGGLE_API_TOKEN`. Never upload it.
  Upload bundles (code + the parquet files needed) as private datasets. Kernel runners are in the scratchpad. Logs only
  appear after a run ends.
- Set `SM_PROC` / `SM_RES` to override the data and results folders (see `common_v2.py`).
- Use `log_progress(...)` for long runs; it appends to `results_v2/progress.md`.

## Ground rules (do not break these)
1. **Satellite-only means no in-situ input.** Never put `sm_5cm`, `SM_lag1d`, `SM_lag_prevS1`, `stemp_5cm_C`
   or anything derived from in-situ data in a satellite-only feature set. `leakage_audit.py` must stay clean.
2. **The headline protocol is leave-stations-out, and the headline model is the equal-weight
   `Hybrid(TabPFN+TempoSAR+Trees)`.** Both were declared before seeing results. Do not pick or rename the
   headline based on test scores. New models are extra rows in the tables, never a silent replacement.
3. **Fit nothing on test data.** Blend weights, calibrations and bias corrections must be fitted inside the
   training folds (nested, grouped by station). Test stations/regions/years never appear in training.
4. **Report all models, including the ones that lose.** Do not set the paper's R² 0.78 beside the v2 R² 0.65:
   different protocol, and the paper's number includes an in-situ input.
5. Do not edit the files in `results_v2/tables/` by hand. Regenerate them from the scripts.
6. Never run the full pipeline (about 5 h) to test a change. Use a single fold or the `--no-deep` flag first.
7. Check `site_wet_frac` and the `VV_site_*` / `site_mean_*` features: confirm they are satellite-derived and,
   for the temporal protocol, not computed from the 2022–23 test dates.

## Baseline to beat (leave-stations-out, satellite-only, 184 stations)
Hybrid: R² 0.647, RMSE 7.28, median station r 0.810. Trees alone: 0.636. TabPFN alone: 0.614.
Raw SMAP L4: 0.366. Removing Sentinel-1 changes R² only from 0.636 to 0.633.
Always report R², RMSE, ubRMSE, bias and median per-station r (use `summarize` from `common.py`).
Current best declared model (v4 final, see below): LOSO R² 0.671, RMSE 7.03; LORO 0.669. It is an extra row, and the v2
headline above stays the headline the paper declared.

## v3 status (2026-11-22). The pre-registration is in `progress.md`. Numbers come from `tables/v3_models.csv`.
- Diagnosis: about 58 % of the hybrid's MSE at unseen stations is a constant per-station offset. Members' errors are 0.90–0.93 correlated.
  Quantile-mapping to SMAP was dropped: SMAP station means are worse than the hybrid's (RMSE 8.4 vs 5.9).
- v3-1 TabPFN context bagging (`exp_v3_tabpfn_bag.py`, Kaggle): TabPFN alone +0.015 to +0.023 R², but the hybrid gains only
  +0.003 (LOSO, n.s.) / +0.005 (LORO). The bag-0 re-run reproduces v2 TabPFN, so TabPFN 10.2.0 is fine.
- v3-2 station-mean correction (`exp_v3_station_mean.py`) with the v2 features: **no gain** (LOSO +0.001, LORO −0.014).
- v3-3 **exploratory** POLARIS 30 m soil data (`gee/polaris_extract.py`, HTTP, no Earth Engine). This is the first real gain:
  Hybrid with Trees+POLARIS is LOSO 0.666 (+0.019, CI +0.006..+0.034) and LORO 0.659 (+0.016, CI +0.007..+0.026).
  The best row is Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS): LOSO 0.669 (+0.022), LORO 0.663 (+0.021).
  TabPFN+POLARIS is *worse* (0.594). Not pre-registered; label it exploratory in the paper.
- v4 (pre-registered; reference = Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS), LOSO 0.669):
  exact NRCS coordinates for POLARIS (`data/raw_v2/stations_exact.csv`) and the relative-wetness (Se) target are both
  **null** (|ΔR²| <= 0.001 for the hybrid; exact+Se -0.005). v4-3 TempoSAR+POLARIS (`exp_v4_temposar.py`, Kaggle): alone
  0.641 vs control 0.623 (LOSO). The declared v4 candidate is LOSO 0.671 / LORO 0.669 (+0.024 / +0.026 vs v2 headline, CIs exclude 0;
  +0.002 / +0.005 vs v4 ref, n.s.).
- v4 daily (`exp_v4_daily.py`, `tables/v4_daily.csv`): the v4 final hybrid's Sentinel-1-day estimates, interpolated onto every
  day, score LOSO 0.666 on S1 days, 0.650 on other days and 0.654 on all days. The v2 hybrid scores 0.642 / 0.621 / 0.625.
  Hold-last (real-time) gives 0.627 on all days. Mean with LGBM-daily gives 0.665 (exploratory).
- Docs updated 2026-11-23: README (before/after table + Mermaid architecture diagram), RESULTS_SUMMARY (v3/v4 section),
  paper_text_draft (§2.7, §3.7, limitations), citations (POLARIS). V3_RESULTS.md is generated by `summary_v3.py`.
  `.gitignore` excludes kaggle.json, .venv, raw caches, test runs, checkpoints and the Colab bundles.
  stations.csv coordinates are off by > 1 km at 21/184 eval stations (Starkville 17.7 km, Lovelock 27.5 km).
- Noise floor: the paired station-bootstrap ΔR² CI is about ±0.01. Gains below about 0.01 are not resolvable.
- Not done yet: spatially blocked folds, rolling-origin split, nested stacking, re-extracting the satellite
  features at the exact NRCS coordinates (needs Earth Engine), and the daily LGBM with POLARIS.

## Original work plan (3 fixes; estimates were guesses, see v3 status above for what happened)
Run these in parallel where possible. Only one touches the GPU.
1. **TabPFN context bagging (GPU, about 75 min).** In `exp_v2_3_models.py`, average 5 station-stratified
   10k-row contexts per fold (different `seed` in `context_subsample`) instead of one. Write per-fold results
   so a crash does not lose the run. Guess: TabPFN alone +0.01–0.02 R².
2. **Per-station bias/scale correction (CPU).** First, on the existing OOF parquet files, split error into
   bias and ubRMSE per station. If bias is a small share, stop. Otherwise try quantile-mapping each
   unseen station's predictions to its own SMAP L4 series, and a second-stage model from static features.
   Both stay satellite-only. Guess: +0.02–0.05 R², uncertain; it may do nothing.
3. **Grouped-nested stacking (CPU, after 1 and 2).** Blend Trees / TempoSAR / TabPFN (and corrected
   predictions if #2 helps) with ridge or a small LGBM. Add as an extra row next to the equal-weight hybrid.
   Guess: +0.005–0.015 R².
Also, independent of the metrics: station-level **cluster bootstrap CIs** on the existing OOF files, and
spatially blocked folds (for example 100 km clusters). Blocked folds need the models rerun, so schedule them separately.

## Known issues and caveats
- USCRN was not used (NCEI returned HTTP 503). Retry it, or use ISMN, for an out-of-network test.
- 2022–23 test years coincide with the loss of Sentinel-1B. A rolling-origin split would separate that effect.
- DailyLSTM is untuned and behind LightGBM (0.567 vs 0.628).
- Optical features already exist (S2 indices, MODIS NBAR/LAI/LST). Check the "without optical / LST" row of
  `v2_6_ablation.csv` before adding more. Candidates: OPTRAM, TVDI, HLS.
- `gee/gee_extract_v2.py` and the earlier `assemble_v2.py` were fixed during the run. Keep the NaN-angle guard.
