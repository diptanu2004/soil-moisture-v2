# Results summary — 2019–2023 re-run (results_v2)

Every number below comes from a file in `results_v2/tables/`; the full set is in
`tables/v2_summary_before_after.md`. RMSE in % volumetric soil moisture at 5 cm.
Run date: 21 Oct 2026. Data: 192 CONUS SCAN stations (186 with 5 cm data), Sentinel-1/2, MODIS,
CHIRPS, ERA5-Land, SMAP, SoilGrids, SRTM, MERIT, WorldCover via Earth Engine.

## Headline: before → after (the fair comparison)
Fair here means: the same rows, satellite-only, at stations, regions or years never used in
training. Each cell is the original paper model → the proposed hybrid (TabPFN v2 + TempoSAR + trees).

| Test | R² | RMSE | median station r |
|---|---|---|---|
| Unseen stations (184 stations) | 0.303 → **0.647** | 10.23 → **7.28** | 0.578 → **0.810** |
| Unseen climate region | 0.153 → **0.643** | 11.27 → **7.32** | 0.503 → **0.807** |
| Unseen years 2022–23 | 0.636 → **0.792** | 7.28 → **5.50** | 0.579 → **0.831** |
| Paper's own 16 stations, unseen (TabPFN v2) | 0.193 → **0.570** | 10.10 → **7.38** | 0.536 → **0.800** |

## Update 22 Nov 2026: v3/v4 improvements (details in `V3_RESULTS.md`)
All rows below come from `tables/v3_models.csv` and `tables/v4_daily.csv`. The v4 final model was declared in
`progress.md` before its results: `Hybrid(TabPFN-bag5 + TempoSAR+POLARIS + Trees+POLARIS(exact, Se))`.
ΔR² is paired against the v2 hybrid (station-cluster bootstrap, 1000 resamples).

| Test | Paper model | v2 hybrid | v4 final | ΔR² v4 − v2 [95% CI] |
|---|---|---|---|---|
| Unseen stations: R² / RMSE / ubRMSE | 0.303 / 10.23 / 10.18 | 0.647 / 7.28 / 7.25 | **0.671 / 7.03 / 7.00** | +0.024 [+0.005, +0.045] |
| Unseen stations: bias / median station r | +1.03 / 0.578 | +0.63 / 0.810 | +0.62 / 0.803 | |
| Unseen climate region: R² / RMSE | 0.153 / 11.27 | 0.643 / 7.32 | **0.669 / 7.05** | +0.026 [+0.008, +0.045] |
| Unseen years 2022–23: R² / RMSE (bagging only ¹) | 0.636 / 7.28 | 0.792 / 5.50 | 0.796 / 5.46 | +0.003 [+0.001, +0.005] |
| Daily, all days, unseen stations (interpolated ²) | | 0.625 / 7.97 | **0.654 / 7.67** | +0.028 [+0.008, +0.050] |
| Daily, Sentinel-1 days only | | 0.642 / 7.36 | **0.666 / 7.11** | +0.024 [+0.003, +0.044] |
| Daily, non-Sentinel-1 days only | | 0.621 / 8.07 | **0.650 / 7.76** | +0.029 [+0.008, +0.051] |
| Daily, all days, unseen regions (interpolated ²) | | 0.618 / 8.05 | **0.646 / 7.75** | +0.028 [+0.008, +0.049] |

¹ POLARIS rows are not built for the temporal test (stations are seen in training); the row is
`Hybrid(TabPFN-bag5+TempoSAR+Trees)`. ² Sentinel-1-day estimates linearly interpolated onto 246,555 common
labelled days (34,504 Sentinel-1 days, 212,051 other days); interpolation uses the next Sentinel-1 date, so it
is a reconstruction. Hold-last (real-time) scores 0.627 on all days.

Experiment by experiment (unseen stations / unseen regions, R²):

| Exp. | Change | Result | Status |
|---|---|---|---|
| diag. | Error decomposition of the v2 hybrid | 58 % of MSE is a constant per-station offset | — |
| v3-1 | TabPFN bag of 5 contexts | TabPFN 0.614 → 0.629; hybrid 0.651 / 0.648 (+0.003 / +0.005) | pre-registered |
| v3-2 | Station-mean correction from v2 features | 0.648 / 0.630 (+0.001 / −0.014) — no gain | pre-registered (declared v3 candidate) |
| v3-3 | POLARIS 30 m soil properties in the trees | Trees 0.636 → 0.669; hybrid 0.669 / 0.663 (+0.022 / +0.021) | exploratory |
| v3-3 | POLARIS in TabPFN | 0.594 — worse | exploratory |
| v4-1 | POLARIS at exact NRCS coordinates | 0.670 / 0.664 (vs v3 best +0.001 / +0.000) | pre-registered |
| v4-2 | Relative-wetness (Se) target for the trees | 0.669 / 0.663 (−0.000 / −0.001) | pre-registered |
| v4-3 | TempoSAR with POLARIS static inputs | TempoSAR 0.623 → 0.641; hybrid 0.674 / 0.668 | pre-registered |
| v4 | **Declared final hybrid** | **0.671 / 0.669** | pre-registered |
| v4 | Best row found (exact POLARIS, raw target) | 0.677 / 0.670 — picked after results, not the headline | exploratory |
| v4 daily | Final hybrid + LGBM-daily, equal mean | 0.665 all days, 0.663 non-S1 days | exploratory |

The paired CI half-width is about ±0.01 R², so the v3-1, v4-1, v4-2 and v4-3 steps are each below what can be
resolved on their own. The gain over v2 (+0.024 to +0.029) is resolved and comes almost entirely from POLARIS.
Median station r does not change: the gain fixes each station's level, not its day-to-day dynamics.

Further improvements (v2):

* **Nowcasting** with yesterday's in-situ value. The paper model with its lag scores R² 0.873
  (RMSE 4.36); TabPFN v2 with a consistent lag scores **0.983 (RMSE 1.59)** at unseen stations. It
  beats persistence (RMSE 1.90) under all three protocols, with 15–18 % lower error. The paper
  model never beats persistence at an unseen station.
* **Few-shot adaptation** to a new station:
  * with 10 in-situ samples, TabPFN in-context goes from R² 0.641 to **0.774** (no retraining);
  * with 20 samples it reaches **0.798**.
* **Daily series** at unseen stations: the paper's recursive loop scores R² −0.345 on its 16
  stations. Across all stations, satellite-only LightGBM scores **0.628** and the DailyLSTM
  pretrained on SMAP **0.567**.
* **Against operational products:** raw SMAP L4 scores R² 0.366 (0.438 calibrated); the hybrid
  scores **0.647**.

## What happened to the paper's own numbers
* **Headline protocol.** The paper's protocol (random 80/20, 15 stations) gives R² 0.780 on
  2019–2023 data. Persistence gives 0.744 on the rows that have a lag. At the validation station
  Shagbark Hills the model scores R² 0.269.
* **Fix ladder** (16 stations, unseen-station satellite-only R²): −0.389 (as published) → −0.028
  with corrected coordinates → 0.032 after all bug fixes. The fixes are necessary but not
  sufficient: the 16-feature design itself has almost no skill at a new station.
* **Lag mismatch.** Making the lag consistent turns a model that loses to persistence (0.691 vs
  0.744) into one that is close to it (0.911 vs 0.957). Adding the improved features and TabPFN
  then beats persistence (see above).
* **Daily loop.** Implementing the drift control that the paper describes improves the daily loop
  from R² −0.345 to −0.138, which is still unusable with the original features.

## Honest caveats (put these in the paper)
1. **Sentinel-1 adds little.** Most of the satellite-only gain comes from adding SMAP and
   ERA5-Land, plus better processing. Removing Sentinel-1 changes R² only from 0.636 to 0.633
   (unseen stations). Sentinel-1 + static features alone reach 0.552, still far above the paper's
   0.303.
2. **TabPFN v2 alone matches but does not beat the trees** in satellite-only mode (0.614 vs 0.636
   at unseen stations). It is the best single model on the paper's 16 stations (0.570), the best
   nowcasting model with the lag, and the best few-shot adapter without training. The hybrid that
   includes it is the best overall.
3. **DailyLSTM is behind LightGBM.** The DailyLSTM (the requested daily model) is behind LightGBM
   for daily reconstruction (0.567 vs 0.628). It was not tuned.
4. **Not compared like-for-like with the paper's headline.** Do **not** set the paper's R² 0.78 next
   to our 0.65: different protocol, and the 0.78 includes an in-situ input. Under the paper's own
   protocol (random split, with lag, all stations) the improved model also wins: 0.986 vs 0.892.
5. **v3/v4 provenance.** POLARIS entered as exploratory after the pre-registered station-mean correction
   failed. The v4 rows were then pre-registered with POLARIS in them. Say so in the paper. Station coordinates
   in `stations.csv` are > 1 km off at 21 of 184 stations (Starkville 17.7 km, Lovelock 27.5 km); the satellite
   features were not re-extracted at the exact coordinates (needs Earth Engine).
6. **Data gaps.** USCRN was not used (NCEI server down, HTTP 503), so the analysis is SCAN-only.
   The 2022–23 test years coincide with the loss of Sentinel-1B.

Leakage audit (`tables/v2_leakage_audit.txt`):

* no in-situ variable is in any satellite-only feature set;
* the folds are disjoint;
* the strongest single predictor correlates at r = 0.73;
* the gap between the random split and the station split (0.897 vs 0.636) is the leakage the paper's
  protocol allows.

## Files
| What | Where |
|---|---|
| Draft Methods / Results / Limitations | `results_v2/paper_text_draft.md` |
| Citations (TabPFN + all datasets) | `results_v2/citations.md` |
| Before/after table (all blocks) | `results_v2/tables/v2_summary_before_after.md` / `.csv` |
| Paper model, paper protocol; fix ladder; daily loop | `tables/v2_1a_*.csv`, `v2_1b_fix_ladder.csv`, `v2_1c_daily_loop.csv` |
| All feature sets × 4 protocols | `tables/v2_2_protocols.csv` |
| Trees / TempoSAR / TabPFN / hybrids | `tables/v2_3_models.csv` |
| Few-shot | `tables/v2_4_fewshot.csv` |
| Daily reconstruction | `tables/v2_5_daily.csv` |
| Ablation + raw SMAP/ERA5 | `tables/v2_6_ablation.csv` |
| Paper's 16 stations only | `tables/v2_7_paper16_subset.csv` |
| Dataset counts | `tables/v2_0_dataset.csv` |
| Figures 1–11 | `results_v2/figures/fig_v2_*.png` |
| Out-of-fold predictions | `tables/*.parquet` |
| v3/v4 all rows (generated) | `results_v2/V3_RESULTS.md` |
| v3/v4 models, bootstrap CIs | `tables/v3_models.csv`, `tables/v3_oof_*.parquet` |
| v4 daily, Sentinel-1 vs other days | `tables/v4_daily.csv`, `tables/v4_daily_preds.parquet` |
| POLARIS soil tables, exact coordinates | `data/raw_v2/polaris_stations[_exact].parquet`, `data/raw_v2/stations_exact.csv` |
| Kaggle GPU logs | `results_v2/logs/kaggle_*/` |
| Progress log | `results_v2/progress.md` |

## Reproduce
```
.venv\Scripts\activate
cd soil_moisture_improvements
python gee/gee_extract_v2.py --stations data/raw_v2/stations_v2.csv --cache data/raw_v2/ee_cache   # ~1 h, resumable
python gee/insitu_v2.py scan data/raw_v2/stations_v2.csv data/raw_v2/scan_hourly                    # ~1 h, resumable
bash src_v2/run_v2.sh        # ~5 h on i7-13620H + RTX 4050 (all experiments, audit, summary, figures)
```
