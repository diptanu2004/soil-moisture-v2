# Draft text for the revised paper

Numbers in this draft are copied from `results_v2/tables/` (file named in brackets). Before
submission, re-check every number against the table it cites. Items marked **[CHECK]** are
reconstructions of the original pipeline that should be confirmed against the original code.

---

## 2 Data and Methods (revised)

### 2.1 Study area, period and in-situ data
We use all 192 Soil Climate Analysis Network (SCAN; Schaefer et al., 2007) stations in the
conterminous United States (CONUS) for 1 January 2019 – 31 December 2023. Hourly volumetric soil
moisture at 2 in (5 cm) and soil temperature at 2 in were retrieved from the NRCS AWDB REST
service. Hourly values outside 0.5–60 % vol and values more than 8 % vol from a centred 25-h
running median were removed. Daily means were computed when at least 12 hourly values were
available. Runs of 10 or more days with an identical daily value (stuck sensor) were removed, as
were days with a mean 5 cm soil temperature ≤ 0.5 °C (frozen soil, where the dielectric sensor
does not measure liquid water). This yields 251,391 daily values at 186 stations
[v2_0_dataset.csv]. The daily SMS value served by the NRCS report services is the reading at the
start of each day, not a daily mean, so daily means were computed from the hourly data.
Stations are grouped into the nine NOAA climate regions (Karl & Koss, 1984).

**Station coordinates.** Coordinates are taken from the NRCS station metadata. The coordinates
listed in the original study differ from the NRCS metadata by 2.2–359.3 km for the 16 study
stations (median 7.1 km; 8 of 16 stations more than 10 km; Shagbark Hills 359.3 km, Abrams
311.3 km, Rogers Farm #1 62.3 km)
[gee/stations_paper16_corrected.csv]. All satellite features were therefore re-extracted. For the
"before" reproduction, the 16 stations were also extracted at the original coordinates.

### 2.2 Satellite, reanalysis and static predictors
All predictors were extracted with Google Earth Engine (Gorelick et al., 2017), one station at a
time, and cached per station and product.

* **Sentinel-1** (Torres et al., 2012) IW GRD in linear power (VV, VH, incidence angle), averaged
  over a 50 m radius around the station (field scale, which reduces speckle and sensitivity to
  coordinate error). Pixels with VV < 10⁻³ or VH < 10⁻⁴ (border noise) are masked, and one
  acquisition is kept per station-day. Polarimetric indices are computed in linear units:
  cross-pol ratio VH/VV (in dB), RVI = 4·VH/(VV+VH) and SPAN = VV+VH. VV is normalised to 40°
  incidence with a per-station linear slope. Change-detection features (scaled VV between the
  5th and 95th percentile, and the anomaly from the median) are computed per station **and per
  relative orbit**, so that acquisition geometries are not mixed. Sentinel-1B failed in
  December 2021, so the median revisit lengthened from 5–6 days (2019–2021) to 7 days
  (2022–2023) [v2_0_dataset.csv].
* **Sentinel-2** L2A surface reflectance (Drusch et al., 2012), pixel-level cloud, shadow and snow
  masking with the Scene Classification Layer (classes 4–7 kept), mean over the 50 m buffer when
  ≥ 50 % of it is clear. Indices: NDVI, NDMI, EVI, NDWI, BSI. Values are linearly interpolated to
  daily over gaps of up to 30 days, together with the age of the nearest clear scene.
* **MODIS**: BRDF-corrected NBAR (MCD43A4 v061; NDVI, NDMI, EVI; quality ≤ 1), LAI (MCD15A3H
  v061; main-algorithm QC), and day/night land-surface temperature from Terra and Aqua (MOD11A1,
  MYD11A1 v061; QC bits 00).
* **Precipitation**: CHIRPS daily (Funk et al., 2015). Derived features: rain totals over 1, 3, 7,
  14 and 30 days, antecedent precipitation indices (k = 0.85, 0.90, 0.95) and days since rain.
* **Reanalysis and passive microwave**: ERA5-Land daily aggregates (Muñoz-Sabater et al., 2021;
  layer 1–2 soil water, 2 m and soil temperature, evaporation, potential evaporation, net solar
  radiation, snow cover). SMAP Level-4 surface and root-zone soil moisture (Reichle et al., 2019),
  averaged to daily on the server. SMAP Level-3 Enhanced radiometer soil moisture (all
  retrievals, because only about 20 % of retrievals at these stations carry the
  "recommended quality" flag).
* **Static**: SRTM elevation and slope, MERIT-Hydro upstream area converted to a topographic wetness
  index, SoilGrids 2.0 sand/clay/silt/bulk density/organic carbon at 0–5 cm, ESA WorldCover land
  cover (mode within 50 m), and Saxton & Rawls (2006) wilting point, field capacity and saturation.

Earth Engine reductions with a 50 m buffer return no value for products coarser than the buffer,
so all coarse products (CHIRPS, ERA5-Land, SMAP, MODIS) were sampled at the station point.

### 2.3 The original model ("before")
The original model was reproduced unchanged on 2019–2023 data at the original coordinates. It
uses 16 anchor features on Sentinel-1 dates:

* single-pixel VV and VH in dB;
* VV/VH ratio, RVI and SPAN computed on the dB numbers;
* Sentinel-2 NDVI, NDMI and EVI with a scene-level cloud filter only (< 20 % cloudy pixels, no
  pixel mask; nearest scene within ±5 days) **[CHECK]**;
* MODIS NDVI and EVI (MOD13Q1, latest 16-day composite) **[CHECK]**;
* elevation and slope;
* day of year and month;
* 2-day CHIRPS precipitation **[CHECK]**;
* `SM_lag1`, the in-situ value at the previous Sentinel-1 acquisition of the station.

Missing values are replaced by the mean of the other features in the same row
(`row.fillna(row.mean())`). The model is the mean of Random Forest (200 trees), XGBoost (300
trees, depth 6, learning rate 0.05) and LightGBM (300 trees, depth 6, learning rate 0.05).
Evaluation follows the original protocol: a random 80/20 split of the anchor rows of the 15
training stations, with Shagbark Hills as the independent validation station.

The original daily reconstruction alternates two models:

* on Sentinel-1 days, the anchor model;
* on other days, a "gap-fill" model with the daily-available subset of the features.

Both models were trained with the lag at the previous Sentinel-1 date but applied recursively
with the previous day's estimate. The "drift control" described in the original paper is not
present in its code.

### 2.4 Corrections
The corrections are applied one at a time (Table 2, "fix ladder") on the 16 original stations:

* S1: NRCS coordinates;
* S2: SAR indices in linear units;
* S3: 50 m field-scale linear mean instead of one pixel;
* S4: Sentinel-2 Scene Classification Layer cloud mask;
* S5: missing values left to the learners (XGBoost/LightGBM native handling; training medians for
  Random Forest) instead of the row-mean fill;
* S6: the lag defined identically in training and application (in-situ value of the previous day).

The drift control was implemented as a daily loop that resets to a satellite-only anchor estimate
on every Sentinel-1 date and uses the consistent 1-day lag on the days in between.

### 2.5 Models
* **Regularised tree ensemble**: Random Forest (400 trees, min. 5 samples per leaf, 50 % of
  features per split), XGBoost (600 trees, depth 5, learning rate 0.03, subsampling 0.8/0.7,
  λ = 5) and LightGBM (600 trees, 31 leaves, learning rate 0.03, subsampling 0.8/0.7, λ = 5);
  prediction = mean of the three.
* **TempoSAR**: a Transformer encoder (2 layers, 4 heads, width 48) over the station's previous 10
  Sentinel-1 acquisitions (backscatter, indices, optical, rain, SMAP L4, ERA5-Land, LST and the
  time gaps). It cross-attends from the current acquisition and fuses static descriptors and a
  land-cover embedding. Trained with a Huber loss, AdamW and a one-cycle schedule (60 epochs),
  with early stopping on 12 % of the training stations held out as validation. Three seeds,
  averaged.
* **TabPFN v2** (Hollmann et al., 2025), a tabular foundation model pretrained on synthetic data
  that predicts by in-context learning, with no gradient training and no hyper-parameter tuning on
  our data. TabPFN v2 was pretrained with at most 10,000 training samples, so the context set is a
  subsample of at most 10,000 training rows. It is stratified so that every training station
  contributes equally; for leave-one-region-out it is balanced by region instead. The same 77
  features as the trees are used, with 4 ensemble members (default 8) to fit the 6 GB laptop GPU.
* **Hybrids**: equal-weight means of the above (no weights fitted). The hybrid of TabPFN, TempoSAR
  and trees was declared the proposed model before the full results were computed
  [results_v2/progress.md].
* **Few-shot station adaptation**: for a station unseen in training, its first k in-situ samples
  (k = 5, 10, 20) are used and evaluation is on all later dates of that station. Methods:
  * bias or linear correction of the out-of-station prediction;
  * L2-SP fine-tuning of TempoSAR (Li et al., 2018);
  * TabPFN in-context adaptation: the k samples are added to the context set, with no training.
    For tractability, one TabPFN fit is made per fold, with the first k samples of every held-out
    station of the fold in the context. No station sees its own later data.
* **Daily reconstruction**: LightGBM on daily drivers (800 trees) and the DailyLSTM of the
  improvement package (60-day input window, 64 hidden units). The LSTM is either trained from
  scratch or pretrained on a dense target available everywhere (ERA5-Land layer-1 soil water, or
  SMAP L4 surface soil moisture) at all days of the training stations, then fine-tuned on in-situ
  data with L2-SP (5 epochs).

### 2.6 Evaluation
Four protocols are reported:

* the original random 80/20 row split, which leaks station identity and serial correlation;
* leave-stations-out, 10-fold, with stations as groups;
* leave-one-region-out over the nine climate regions;
* temporal: train 2019–2021, test 2022–2023. The test period coincides with the loss of
  Sentinel-1B.

Every model is reported with and without an in-situ lag input, always next to the matching
**persistence** baseline, which is the lag value itself used as the estimate. Metrics:

* coefficient of determination R² and RMSE;
* unbiased RMSE (Entekhabi et al., 2010) and mean bias, all pooled;
* the median of per-station Pearson r.

All feature sets of one protocol are scored on the same 33,084 Sentinel-1 station-days at 184
stations (stations with ≥ 20 labelled dates and both lag definitions available)
[v2_0_dataset.csv]. A leakage audit [v2_leakage_audit.txt] confirmed that no in-situ-derived
variable enters any satellite-only feature set and that the folds are disjoint for stations,
regions and time. No single predictor correlates with in-situ soil moisture above r = 0.73
(SMAP L4 surface wetness).

Uncertainty is estimated with a station-cluster bootstrap: stations are resampled with replacement
(1000 resamples), and pooled R² and RMSE are recomputed. Differences between two models are paired
(the same resample for both). The 95 % interval of a paired ΔR² is about ±0.01 at unseen stations,
which is the smallest difference these data can resolve [v3_models.csv].

### 2.7 Improvements to the proposed hybrid (v3/v4)
Every change below was added as an extra row; the v2 headline model was not replaced. Each round
was declared in the progress log before its results were computed [results_v2/progress.md],
except the first POLARIS experiment, which is labelled exploratory.

* **Error diagnosis.** The out-of-fold error of the hybrid at unseen stations was split into a
  constant per-station offset and the remainder. The offset is about 58 % of the mean squared
  error (44 % in the temporal test). Quantile-mapping to SMAP L4 was dropped because SMAP's
  station means are further from the truth than the hybrid's (station-mean RMSE 8.42 vs 5.92).
* **TabPFN context bagging (v3-1).** Five station-stratified 10,000-row contexts per fold instead
  of one, with predictions averaged. Bag 0 uses the v2 seeds.
* **Station-mean correction (v3-2).** A second-stage random forest predicts each station's mean
  wetness from station-level aggregates of the satellite and static features. The blend weight is
  fitted on inner folds grouped by station.
* **POLARIS soil properties (v3-3, exploratory; v4).** POLARIS v1.0 (Chaney et al., 2019) 30 m
  SSURGO-based maps of saturated and residual water content, sand, silt, clay, bulk density,
  organic matter, saturated hydraulic conductivity and the van Genuchten n and α, at 0–5 and
  5–15 cm. Each is taken as the point value and the mean of a 33 × 33-pixel (~1 km) window, read
  over HTTP from the public tiles. These are static map products with no in-situ input. In v4 the
  extraction was repeated at the NRCS AWDB station coordinates. Those differ from the station list
  used so far by more than 1 km at 21 of 184 stations.
* **Relative-wetness target (v4-2).** The trees are fitted on Se = (θ/100 − θr)/(θs − θr), with
  POLARIS θs and θr at 0–5 cm (denominator floored at 0.05), and back-transformed to % vol.
* **TempoSAR with POLARIS (v4-3).** The POLARIS window means are added to TempoSAR's static
  descriptors. A re-run of TempoSAR without them on the same hardware is the control.
* **Final model (v4).** The equal-weight mean of TabPFN (bag of 5), TempoSAR + POLARIS, and the
  trees + POLARIS at exact coordinates with the Se target. It was declared before the v4 results.
* **Daily product.** The hybrid predicts only on Sentinel-1 dates. Its out-of-fold estimates are
  linearly interpolated, or held, onto every day at the same held-out station. They are scored
  separately on Sentinel-1 days and on all other days [v4_daily.csv].

TabPFN and TempoSAR runs for v3/v4 used Kaggle GPU T4 ×2 with `tabpfn` 10.2.0. Its re-run of the v2
TabPFN context (bag 0) scores R² 0.613 vs 0.614 in v2.

---

## 3 Results (revised)

All values below are in `results_v2/tables/v2_summary_before_after.md` (built from the per-experiment
tables named in brackets). RMSE, ubRMSE and bias are in % volumetric soil moisture at 5 cm.

### 3.1 Reproducing the original evaluation
With 2019–2023 data and the original pipeline, the ensemble reaches R² = 0.780 (RMSE 5.29) on the
random 20 % test rows of the 15 training stations [v2_1a]. This is the protocol used for the
headline number of the original study. Two diagnostics show that this number does not measure
retrieval skill:

* **Persistence.** Persistence, i.e. reusing the lag value (the in-situ value at the previous
  Sentinel-1 date), reaches R² = 0.744 on the 567 test rows that have a lag (the ensemble: 0.792
  on those rows).
* **Unseen station.** At the independent validation station Shagbark Hills, the model gives
  R² = 0.269 (RMSE 7.14, 166 rows). On the 156 rows with a lag, persistence gives 0.268 and the
  model 0.410.

On the original daily loop at Shagbark Hills, R² = 0.051 (RMSE 8.17) [v2_1c].

Scaled to all CONUS SCAN stations (33,084 Sentinel-1 station-days at 184 stations), the same model
reaches the following satellite-only R² (without the lag input) [v2_2]:

| Protocol | R² |
|---|---|
| random 80/20 split | 0.790 |
| unseen stations (leave-stations-out) | 0.303 |
| unseen regions (leave-one-region-out) | 0.153 |
| unseen years (2022–2023) | 0.636 |

The gap between 0.790 and 0.303 is the information that the random split leaks through station
identity.

With the lag, the model scores R² 0.873 at unseen stations, against 0.850 for persistence. Almost
all of its skill comes from the in-situ input.

### 3.2 Effect of the corrections (16 original stations, leave-one-station-out)
Satellite-only R² improves at each step:

| Step | Correction | R² |
|---|---|---|
| S0 | as published | −0.389 |
| S1 | + NRCS coordinates | −0.028 |
| S2 | + SAR indices in linear units | −0.014 |
| S3 | + 50 m field-scale mean | 0.032 |
| S4 | + Sentinel-2 SCL cloud mask | 0.035 |
| S5 | + NaN-aware learners | 0.032 |

[v2_1b; Fig. 2]. Correcting the coordinates is by far the largest single effect (+0.36). After
S5 the 16-feature design still has essentially no skill at an unseen station: the bugs were not
the main limitation, the feature set was.

With the lag, the original definition (in-situ value at the previous Sentinel-1 date) gives
R² 0.691 after S5, which is below persistence (0.744). Using the consistent 1-day lag (S6) gives
0.911, which is still below 1-day persistence (0.957). The original model with its lag therefore
never outperforms persistence at an unseen station.

The original daily loop applies the gap-fill model recursively to its own previous estimate,
without drift control. At unseen stations it scores R² −0.345 (bias +5.95). Implementing the
drift control (reset to a satellite-only anchor estimate on every Sentinel-1 date) improves this
to −0.138, but the loop remains unusable with the 16 features [v2_1c].

### 3.3 Satellite-only retrieval with the improved pipeline (all stations)
The improved feature set (77 satellite, reanalysis and static predictors, no in-situ input)
raises satellite-only skill under every protocol [v2_2, v2_3; Fig. 1, 3, 5]. Each cell gives
paper model (trees on the 16 paper features, no lag) → proposed hybrid of TabPFN v2 + TempoSAR +
trees:

| Protocol | R² | RMSE | ubRMSE | median per-station r |
|---|---|---|---|---|
| unseen stations | 0.303 → **0.647** | 10.23 → **7.28** | 10.18 → **7.25** | 0.578 → **0.810** |
| unseen climate region | 0.153 → **0.643** | 11.27 → **7.32** | — | 0.503 → **0.807** |
| unseen years 2022–23 | 0.636 → **0.792** | 7.28 → **5.50** | — | 0.579 → **0.831** |

The proposed hybrid was declared before the results were computed. It is the best or tied-best
model under every protocol. Its components on their own:

| Model | Unseen stations | Unseen region | Unseen years |
|---|---|---|---|
| regularised trees | 0.636 | 0.642 | 0.795 |
| TempoSAR | 0.621 | 0.602 | 0.699 |
| TabPFN v2 | 0.614 | 0.603 | 0.782 |

TabPFN v2 reaches the accuracy of the tuned tree ensemble with no training and no tuning. Alone it
has a positive bias (+1.58 at unseen stations), which the hybrid reduces (+0.63). On the 16
original stations alone (predictions from the same leave-stations-out models), TabPFN v2 is the
best single model: R² 0.193 → 0.570 and RMSE 10.10 → 7.38 [v2_7].

**Where the skill comes from** [v2_6; Fig. 11]. Used directly, operational products give:

| Product | R² raw | R² after calibration on the training stations |
|---|---|---|
| SMAP L4 | 0.366 | 0.438 |
| SMAP L3E | 0.348 | 0.448 |
| ERA5-Land | −0.039 | 0.435 |

The learned models improve on these by about 0.2 in R². Removing feature groups from the tree
ensemble (unseen stations):

| Feature set | R² |
|---|---|
| all features | 0.636 |
| without Sentinel-1 | 0.633 |
| without optical/LST | 0.634 |
| without SMAP and ERA5-Land | 0.602 |
| Sentinel-1 + static only | 0.552 |

The results are similar for unseen regions. At unseen stations, the satellite-only skill is
therefore carried mostly by the passive-microwave and reanalysis products combined with static
soil and terrain descriptors. Sentinel-1 is largely redundant with them. On its own, however, the
corrected Sentinel-1 processing with static descriptors (0.552) is far better than the original
16-feature design (0.303).

### 3.4 Nowcasting with a consistent 1-day in-situ lag
When yesterday's in-situ value is available, TabPFN v2 is the only model that clearly beats 1-day
persistence under all three protocols [v2_3; Fig. 4]:

| Protocol | TabPFN v2 RMSE | Persistence RMSE | Error reduction |
|---|---|---|---|
| unseen stations | 1.59 | 1.90 | 16 % |
| unseen region | 1.61 | 1.90 | 15 % |
| unseen years | 1.58 | 1.92 | 18 % |

The regularised trees with the same lag also beat persistence, by less: RMSE 1.80 / 1.79 / 1.63.

### 3.5 Few-shot adaptation to a new station
For a station not seen in training, its first k in-situ samples are used for adaptation, and
evaluation is on all its later dates [v2_4; Fig. 6].

With k = 10:

| Method | R² | ubRMSE |
|---|---|---|
| no adaptation | 0.641 | 7.20 |
| bias correction (trees) | 0.724 | — |
| L2-SP fine-tuned TempoSAR | 0.772 | — |
| TabPFN in-context (k samples added to the context, no training) | 0.774 | 5.71 |
| fine-tuned TempoSAR + bias-corrected trees | 0.779 | — |

With k = 20, TabPFN in-context reaches R² 0.798 (ubRMSE 5.43); with k = 5 it reaches 0.734.
Mean/std (linear) rescaling from the first 10 samples is harmful (e.g. trees R² 0.163): the
spread of a short early record does not represent the station's full range.

### 3.6 Daily reconstruction
Over 243,215 station-days at unseen stations (5-fold leave-stations-out) [v2_5; Fig. 7]:

| Method | R² | RMSE |
|---|---|---|
| satellite-only LightGBM on the daily drivers | 0.628 | 7.92 |
| linear interpolation of the hybrid's satellite estimates between Sentinel-1 dates | 0.626 | 7.94 |
| DailyLSTM, from scratch | 0.515 | — |
| DailyLSTM, pretrained on SMAP L4 | 0.567 | — |
| DailyLSTM, pretrained on ERA5-Land | 0.558 | — |
| original daily loop (16 original stations) | −0.345 | — |

Pretraining the DailyLSTM on a dense target that is available everywhere improves it by about
0.05 in R², but it does not reach the tree model.

### 3.7 Improved hybrid (v3/v4)
[v3_models.csv, V3_RESULTS.md]. ΔR² is paired against the v2 hybrid, with 95 % station-cluster
bootstrap intervals.

| Model | Unseen stations R² | ΔR² [95 % CI] | Unseen region R² | ΔR² [95 % CI] |
|---|---|---|---|---|
| v2 hybrid (headline) | 0.647 | — | 0.643 | — |
| + TabPFN bag of 5 | 0.651 | +0.003 [−0.001, +0.007] | 0.648 | +0.005 [+0.001, +0.010] |
| + station-mean correction (declared v3 candidate) | 0.648 | +0.001 [−0.009, +0.012] | 0.630 | −0.014 [−0.031, +0.001] |
| + bag of 5, trees + POLARIS (exploratory) | 0.669 | +0.022 [+0.009, +0.037] | 0.663 | +0.021 [+0.010, +0.032] |
| **v4 final** (declared before results) | **0.671** | **+0.024 [+0.005, +0.045]** | **0.669** | **+0.026 [+0.008, +0.045]** |

For the v4 final model at unseen stations: RMSE 7.28 → 7.03, ubRMSE 7.25 → 7.00, bias +0.63 →
+0.62. The median per-station r is 0.810 → 0.803. The gain comes from the station's level, not from
its temporal dynamics. POLARIS raises the tree ensemble alone from 0.636 to 0.669. TempoSAR with
POLARIS reaches 0.641, against 0.623 for its re-run control. TabPFN with POLARIS is worse (0.594).
Three steps change the hybrid by less than the ±0.01 noise floor: exact coordinates for POLARIS
(+0.001), the Se target (−0.000) and the bagging step. The best combination found (exact
coordinates, raw target) reaches 0.677 / 0.670, but it was identified after the results and is not
the reported model. In the temporal test, only bagging applies (0.792 → 0.796).

**Daily, Sentinel-1 vs other days** [v4_daily.csv]. 246,555 common labelled days at unseen
stations, of which 34,504 are Sentinel-1 days. With linear interpolation between Sentinel-1 dates:

| Model | Sentinel-1 days | Other days | All days |
|---|---|---|---|
| v2 hybrid | 0.642 | 0.621 | 0.625 |
| v4 final | 0.666 | 0.650 | 0.654 (ΔR² +0.028 [+0.008, +0.050]) |
| v4 final, hold-last (real-time) | 0.663 | 0.619 | 0.627 |
| LGBM-daily satOnly (v2-5) | 0.627 | 0.627 | 0.629 |

The daily models score about the same on Sentinel-1 days and other days. The interpolated hybrid
loses 0.016 in R² between overpasses. At unseen regions the v4 final scores 0.662 / 0.642 / 0.646.
Averaging the interpolated v4 hybrid with LGBM-daily gives 0.665 on all days (exploratory).

---

## 4 Limitations

* **In-situ data.** In-situ data are from SCAN only. USCRN was planned but the NCEI server was
  unavailable (HTTP 503) during the study, so the network does not cover the Northwest (4
  stations) or the Upper Midwest (6) well.
* **Unseen-years test.** The 2022–2023 test period coincides with the loss of Sentinel-1B (median
  revisit 5–6 → 7 days). The temporal result therefore mixes temporal transfer with a change in
  sampling. Site descriptors computed from the full satellite record (e.g. backscatter
  percentiles) use 2022–2023 satellite data, but never in-situ data, in that test.
* **SMAP/ERA5 dominance.** Most of the satellite-only skill comes from SMAP L4/L3E and ERA5-Land.
  These are coarse (9–11 km) products that are themselves model- or retrieval-based, so the
  "station-scale" claim rests on how the learned model maps coarse soil moisture to the point
  scale. That mapping still leaves RMSE ≈ 7 % vol at unseen stations.
* **Reconstructed details.** Parts of the original pipeline were reconstructed from the
  description of the earlier analysis rather than the original code: the Sentinel-2 matching
  window, the MODIS product, the precipitation aggregation and the gap-fill feature set. The
  "before" numbers depend on these choices.
* **TabPFN settings.** TabPFN v2 was used within its pretraining limit of 10,000 context rows,
  with 4 instead of 8 ensemble members, on a 6 GB GPU. The few-shot TabPFN context contains the
  first k samples of all held-out stations of a fold, not only the target station's.
* **No hyper-parameter search.** No hyper-parameter search was done for any model; all deep models
  use fixed settings, and the DailyLSTM was trained for 10 epochs without early stopping. Results
  for TempoSAR and the DailyLSTM are therefore lower bounds of what these architectures can do.
* **Exploratory origin of POLARIS.** POLARIS was first added after the pre-registered
  station-mean correction failed. The v4 rows that include it were pre-registered, but the choice
  to use soil maps was informed by v3 results. POLARIS is CONUS-only and SSURGO-based, which
  limits transfer outside the US.
* **Coordinates of the satellite features.** The NRCS coordinates differ from the station list
  used for the Earth Engine extraction by more than 1 km at 21 of 184 stations (up to 27.5 km).
  Only POLARIS was re-extracted at the exact coordinates.
* **Pooled metrics.** Pooled R² across stations includes between-station variance. The median
  per-station r and ubRMSE are reported alongside for this reason.
