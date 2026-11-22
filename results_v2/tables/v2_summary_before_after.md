# Before / after summary (all values from results_v2/tables)
RMSE, ubRMSE, bias in % volumetric soil moisture (5 cm).

### 1 BEFORE: paper model as published, paper protocol (16 stations, paper coordinates)

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| random 80/20 test rows, 15 training stations (paper's headline protocol) | 581 | 0.780 | 5.29 | 5.29 | +0.16 | 0.694 |
| Shagbark Hills, independent validation station | 166 | 0.269 | 7.14 | 6.87 | -1.95 | 0.612 |
| Shagbark Hills, persistence of the lag (same rows with a lag) | 156 | 0.268 | 7.09 | 7.08 | +0.21 | 0.636 |
| daily loop at Shagbark Hills (recursive, as in the paper) | 1155 | 0.051 | 8.17 | 7.58 | -3.05 | 0.544 |

### 2 BEFORE, honest evaluation: same model, all CONUS SCAN stations

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| random80/20: paper model with lag (prevS1) | 6617 | 0.892 | 4.02 | 4.02 | +0.02 | 0.839 |
| random80/20: persistence (lag at previous S1 date) | 6617 | 0.856 | 4.65 | 4.65 | +0.11 | 0.799 |
| random80/20: paper model without lag (satellite-only) | 6617 | 0.790 | 5.61 | 5.61 | +0.01 | 0.677 |
| leave-stations-out: paper model with lag (prevS1) | 33084 | 0.873 | 4.36 | 4.36 | +0.13 | 0.804 |
| leave-stations-out: persistence (lag at previous S1 date) | 33084 | 0.850 | 4.75 | 4.75 | +0.09 | 0.769 |
| leave-stations-out: paper model without lag (satellite-only) | 33084 | 0.303 | 10.23 | 10.18 | +1.03 | 0.578 |
| leave-one-region-out: paper model with lag (prevS1) | 33084 | 0.864 | 4.51 | 4.50 | +0.27 | 0.795 |
| leave-one-region-out: persistence (lag at previous S1 date) | 33084 | 0.850 | 4.75 | 4.75 | +0.09 | 0.769 |
| leave-one-region-out: paper model without lag (satellite-only) | 33084 | 0.153 | 11.27 | 11.22 | +1.13 | 0.503 |
| temporal(2022-23): paper model with lag (prevS1) | 11479 | 0.858 | 4.55 | 4.55 | +0.12 | 0.758 |
| temporal(2022-23): persistence (lag at previous S1 date) | 11479 | 0.818 | 5.15 | 5.15 | +0.09 | 0.709 |
| temporal(2022-23): paper model without lag (satellite-only) | 11479 | 0.636 | 7.28 | 7.24 | +0.75 | 0.579 |

### 3 FIXES on the paper's 16 stations (leave-one-station-out, satellite-only)

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| S0 paper as published | 3070 | -0.389 | 13.62 | 13.34 | +2.77 | 0.482 |
| S1 + corrected coordinates | 2966 | -0.028 | 11.59 | 11.27 | +2.69 | 0.478 |
| S2 + SAR indices in linear units | 2966 | -0.014 | 11.51 | 11.20 | +2.64 | 0.472 |
| S3 + 50 m field-scale SAR mean | 2966 | 0.032 | 11.25 | 10.77 | +3.22 | 0.531 |
| S4 + Sentinel-2 SCL cloud mask | 2966 | 0.035 | 11.23 | 10.81 | +3.06 | 0.508 |
| S5 + no row-mean fill (NaN-aware) | 2966 | 0.032 | 11.25 | 10.77 | +3.26 | 0.525 |
| S6 + consistent 1-day lag | 2966 | 0.032 | 11.25 | 10.77 | +3.26 | 0.525 |

### 4 AFTER: satellite-only models, all stations (no in-situ input)

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| leave-stations-out: paper16 fixed, trees | 33084 | 0.317 | 10.13 | 10.05 | +1.21 | 0.593 |
| leave-stations-out: Trees | 33084 | 0.636 | 7.39 | 7.39 | +0.25 | 0.804 |
| leave-stations-out: TempoSAR | 33084 | 0.621 | 7.54 | 7.54 | +0.06 | 0.765 |
| leave-stations-out: TabPFN | 33084 | 0.614 | 7.61 | 7.45 | +1.58 | 0.801 |
| leave-stations-out: Hybrid(TabPFN+TempoSAR+Trees) | 33084 | 0.647 | 7.28 | 7.25 | +0.63 | 0.810 |
| leave-one-region-out: paper16 fixed, trees | 33084 | 0.156 | 11.26 | 11.17 | +1.36 | 0.521 |
| leave-one-region-out: Trees | 33084 | 0.642 | 7.33 | 7.33 | +0.16 | 0.794 |
| leave-one-region-out: TempoSAR | 33084 | 0.602 | 7.72 | 7.71 | +0.48 | 0.773 |
| leave-one-region-out: TabPFN | 33084 | 0.603 | 7.72 | 7.62 | +1.22 | 0.796 |
| leave-one-region-out: Hybrid(TabPFN+TempoSAR+Trees) | 33084 | 0.643 | 7.32 | 7.29 | +0.62 | 0.807 |
| temporal(2022-23): paper16 fixed, trees | 11479 | 0.647 | 7.17 | 7.13 | +0.75 | 0.601 |
| temporal(2022-23): Trees | 11479 | 0.795 | 5.47 | 5.47 | -0.18 | 0.816 |
| temporal(2022-23): TempoSAR | 11479 | 0.699 | 6.62 | 6.62 | +0.10 | 0.803 |
| temporal(2022-23): TabPFN | 11479 | 0.782 | 5.64 | 5.64 | -0.23 | 0.826 |
| temporal(2022-23): Hybrid(TabPFN+TempoSAR+Trees) | 11479 | 0.792 | 5.50 | 5.50 | -0.10 | 0.831 |

### 5 AFTER: with the consistent 1-day in-situ lag (nowcasting), vs persistence

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| leave-stations-out: Trees(+lag1d) | 33084 | 0.978 | 1.80 | 1.80 | +0.04 | 0.972 |
| leave-stations-out: TabPFN(+lag1d) | 33084 | 0.983 | 1.59 | 1.59 | -0.09 | 0.975 |
| leave-stations-out: Persistence(1d) | 33084 | 0.976 | 1.90 | 1.90 | +0.02 | 0.965 |
| leave-one-region-out: Trees(+lag1d) | 33084 | 0.979 | 1.79 | 1.79 | -0.02 | 0.972 |
| leave-one-region-out: TabPFN(+lag1d) | 33084 | 0.983 | 1.61 | 1.60 | -0.12 | 0.975 |
| leave-one-region-out: Persistence(1d) | 33084 | 0.976 | 1.90 | 1.90 | +0.02 | 0.965 |
| temporal(2022-23): Trees(+lag1d) | 11479 | 0.982 | 1.63 | 1.63 | +0.01 | 0.978 |
| temporal(2022-23): TabPFN(+lag1d) | 11479 | 0.983 | 1.58 | 1.57 | -0.06 | 0.979 |
| temporal(2022-23): Persistence(1d) | 11479 | 0.975 | 1.92 | 1.92 | +0.05 | 0.970 |

### 6 AFTER: few-shot adaptation to a new station (k = 10 first samples)

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid(TabPFN+Trees)|none | 31244 | 0.641 | 7.26 | 7.20 | +0.91 | 0.808 |
| Trees|bias | 31244 | 0.724 | 6.37 | 6.36 | +0.29 | 0.806 |
| FT-TempoSAR | 31244 | 0.772 | 5.79 | 5.79 | +0.19 | 0.743 |
| TabPFN in-context | 31244 | 0.774 | 5.76 | 5.71 | +0.80 | 0.809 |
| Hybrid(TabPFN in-context+Trees|bias) | 31244 | 0.774 | 5.77 | 5.74 | +0.54 | 0.819 |
| Hybrid(FT-TempoSAR+Trees|bias) | 31244 | 0.779 | 5.70 | 5.70 | +0.24 | 0.799 |

### 7 AFTER: daily reconstruction at unseen stations

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| LGBM-daily satOnly | 243215 | 0.628 | 7.92 | 7.90 | +0.52 | 0.802 |
| LGBM-daily +lag1d | 243215 | 0.969 | 2.27 | 2.26 | +0.17 | 0.973 |
| DailyLSTM satOnly (scratch) | 243215 | 0.515 | 9.05 | 9.02 | +0.70 | 0.695 |
| DailyLSTM satOnly (pretrained ERA5-Land) | 243215 | 0.558 | 8.64 | 8.57 | +1.11 | 0.736 |
| DailyLSTM satOnly (pretrained SMAP L4) | 243215 | 0.567 | 8.54 | 8.53 | +0.55 | 0.742 |
| Anchor hold-last (Hybrid(TabPFN+TempoSAR+Trees)) | 243215 | 0.599 | 8.23 | 8.19 | +0.72 | 0.680 |
| Anchor linear interp (Hybrid(TabPFN+TempoSAR+Trees)) | 243215 | 0.626 | 7.94 | 7.91 | +0.66 | 0.747 |
| Persistence(1d) | 243215 | 0.975 | 2.05 | 2.05 | +0.02 | 0.965 |
| paper's 16 stations, 4-fold LOSO: drift-controlled loop (satellite reset + 1-day lag) | 22855 | -0.138 | 12.63 | 12.24 | +3.14 | 0.537 |
| paper's 16 stations, 4-fold LOSO: paper loop (recursive, no drift control) | 22855 | -0.345 | 13.73 | 12.37 | +5.95 | 0.491 |
| paper's 16 stations, 4-fold LOSO: persistence (in-situ yesterday) | 22589 | 0.958 | 2.41 | 2.41 | +0.02 | 0.947 |
| paper's 16 stations, 4-fold LOSO: paper loop fed with in-situ lag | 22855 | 0.742 | 6.01 | 5.65 | +2.06 | 0.873 |

### 8 WHERE THE SKILL COMES FROM: raw products and feature-group ablation (trees, leave-stations-out)

| setting | n | R² | RMSE | ubRMSE | bias | median station r |
|---|---:|---:|---:|---:|---:|---:|
| SMAP L4 surface SM (raw) | 33084 | 0.366 | 9.75 | 9.72 | +0.80 | 0.722 |
| SMAP L4 surface SM (raw) + linear calibration | 33084 | 0.438 | 9.18 | 9.18 | +0.05 | 0.722 |
| SMAP L3E SM (raw) | 30065 | 0.348 | 9.95 | 9.69 | +2.23 | 0.676 |
| SMAP L3E SM (raw) + linear calibration | 30065 | 0.448 | 9.15 | 9.15 | +0.01 | 0.676 |
| ERA5-Land layer-1 SM (raw) | 33084 | -0.039 | 12.49 | 10.72 | +6.40 | 0.733 |
| ERA5-Land layer-1 SM (raw) + linear calibration | 33084 | 0.435 | 9.20 | 9.20 | +0.03 | 0.733 |
| all improved v2 features | 33084 | 0.636 | 7.39 | 7.39 | +0.25 | 0.804 |
| without Sentinel-1 | 33084 | 0.633 | 7.42 | 7.41 | +0.26 | 0.799 |
| without SMAP + ERA5-Land | 33084 | 0.602 | 7.73 | 7.72 | +0.33 | 0.753 |
| without optical / LST | 33084 | 0.634 | 7.41 | 7.41 | +0.22 | 0.798 |
| Sentinel-1 + static only | 33084 | 0.552 | 8.20 | 8.18 | +0.48 | 0.621 |
| SMAP + ERA5-Land + static only | 33084 | 0.625 | 7.50 | 7.50 | +0.27 | 0.783 |
