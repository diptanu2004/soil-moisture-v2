# Progress log

## 2026-10-21 — Step 1 (setup check) — DONE, waiting for go-ahead
- Python 3.12.6 venv OK. Installed pyarrow 26.0.0, xgboost 3.4.1, geopandas 1.2.0 (tabpfn 9.2.0, earthengine-api 1.7.49, huggingface_hub 2.2.0 were already there).
- Hardware: i7-13620H (10C/16T), 15.7 GB RAM, RTX 4050 Laptop 6 GB (driver CUDA 12.7), 111 GB free on C:.
- torch in venv is CPU-only (2.14.1+cpu); CUDA 12.6 wheel downloaded but NOT installed (awaiting decision).
- Earth Engine OK (project soil-moisture-iiita). S1 VV at Ames, Jun 2019: 5 scenes, 1.5 s. All 20 collection IDs in gee_extract_v2.py exist.
- gee_extract_v2.py needs fixes before the full run (bounding-box reduceRegions -> per-station; coarse products empty with 50 m buffer -> point sampling; SMAP memory limit -> yearly chunks; Drive export -> computeFeatures/getRegion with on-disk cache).
- Hugging Face OK (whoami works). TabPFN v2 regressor loads; toy R2 0.988; 3000 train rows = 40 s on CPU.
- SCAN Report Generator URL works (Ames, 10 days). Values are start-of-day, not daily mean.
- USCRN: NCEI /pub returns 503 (outage); mirror lacks 2019-2022 files. Blocked for now.

## 2026-10-21 — Step 2 started (go-ahead received)
- 14:35 CUDA torch 2.14.1+cu126 installed; RTX 4050 visible to torch.
- 14:40 gee_extract_v2.py: added cached per-station pull (run_cached). Fixed: bounding-box over-pull, empty coarse products (point sampling), SMAP memory limit (yearly chunks, L4 daily mean on server), S2 Element/Image bug. Paper-style inputs (S1 single pixel dB, unmasked S2 at point, MOD13Q1 NDVI/EVI) are pulled alongside the v2 inputs.
- Stations: 192 CONUS SCAN (NRCS coords; includes the paper's 16) + the paper's 16 at the paper's own (wrong) coordinates for an honest "before". USCRN still HTTP 503 at NCEI -> SCAN only for now.
- 14:49 EE extraction running: 2384 jobs, 8 workers, ETA ~1 h. Cache: data/raw_v2/ee_cache/<product>/<sid>.parquet (resumable).
- 14:58 SCAN hourly (AWDB REST API) download running, 3 workers; daily mean from hourly, QC + frozen-soil mask. Cache: data/raw_v2/scan_hourly/.
- 15:12 assemble_v2.py rewritten for the cache (QC + units; paper-style table paper_anchor/paper_daily). Fixed on real data: MODIS LST night QC read as null by getRegion (was dropping every night value); pandas datetime unit mismatch.
- 15:12 Experiment code written in src_v2/: exp_v2_1_before (paper model + fix ladder + daily loop), exp_v2_2_protocols (all stations x 4 protocols), exp_v2_3_models (TempoSAR GPU, TabPFN v2, hybrids), exp_v2_4_fewshot, exp_v2_5_daily (DailyLSTM GPU). Smoke tests on partial data in progress.
- 15:23 Smoke tests on partial data pass for v2-1 (before + fix ladder + daily loop) and TempoSAR on GPU (26 s per fold per seed).
- 15:23 PRE-DECLARED (before seeing full results): the proposed satellite-only model is the equal-weight hybrid of TabPFN v2 + TempoSAR + regularised trees ("Hybrid(TabPFN+TempoSAR+Trees)"); the headline protocol is leave-stations-out. All other models are reported in full tables; no model is picked by its test score.
- 15:23 TabPFN timing on the RTX 4050: ~90 s per fold (10k context, 4 estimators). Few-shot TabPFN uses one fit per fold and k (context = training stations + first k samples of every held-out station) instead of one fit per station (would be ~12 h).
- 15:32 Smoke tests passed for v2-5 daily (GPU DailyLSTM), figures and summary scripts. Pipeline driver src_v2/run_v2.sh ready; waiting for downloads (EE ETA ~15:50, SCAN ~16:00), then the full analysis (~5 h).
- 15:44 All five experiments smoke-tested end to end on partial data (v2-3 and v2-4 on one fold). HANDS-OFF CHAIN STARTED (src_v2/run_after_downloads.sh, detached): waits for the downloads, re-runs them until complete (retries every 10 min if the network is down), then runs the full analysis offline (src_v2/run_v2.sh). Windows idle-sleep is held off while it runs (keep the lid open / plugged in; a closed lid still sleeps). Each step writes "step <name> ok" or "STEP <name> FAILED" below.
- 15:53 downloads complete: EE 2384/2384, SCAN 960/960
- 15:53 analysis pipeline started (offline)
- 15:54 STEP assemble FAILED (see results_v2/logs/assemble.log)
- 2026-10-21 15:54 v2-1 started
- 15:54 STEP v2_2 FAILED (see results_v2/logs/v2_2.log)
- 15:55 assemble fix (NaN incidence angle at one station broke the per-station angle fit); assembly OK: 186 stations with 5 cm data, 34763 labelled S1 days, 251391 labelled station-days. Pipeline relaunched.
- 15:55 downloads complete: EE 2384/2384, SCAN 960/960
- 15:55 analysis pipeline started (offline)
- 15:55 step assemble ok
- 2026-10-21 15:55 v2-1 started
- 2026-10-21 15:55 v2-2 started: 33084 rows, 184 stations, 9 regions
- 2026-10-21 15:55 v2-5 daily started: 186 stations, 251391 labelled days, 36 drivers, device cuda
- 2026-10-21 16:01 v2-2 protocol random80/20 done (5.4 min)
- 2026-10-21 16:03 v2-5 fold 0 done (7.4 min)
- 2026-10-21 16:05 v2-1 fix ladder: S0 paper as published done
- 2026-10-21 16:10 v2-5 fold 1 done (15.1 min)
- 2026-10-21 16:15 v2-1 fix ladder: S1 + corrected coordinates done
- 2026-10-21 16:18 v2-5 fold 2 done (22.3 min)
- 2026-10-21 16:21 v2-1 fix ladder: S2 + SAR indices in linear units done
- 2026-10-21 16:23 v2-5 fold 3 done (27.3 min)
- 2026-10-21 16:27 v2-1 fix ladder: S3 + 50 m field-scale SAR mean done
- 2026-10-21 16:28 v2-5 fold 4 done (33.0 min)
- 2026-10-21 16:28 v2-5 finished
- 16:28 step v2_5 ok
- 2026-10-21 16:34 v2-1 fix ladder: S4 + Sentinel-2 SCL cloud mask done
- 2026-10-21 16:41 v2-1 fix ladder: S5 + no row-mean fill (NaN-aware) done
- 2026-10-21 16:49 v2-1 fix ladder: S6 + consistent 1-day lag done
- 2026-10-21 16:53 v2-2 protocol leave-stations-out done (57.9 min)
- 2026-10-21 17:01 v2-1 daily loop paper: fold with 1 test stations done (12.1 min)
- 2026-10-21 17:14 v2-1 daily loop nrcs: fold with 4 test stations done (24.8 min)
- 2026-10-21 17:30 v2-1 daily loop nrcs: fold with 4 test stations done (40.5 min)
- 2026-10-21 17:34 v2-2 protocol leave-one-region-out done (98.9 min)
- 2026-10-21 17:38 v2-2 protocol temporal(2022-23) done (102.4 min)
- 2026-10-21 17:38 v2-2 finished
- 17:38 step v2_2 ok
- 2026-10-21 17:38 v2-3 started (33084 rows)
- 2026-10-21 17:40 v2-1 daily loop nrcs: fold with 4 test stations done (50.6 min)
- 2026-10-21 17:45 v2-1 daily loop nrcs: fold with 4 test stations done (55.7 min)
- 2026-10-21 17:45 v2-1 finished
- 17:45 step v2_1 ok
- 2026-10-21 17:52 v2-6 ablation started
- 2026-10-21 18:16 v2-6 leave-stations-out done (23.9 min)
- 2026-10-21 18:23 v2-3 leave-stations-out done (45.6 min)
- 18:24 Reviewed v2-1, v2-2, v2-5, v2-3 (leave-stations-out) and v2-6 (LOSO ablation). Leakage audit clean. Added experiment v2-6 (ablation + raw SMAP/ERA5 baselines) because SMAP/ERA5 are the strongest predictors; finding: removing Sentinel-1 barely changes satellite-only skill at unseen stations.
- 2026-10-21 18:36 v2-6 leave-one-region-out done (44.4 min)
- 2026-10-21 18:36 v2-6 finished
- 2026-10-21 19:08 v2-3 leave-one-region-out done (90.6 min)
- 2026-10-21 19:17 v2-3 temporal(2022-23) done (99.8 min)
- 2026-10-21 19:17 v2-3 finished
- 19:18 step v2_3 ok
- 2026-10-21 19:18 v2-5 finished
- 19:18 step v2_5_interp ok
- 2026-10-21 19:18 v2-4 few-shot started
- 2026-10-21 19:25 v2-4 fold 0 done (7.0 min)
- 2026-10-21 19:32 v2-4 fold 1 done (14.2 min)
- 2026-10-21 19:39 v2-4 fold 2 done (21.1 min)
- 2026-10-21 19:47 v2-4 fold 3 done (28.8 min)
- 2026-10-21 19:54 v2-4 fold 4 done (36.0 min)
- 2026-10-21 20:01 v2-4 fold 5 done (43.4 min)
- 2026-10-21 20:08 v2-4 fold 6 done (50.4 min)
- 2026-10-21 20:15 v2-4 fold 7 done (57.6 min)
- 2026-10-21 20:23 v2-4 fold 8 done (64.8 min)
- 2026-10-21 20:30 v2-4 fold 9 done (72.0 min)
- 2026-10-21 20:30 v2-4 finished
- 20:30 step v2_4 ok
- 20:30 step audit ok
- 20:30 step summary ok
- 20:30 step figures ok
- 20:30 pipeline finished
- 21:32 step paper16 ok (run after the pipeline; added mid-run)
- 21:35 DONE. Figures re-rendered (layout fixes), summary table, RESULTS_SUMMARY.md, paper_text_draft.md (Methods + Results + Limitations) and citations.md written. run_v2.sh now also runs v2-6 and dataset_stats.
- 2026-11-22 PRE-REGISTRATION v3 (written before any v3 result exists). v2 rows are kept unchanged; the v2 headline stays Hybrid(TabPFN+TempoSAR+Trees).
  New rows (all satellite-only, LOSO / LORO; bagging also temporal):
    TabPFN-bag5            mean of 5 station-stratified 10k contexts (bag 0 = v2-3 seeds), n_estimators=4
    TabPFN-bag0(rerun)     bag 0 alone, re-run with the installed TabPFN version (isolates version drift)
    Hybrid(TabPFN-bag5+TempoSAR+Trees)   equal weights
    <hybrid> +StationMean(w=0.5)          anomaly of the hybrid at the station + 0.5*hybrid station mean + 0.5*station-mean model (RF on station-level satellite/static aggregates, fitted on training stations only)
    <hybrid> +StationMean(w nested)       same, w fitted on inner station folds (LGBM proxy)
    <hybrid> +StationMean(w nested)+scale anomaly scale factor fitted on the same inner folds
  v3 candidate (declared now): Hybrid(TabPFN-bag5+TempoSAR+Trees) +StationMean(w=0.5). Temporal protocol: station-mean correction not applied (stations are seen in training).
  Station-level cluster bootstrap CIs (1000 reps, paired differences vs the v2 headline) reported for every row.
- 2026-11-22 10:29 v3-2 station-mean started (184 stations, 113 station features)
- 2026-11-22 10:33 v3-2 leave-stations-out done (3.6 min)
- 2026-11-22 10:36 v3-2 leave-one-region-out done (6.8 min)
- 2026-11-22 10:36 v3-2 finished
- 2026-11-22 10:50 v3-2 result: station-mean correction does not help (LOSO dR2 +0.001 [-0.009, +0.012]; LORO -0.008..-0.014). Stage-2 station-mean errors are 0.94 correlated with the hybrid's.
- 2026-11-22 10:50 EXPLORATORY v3-3 (added after seeing v3-2, NOT pre-registered): POLARIS v1.0 30 m soil properties (10 variables x 0-5/5-15 cm, point + ~1 km window mean) read from hydrology.cee.duke.edu. Rows: Trees+POLARIS, Hybrid(TabPFN+TempoSAR+Trees+POLARIS) (+bag5 variant), and the station-mean correction with POLARIS. Gate: run the row-level trees only if POLARIS lowers the stage-2 station-mean RMSE below the hybrid's (5.92, LOSO). Report whatever happens.
- 2026-11-22 11:03 v3-2_polaris station-mean started (184 stations, 153 station features)
- 2026-11-22 11:07 v3-2_polaris leave-stations-out done (3.9 min)
- 2026-11-22 11:10 v3-2_polaris leave-one-region-out done (7.3 min)
- 2026-11-22 11:10 v3-2_polaris finished
- 2026-11-22 11:11 v3-3 POLARIS trees started (40 POLARIS columns)
- 2026-11-22 11:23 v3-3 leave-stations-out done (12.7 min)
- 2026-11-22 11:34 v3-3 leave-one-region-out done (23.8 min)
- 2026-11-22 11:34 v3-3 finished
- 2026-11-22 12:45 v3-1 bagging done on Kaggle 2xT4 (tabpfn 10.2.0; LOSO 95 min, LORO 102 min, temporal 27 min). Bag-0 re-run reproduces v2 TabPFN (LOSO 0.613 vs 0.614), so version drift is negligible.
  TabPFN-bag5 vs single context: LOSO 0.629 vs 0.613, LORO 0.626 vs 0.603, temporal 0.793 vs 0.782. Hybrid(TabPFN-bag5+TempoSAR+Trees): LOSO 0.651 (dR2 +0.003 [-0.001, +0.007]), LORO 0.648 (+0.005 [+0.001, +0.010]), temporal 0.796 (+0.003 [+0.001, +0.005]).
  Declared v3 candidate (Hybrid +StationMean w=0.5): LOSO 0.648 (+0.001, n.s.), LORO 0.630 (-0.014). It does not improve on v2.
  Exploratory v3-3 POLARIS: TabPFN+POLARIS is worse (LOSO 0.594). Best exploratory row: Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS), LOSO 0.669 (+0.022 [+0.009, +0.037]), LORO 0.663 (+0.021 [+0.010, +0.032]). Full table: tables/v3_models.csv, V3_RESULTS.md.
- 2026-11-22 PRE-REGISTRATION v4 (before any v4 result). Reference for v4 = Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS) (v3 exploratory, LOSO 0.669 / LORO 0.663). v2 and v3 rows stay unchanged. Protocols LOSO + LORO, same folds, satellite-only, bootstrap CIs vs both the v2 headline and the v4 reference.
  v4-1 exact coordinates: NRCS AWDB station metadata (4-5 decimals). POLARIS re-extracted there -> rows Trees+POLARIS(exact), Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS(exact)). Only the POLARIS columns move; S1/S2/MODIS/SMAP/ERA5 stay at the old coordinates.
  v4-2 relative-wetness target: trees fitted on Se = (sm/100 - theta_r)/(theta_s - theta_r) with POLARIS 0-5 cm window means (SoilGrids Saxton-Rawls fallback where POLARIS is missing), back-transformed. Rows Trees+POLARIS(Se) and the corresponding hybrid. Run with the v3 POLARIS table; repeated with exact-coordinate POLARIS if time allows (reported as a separate row).
  v4-3 TempoSAR with POLARIS static inputs (Kaggle 2xT4). TempoSAR(rerun, Kaggle) is the control on the other GPU. Rows TempoSAR(rerun), TempoSAR+POLARIS, and Hybrid(TabPFN-bag5+TempoSAR+POLARIS+Trees+POLARIS).
  v4 candidate (declared now): Hybrid(TabPFN-bag5 + TempoSAR+POLARIS + Trees+POLARIS(exact, Se)), i.e. all three changes together, equal weights.
- 2026-11-22 13:35 v4 trees Trees+POLARIS(Se) started
- 2026-11-22 13:49 v4 trees Trees+POLARIS(Se) leave-stations-out done (13.6 min)
- 2026-11-22 13:58 v4 trees Trees+POLARIS(exact) started
- 2026-11-22 13:58 v4 trees Trees+POLARIS(exact, Se) started
- 2026-11-22 14:01 v4 trees Trees+POLARIS(Se) leave-one-region-out done (25.7 min)
- 2026-11-22 14:01 v4 trees Trees+POLARIS(exact) started
- 2026-11-22 14:14 v4 trees Trees+POLARIS(exact) leave-stations-out done (13.3 min)
- 2026-11-22 14:26 v4 trees Trees+POLARIS(exact) leave-one-region-out done (25.0 min)
- 2026-11-22 14:26 v4 trees Trees+POLARIS(exact, Se) started
- 2026-11-22 14:39 v4 trees Trees+POLARIS(exact, Se) leave-stations-out done (13.5 min)
- 2026-11-22 14:35 v4-1 result (exact NRCS coordinates; 182/184 eval stations matched, 21 moved > 1 km, Starkville 17.7 km and Lovelock 27.5 km off in stations.csv): Hybrid(TabPFN-bag5+TempoSAR+Trees+POLARIS(exact)) LOSO 0.670 (vs v4 ref +0.001 [-0.006, +0.007]), LORO 0.664 (+0.000 [-0.007, +0.008]). Null. The POLARIS ~1 km window means barely change (r 0.95 approx vs exact).
- 2026-11-22 14:35 v4-2 result (Se target, v3 POLARIS): Hybrid(...+Trees+POLARIS(Se)) LOSO 0.669 (-0.000 [-0.005, +0.004]), LORO 0.663 (-0.001 [-0.005, +0.004]). Null.
- 2026-11-22 14:35 v4-3 (TempoSAR+POLARIS, Kaggle) queued since 13:33 waiting for a T4x2 machine; quota OK (27 h left).
- 2026-11-22 14:51 v4 trees Trees+POLARIS(exact, Se) leave-one-region-out done (25.5 min)
- 2026-11-22 v4-3 done on Kaggle (job finally ran; ~17 min on T4x2). TempoSAR+POLARIS alone: LOSO 0.641 vs control TempoSAR(rerun) 0.623; LORO 0.618 vs 0.597.
  Declared v4 candidate Hybrid(TabPFN-bag5+TempoSAR+POLARIS+Trees+POLARIS(exact, Se)): LOSO 0.671 (dR2 vs v2 headline +0.024, CI +0.005..+0.045; vs v4 ref +0.002, CI -0.010..+0.014, n.s.),
  LORO 0.669 (+0.026, CI +0.008..+0.045; vs v4 ref +0.005, n.s.). Best row (not declared): Hybrid(TabPFN-bag5+TempoSAR+POLARIS+Trees+POLARIS(exact)) LOSO 0.677 / LORO 0.670, vs v4 ref +0.007 n.s.
  Colab route dropped (Kaggle finished first).
- 2026-11-22 v4 daily (`src_v2/exp_v4_daily.py`, no fitting): v4 final hybrid's OOF anchor predictions interpolated onto every labelled day (v2-5 method), common 246,555 days.
  LOSO linear interp: all days 0.654 (v2 interp 0.625, dR2 +0.028 CI +0.008..+0.050); S1 days 0.666; non-S1 days 0.650. LORO: all 0.646, S1 0.662, non-S1 0.642.
  Exploratory: mean(v4 final interp, LGBM-daily satOnly) LOSO all days 0.665, non-S1 0.663 (+0.042 vs v2 interp). Table: tables/v4_daily.csv.
- 2026-11-23 docs updated with v3/v4 + daily results: README (before/after + architecture diagram), RESULTS_SUMMARY, paper_text_draft §2.7/§3.7, citations, CLAUDE.md; V3_RESULTS.md regenerated (now includes v4 daily tables).
