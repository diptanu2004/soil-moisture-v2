# Citations

Entries marked *(verify DOI)* were written from memory of the dataset landing pages; check the DOI
against the provider page before submission. All others are standard journal references.

## New model
- **TabPFN v2** — Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B.,
  Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular
  foundation model. *Nature*, 637, 319–326. https://doi.org/10.1038/s41586-024-08328-6
  Model weights: Prior Labs, `Prior-Labs/TabPFN-v2-reg` (Hugging Face); software: `tabpfn` Python
  package v9.2.0 (`ModelVersion.V2`) for v2; v10.2.0 (`ModelVersion.V2`) for the v3 context bagging on
  Kaggle (its bag-0 re-run reproduces the v2 TabPFN score: 0.613 vs 0.614).
- TabPFN (v1, background) — Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). TabPFN:
  A transformer that solves small tabular classification problems in a second. *ICLR 2023*.

## In-situ data
- **SCAN** — Schaefer, G. L., Cosh, M. H., & Jackson, T. J. (2007). The USDA Natural Resources
  Conservation Service Soil Climate Analysis Network (SCAN). *Journal of Atmospheric and Oceanic
  Technology*, 24(12), 2073–2077. https://doi.org/10.1175/2007JTECHA930.1
  Data access: NRCS National Water and Climate Center, AWDB REST API
  (https://wcc.sc.egov.usda.gov/awdbRestApi/), hourly SMS and STO at −2 in, accessed 2026-10-21.
- USCRN (planned, not used: NCEI server returned HTTP 503 during the run) — Bell, J. E., et al.
  (2013). U.S. Climate Reference Network soil moisture and temperature observations. *Journal of
  Hydrometeorology*, 14(3), 977–988. https://doi.org/10.1175/JHM-D-12-0146.1

## Satellite, reanalysis and static data (all accessed through Google Earth Engine)
- **Google Earth Engine** — Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R.
  (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of
  Environment*, 202, 18–27. https://doi.org/10.1016/j.rse.2017.06.031
- **Sentinel-1** (COPERNICUS/S1_GRD_FLOAT) — Torres, R., et al. (2012). GMES Sentinel-1 mission.
  *Remote Sensing of Environment*, 120, 9–24. https://doi.org/10.1016/j.rse.2011.05.028
- **Sentinel-2** (COPERNICUS/S2_SR_HARMONIZED, Scene Classification Layer) — Drusch, M., et al.
  (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote
  Sensing of Environment*, 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026
- **MODIS MCD43A4 v061** (NBAR) — Schaaf, C., & Wang, Z. (2021). MODIS/Terra+Aqua BRDF/Albedo Nadir
  BRDF Adjusted Ref Daily L3 Global 500m V061. NASA LP DAAC. https://doi.org/10.5067/MODIS/MCD43A4.061
- **MODIS MOD13Q1 v061** (paper's NDVI/EVI) — Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day
  L3 Global 250m SIN Grid V061. NASA LP DAAC. https://doi.org/10.5067/MODIS/MOD13Q1.061
- **MODIS MCD15A3H v061** (LAI) — Myneni, R., Knyazikhin, Y., & Park, T. (2021). MODIS/Terra+Aqua
  Leaf Area Index/FPAR 4-day L4 Global 500m SIN Grid V061. NASA LP DAAC.
  https://doi.org/10.5067/MODIS/MCD15A3H.061
- **MODIS MOD11A1 / MYD11A1 v061** (LST) — Wan, Z., Hook, S., & Hulley, G. (2021). MODIS/Terra (Aqua)
  Land Surface Temperature/Emissivity Daily L3 Global 1km SIN Grid V061. NASA LP DAAC.
  https://doi.org/10.5067/MODIS/MOD11A1.061 ; https://doi.org/10.5067/MODIS/MYD11A1.061
- **CHIRPS** — Funk, C., et al. (2015). The climate hazards infrared precipitation with stations — a new
  environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
  https://doi.org/10.1038/sdata.2015.66
- **ERA5-Land** (ECMWF/ERA5_LAND/DAILY_AGGR) — Muñoz-Sabater, J., et al. (2021). ERA5-Land: a
  state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13,
  4349–4383. https://doi.org/10.5194/essd-13-4349-2021
- **SMAP L4** (NASA/SMAP/SPL4SMGP/008) — Reichle, R. H., et al. (2019). Version 4 of the SMAP Level-4
  soil moisture algorithm and data product. *Journal of Advances in Modeling Earth Systems*, 11,
  3106–3130. https://doi.org/10.1029/2019MS001729 — dataset: SMAP L4 Global 3-hourly 9 km EASE-Grid
  Surface and Root Zone Soil Moisture Geophysical Data, Version 8, NSIDC DAAC *(verify DOI)*.
- **SMAP L3 Enhanced** (NASA/SMAP/SPL3SMP_E/005) — O'Neill, P. E., et al. (2021). SMAP Enhanced L3
  Radiometer Global and Polar Grid Daily 9 km EASE-Grid Soil Moisture, Version 5. NSIDC DAAC.
  https://doi.org/10.5067/4DQ54OUIJ9DL *(verify DOI)*
- **SoilGrids 2.0** — Poggio, L., et al. (2021). SoilGrids 2.0: producing soil information for the globe
  with quantified spatial uncertainty. *SOIL*, 7, 217–240. https://doi.org/10.5194/soil-7-217-2021
- **SRTM** (USGS/SRTMGL1_003) — Farr, T. G., et al. (2007). The Shuttle Radar Topography Mission.
  *Reviews of Geophysics*, 45, RG2004. https://doi.org/10.1029/2005RG000183
- **MERIT Hydro** — Yamazaki, D., et al. (2019). MERIT Hydro: a high-resolution global hydrography map
  based on latest topography dataset. *Water Resources Research*, 55, 5053–5073.
  https://doi.org/10.1029/2019WR024873
- **ESA WorldCover v200** — Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo.
  https://doi.org/10.5281/zenodo.7254221
- **POLARIS v1.0** (v3/v4 soil properties, 30 m; read over HTTP from
  http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0, not through Earth Engine) — Chaney, N. W.,
  Minasny, B., Herman, J. D., Nauman, T. W., Brungard, C. W., Morgan, C. L. S., McBratney, A. B., Wood,
  E. F., & Yimam, Y. (2019). POLARIS soil properties: 30-m probabilistic maps of soil properties over
  the contiguous United States. *Water Resources Research*, 55, 2916–2938.
  https://doi.org/10.1029/2018WR022797 *(verify DOI)*
- **NRCS AWDB station metadata** (exact SCAN coordinates for the v4 POLARIS re-extraction) — AWDB REST
  API `/stations` endpoint, network SCAN, accessed 2026-11-22.
- 2015–2018 package data (earlier experiments) — kashingtonDC/RZSM, GitHub repository
  (https://github.com/kashingtonDC/RZSM).

## Methods
- Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic
  matter for hydrologic solutions. *Soil Science Society of America Journal*, 70, 1569–1578.
  https://doi.org/10.2136/sssaj2005.0117
- Entekhabi, D., Reichle, R. H., Koster, R. D., & Crow, W. T. (2010). Performance metrics for soil
  moisture retrievals and application requirements. *Journal of Hydrometeorology*, 11, 832–840.
  https://doi.org/10.1175/2010JHM1223.1 (ubRMSE)
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale
  ecological mapping models. *Nature Communications*, 11, 4540. https://doi.org/10.1038/s41467-020-18321-y
- Li, X., Grandvalet, Y., & Davoine, F. (2018). Explicit inductive bias for transfer learning with
  convolutional networks (L2-SP). *ICML 2018*, PMLR 80, 2825–2834.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.
- Karl, T. R., & Koss, W. J. (1984). Regional and national monthly, seasonal, and annual temperature
  weighted by area, 1895–1983. *Historical Climatology Series 4-3*, NOAA NCDC. (climate regions)
