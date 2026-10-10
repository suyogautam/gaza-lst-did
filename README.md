# Land surface warming after tree-cover and building loss in the Gaza Strip

Earth Engine code and main result tables for the manuscript *Land surface warming after tree-cover and building loss in the Gaza Strip: a dose–response analysis with Landsat and ECOSTRESS* (Suyog Gautam, University of Iowa).

This repository does not hold the extracted satellite data, the analysis notebooks or the figures. To reproduce the work, pull the raw data yourself with the two scripts below and build the analysis from the methods section of the paper.

## Contents

```
gee/01_extract_landsat.py     grid, static layers and monthly Landsat 8/9 composites
gee/02_extract_ecostress.py   ECOSTRESS overpasses on the same grid (run after 01)
results/                      main result tables (CSV, temperatures in °C)
```

## Extracting the data

```bash
pip install -r requirements.txt
python gee/01_extract_landsat.py
python gee/02_extract_ecostress.py
```

Run both from the repository root. You need a Google Earth Engine account and a cloud project; put the project name in `CFG['PROJECT']` at the top of each script. Pixels come straight into Python through `ee.data.computePixels`, so nothing goes through Google Drive. Files already on disk are reused, so a stopped run can be restarted.

`01_extract_landsat.py` writes to `data/`:

| File | Content |
|---|---|
| `grid.json` | 100 m grid, UTM 36N (EPSG:32636), Gaza Strip plus a 25 km buffer |
| `static_raw.npz` | ESA WorldCover 2021 class fractions, SRTM elevation and FAO GAUL country code per cell |
| `cells.pkl` | candidate cells: all Gaza land cells and eligible Israeli and Egyptian land cells, with strata |
| `monthly/YYYY_MM.npz` | per cell and month, Oct 2018 – Sep 2026: LST, NDVI, albedo, the emissivity assumed by the Landsat ST product, clear-observation count and coverage |

Landsat pixels are masked with QA_PIXEL (fill, dilated cloud, cirrus, cloud, cloud shadow, snow, water), QA_RADSAT and ST_QA ≤ 3 K.

`02_extract_ecostress.py` writes to `data/ecostress/`:

| File | Content |
|---|---|
| `inventory.csv` | every ECOSTRESS tile over the region with its time and the share of Gaza that is clear |
| `selected_overpasses.csv` | dry-season (May–Oct) overpasses at 09:00–11:30 and 21:00–05:00 local solar time with ≥ 30% of Gaza clear |
| `scenes/pass_NNNNN.npz` | per overpass and cell: LST, wide-band emissivity and the share of valid 70 m pixels |

Values in the `.npz` files are int16 with fixed scale factors (described at the top of each script).

### Damage data

The building-damage dose comes from the UNOSAT *Gaza Strip Comprehensive Damage Assessment*, imagery of 11 October 2025 (product 4213). It is not redistributed here. Download it from <https://unosat.org/products/4213>. The per-cell doses used in the paper are in `results/analysis_cells.csv`.

## Result tables

Effects are imputation difference-in-differences estimates in °C against matched land in Israel and Egypt, with the recent pre-war baseline (Oct 2020 – Sep 2023). Unless a file has a `season` column, they are for the dry season (May–Oct). `lo` and `hi` are 95% block-bootstrap intervals (999 draws, 2 km blocks). Dose bins `Q1`–`Q4` are quartiles of dose among treated Gaza cells in each stratum, with the dose range in the label; `none` is Gaza cells below the treatment threshold. Built-up, mixed and bare/sand cells use the UNOSAT building dose (destroyed-equivalent structures per ha); tree-cover and cropland cells use NDVI loss.

| File | Content | In the paper |
|---|---|---|
| `analysis_cells.csv` | one row per cell in the analysis: location, land cover, stratum, dose, dose bin, match status | Table 2 |
| `imputation_dose_season.csv` | effect by stratum and dose bin, dry and wet season | Table 3, Fig. 6 |
| `imputation_contrast_season.csv` | effect of each dose bin minus the `none` bin of the same stratum | Table 3 |
| `dose_slope_permutation.csv` | dose slope and block-permutation p value | Table 3 |
| `imputation_since_q.csv` | effect by quarter since October 2023 | Fig. 7 |
| `imputation_moy.csv` | effect by calendar month | Fig. 8 |
| `imputation_dose_wy_dry.csv` | effect by war-year (Oct–Sep) | Table A.2 |
| `event_study_wy_dry.csv` | two-way fixed-effects event study by war-year, pre-war years included (intervals clustered by 2 km block, not bootstrapped) | Fig. 5, Table A.1 |
| `cell_effects_dry.csv` | per-cell post-war effect | Fig. 9 |
| `ecostress_{morning,night}_lst.csv` | ECOSTRESS LST effect by stratum and dose bin | Table 6 |
| `ecostress_{morning,night}_lst_con.csv` | same, minus the `none` bin | Table 6 |
| `ecostress_morning_lst_twoway.csv`, `ecostress_morning_lst_con_twoway.csv` | morning intervals that also resample overpasses | Table 6 |
| `ecostress_{morning,night}_emis.csv` | change in ECOSTRESS wide-band emissivity | Table 6 |

Robustness checks, placebo tests, balance tables and mechanism results are reported in the paper and are not included here.

## Data sources

- Landsat 8–9 OLI/TIRS Collection 2 Level-2: Earth Resources Observation and Science (EROS) Center (2020), U.S. Geological Survey, doi:10.5066/P9OGBGM6
- ECOSTRESS: Hook, S., Hulley, G. (2022). ECOSTRESS Tiled Land Surface Temperature and Emissivity Instantaneous L2 Global 70 m V002. NASA LP DAAC, doi:10.5067/ECOSTRESS/ECO_L2T_LSTE.002
- ESA WorldCover 10 m 2021 v200: Zanaga et al. (2022), doi:10.5281/zenodo.7254221
- SRTM: Farr et al. (2007), doi:10.1029/2005RG000183
- FAO GAUL 2015 (Google Earth Engine `FAO/GAUL/2015/level0`)
- UNITAR–UNOSAT (2025). Gaza Strip Comprehensive Damage Assessment, imagery of 11 October 2025, product 4213

All Earth Engine data were accessed in October 2026.

## Licence

Code: MIT (see `LICENSE`). Result tables follow the terms of the data sources above.
