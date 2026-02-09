# The Blue Spine - Complete Operation Guide
## Tianjin Haihe Cooling Project - Step-by-Step Guide

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Phase 0: GEE Data Acquisition](#phase-0-gee-data-acquisition)
4. [Phase 1: Data Preprocessing](#phase-1-data-preprocessing)
5. [Phase 2: LST Buffer Analysis](#phase-2-lst-buffer-analysis)
6. [Phase 3: GWR Univariate Regression](#phase-3-gwr-univariate-regression)
7. [Phase 4: Spatial Autocorrelation Analysis](#phase-4-spatial-autocorrelation-analysis)
8. [Phase 5: Seasonal Time-Series Analysis](#phase-5-seasonal-time-series-analysis)
9. [Phase 6: Multivariate GWR](#phase-6-multivariate-gwr)
10. [Phase 7: Riverside Corridor Analysis](#phase-7-riverside-corridor-analysis)
11. [GitHub Version Control](#github-version-control)

---

## Project Overview

### Current Progress

| Phase | Script | Status | Description |
|-------|--------|--------|-------------|
| 0 | `00 GEE_data_acquisition.js` | Done | GEE data export (v1 + v2) |
| 1 | `01 preprocessing.py` | Done | Band extraction, water mask |
| 2 | `02 LST retrieval.py` | Done | Multi-ring buffer, zonal statistics |
| 3 | `03 GWR analysis.py` | Done | Univariate GWR analysis |
| 4 | `04 spatial_autocorrelation.py` | Done | Moran's I, LISA, Gi* |
| 5 | `05 seasonal_analysis.py` | Done | Sinusoidal model, seasonal aggregation |
| 6 | `06 multivariate_GWR.py` | Done | Multivariate GWR (LST ~ Distance + NDVI + NDBI) |
| 7 | `07 riverside_analysis.py` | Done | Riverside corridor 0-1500m focused analysis |

---

## Environment Setup

### System Requirements
- **Python 3.10+** (recommended 3.13)
- **macOS / Windows / Linux**
- **Google Earth Engine account** (for data download)

### Virtual Environment Setup

```bash
# 1. Navigate to project directory
cd /path/to/Tianjin_Haihe_Cooling

# 2. Create virtual environment
python3 -m venv Tianjin

# 3. Activate virtual environment
source Tianjin/bin/activate     # macOS/Linux
# Tianjin\Scripts\activate      # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependency List
```
numpy, pandas, rasterio, geopandas, shapely, scipy,
matplotlib, openpyxl, libpysal, esda, splot
```

### Running Scripts
```bash
# Ensure running from project root (config.py uses relative paths)
cd /path/to/Tianjin_Haihe_Cooling

# Run in order
python "Scripts/01 preprocessing.py"
python "Scripts/02 LST retrieval.py"
python "Scripts/03 GWR analysis.py"
python "Scripts/04 spatial_autocorrelation.py"
python "Scripts/05 seasonal_analysis.py"
python "Scripts/06 multivariate_GWR.py"
python "Scripts/07 riverside_analysis.py"
```

> **Note**: All scripts share `Scripts/config.py`. Paths, parameters, and month constants are managed centrally there.

---

## Phase 0: GEE Data Acquisition

### Script: `Scripts/00 GEE_data_acquisition.js`

### How to Run
1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Paste the script content
3. Click Run -> Export each task from the Tasks panel
4. Download from Google Drive to `Data/Raw_TIF/`

### Data Versions

| Version | Filename Format | Bands | Usage |
|---------|----------------|-------|-------|
| v1 | `Tianjin_Monthly_Median_XX.tif` | 2 (LST, NDWI) | Scripts 01-05 |
| v2 | `Tianjin_Monthly_v2_XX.tif` | 4 (LST, NDVI, NDBI, NDWI) | Scripts 06-07 |

---

## Phase 1: Data Preprocessing

### Script: `Scripts/01 preprocessing.py`

### Features
- Validate 12-month TIF data completeness
- Extract LST (Band 1) and NDWI (Band 2)
- Generate water binary mask based on NDWI threshold (0.1)

### Output
```
Data/Processed/Month_XX/    -> LST_XX.tif, NDWI_XX.tif, Water_Binary_XX.tif
Data/Vector/                -> Water_Polygon_07.shp (July reference)
```

---

## Phase 2: LST Buffer Analysis

### Script: `Scripts/02 LST retrieval.py`

### Features
- Create multi-ring buffers (30m-1500m) centered on `Haihe_River.shp`
- Calculate mean LST per ring -> extract cooling gradient
- Determine TVoE (Threshold of Vanishing Effect)

### Output
```
Data/Vector/Haihe_Buffers_Analysis.shp     -> Multi-ring buffer vector
Data/Gradient_Month_XX.xlsx                -> Monthly gradient data
Data/All_Months_Gradient.xlsx              -> Combined summary
Maps/Buffer_Analysis/                      -> Gradient curves, scatter plots
```

---

## Phase 3: GWR Univariate Regression

### Script: `Scripts/03 GWR analysis.py`

### Features
- Create 100m-spacing sample grid (~27,500 points/month)
- Global OLS regression: LST ~ Distance
- Local weighted regression (GWR-like): spatially-varying coefficients
- Gaussian kernel, bandwidth 500m

### Output
```
Data/GWR_Results/GWR_Samples_XX.csv        -> Sample point data
Maps/GWR_SingleVar/                        -> Regression scatter plots, coefficient maps, R2 maps
```

---

## Phase 4: Spatial Autocorrelation Analysis

### Script: `Scripts/04 spatial_autocorrelation.py`

### Features
- **Global Moran's I**: Detect whether LST exhibits spatial clustering
- **Local Moran's I (LISA)**: Identify HH/LL/HL/LH clusters
- **Getis-Ord Gi\***: Hot/cold spot statistical significance

### Output
```
Data/Spatial_Stats/Spatial_Stats_XX.csv/.shp  -> Monthly results
Data/Spatial_Stats/Spatial_Autocorrelation_Summary.csv
Maps/Spatial_Autocorrelation/                 -> Moran scatter plots, LISA cluster maps, Gi* hotspot maps
```

---

## Phase 5: Seasonal Time-Series Analysis

### Script: `Scripts/05 seasonal_analysis.py`

### Features
- Extract 12-month Delta-T, TVoE, gradient slope
- Fit sinusoidal model: Delta-T(t) = A*sin(2*pi*t/12 + phi) + C
- Calculate coefficient of variation (CV) for cooling stability assessment
- Aggregate by four seasons

### Output
```
Data/Seasonal_Metrics_Summary.csv
Maps/Seasonal_Analysis/         -> Seasonal cycle plots, box plots, phase analysis plots
```

---

## Phase 6: Multivariate GWR

### Script: `Scripts/06 multivariate_GWR.py`

### Features
- **Model**: LST = B0 + B1(Distance) + B2(NDVI) + B3(NDBI) + e
- Requires v2 data (4 bands)
- Spatial visualization of each coefficient
- Comparison with global OLS

### Output
```
Data/GWR_Multivariate/GWR_Multivariate_XX.csv/.shp
Data/GWR_Multivariate/GWR_Multivariate_Summary.csv
Maps/GWR_Multivariate/        -> Coefficient spatial distribution maps, monthly comparison plots
```

---

## Phase 7: Riverside Corridor Analysis

### Script: `Scripts/07 riverside_analysis.py`

### Features
- Focus on GWR coefficients within 0-1500m of riverside
- Analyze variable influence by distance bands (0-100, 100-200, ..., 1000-1500m)
- Seasonal comparison

### Output
```
Maps/Riverside_Analysis/    -> Corridor coefficient maps, variable importance plots, seasonal comparison plots
```

---

## Project File Structure

```
Tianjin_Haihe_Cooling/
|-- Data/
|   |-- Raw_TIF/                    # GEE raw TIF (v1 + v2)
|   |-- Processed/Month_XX/         # Single-band extraction results
|   |-- Vector/                     # Haihe_River.shp, Buffers
|   |-- GWR_Results/                # Univariate GWR sample data
|   |-- GWR_Multivariate/           # Multivariate GWR results
|   |-- Spatial_Stats/              # Spatial autocorrelation results
|   |-- Gradient_Month_XX.xlsx      # Monthly gradient data
|   `-- Seasonal_Metrics_Summary.csv
|
|-- Maps/
|   |-- Buffer_Analysis/            # Buffer gradient plots (Script 02)
|   |-- GWR_SingleVar/              # Univariate GWR plots (Script 03)
|   |-- Spatial_Autocorrelation/    # Moran, LISA, Gi* (Script 04)
|   |-- Seasonal_Analysis/          # Seasonal analysis plots (Script 05)
|   |-- GWR_Multivariate/           # Multivariate coefficient plots (Script 06)
|   `-- Riverside_Analysis/         # Riverside corridor analysis (Script 07)
|
|-- Scripts/
|   |-- config.py                   # * Shared config (paths, constants, parameters)
|   |-- 00 GEE_data_acquisition.js
|   |-- 01 preprocessing.py
|   |-- 02 LST retrieval.py
|   |-- 03 GWR analysis.py
|   |-- 04 spatial_autocorrelation.py
|   |-- 05 seasonal_analysis.py
|   |-- 06 multivariate_GWR.py
|   `-- 07 riverside_analysis.py
|
|-- Docs/
|   `-- OPERATION_GUIDE.md          # This file
|-- .gitignore
|-- requirements.txt
|-- README.md
`-- LICENSE
```

---

## GitHub Version Control

### .gitignore Strategy
The following files/folders are **not** uploaded to GitHub:
- `*.tif` -- Raster data (too large)
- `*.gdb/` -- ArcGIS geodatabase
- `*.aprx`, `*.atbx` -- ArcGIS project files
- `Tianjin/` -- Python virtual environment
- `__pycache__/` -- Python cache

### Common Git Commands

```bash
# Check status
git status

# Add and commit
git add -A
git commit -m "Update: description of changes"

# Push to GitHub
git push origin main

# Pull latest
git pull origin main
```

---

## FAQ

### Q: Script throws `ModuleNotFoundError`?
**A:** Make sure the virtual environment is activated: `source Tianjin/bin/activate`

### Q: matplotlib UTF-8 error?
**A:** On macOS, delete AppleDouble files: `find Tianjin/ -name "._*" -delete`

### Q: GWR runs slowly?
**A:** Increase `SAMPLE_SPACING_GWR` in `config.py` (e.g. 100 -> 200) to reduce sample points

### Q: Multivariate GWR cannot find data?
**A:** v2 data is required. Run the updated GEE script to export `Tianjin_Monthly_v2_XX.tif`

### Q: GitHub push rejected?
**A:** Large files may not be ignored. Check `git status` and `.gitignore`

---

**Congyuan Zheng | CU Boulder | GEOG 4503**
