# The Blue Spine: Spatiotemporal Analysis of Urban Cooling Island (UCI) Intensity in Tianjin
### Quantifying the Micro-climatic Regulation of the Haihe River (2020–2025)

![Status](https://img.shields.io/badge/Status-Analysis%20Complete-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Overview

Tianjin, a megacity in Northern China, faces intensifying **Urban Heat Island (UHI)** effects due to rapid urbanization and surface sealing. While the Haihe River acts as the city's "Blue Spine," its thermodynamic interaction with the surrounding urban fabric remains dynamic and spatially heterogeneous.

**This project implements a multi-temporal composite analysis using Landsat 8/9 OLI/TIRS imagery.** To mitigate cloud contamination and isolate phenological patterns, the workflow aggregates five years of data (2020–2025) into **12 representative monthly baselines**. This approach allows for a robust quantification of the seasonal fluctuations in **Urban Cooling Island (UCI)** intensity and the **Threshold Value of Efficiency (TVoE)** without the bias of single-date anomalies.

---

## Current Progress (2026-02-04)

### ✅ Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | GEE data acquisition (12 monthly composites) | ✅ Complete |
| **Phase 1** | Preprocessing (band extraction, water masking) | ✅ Complete |
| **Phase 2** | Buffer analysis & zonal statistics (12 months) | ✅ Complete |
| **Phase 3** | GWR/Local regression analysis (12 months) | ✅ Complete |
| **Phase 4** | Visualization & chart generation | ✅ Complete |

### 📊 Key Outputs Generated
- 12 monthly LST gradient datasets (`Data/Gradient_Month_XX.xlsx`)
- 12 monthly GWR sample datasets (`Data/GWR_Samples_XX.csv`, ~27,500 points each)
- 60+ visualization charts (`Maps/`)
- Multi-ring buffer shapefiles (`Data/Vector/`)

---

## Key Scientific Objectives

1.  **Quantify Cooling Intensity:** Calculate the temperature difference ($\Delta T$) between the water body and the urban matrix.
2.  **Determine Threshold Distance:** Identify the maximum distance (buffer zone) where the Haihe River significantly mitigates LST, referencing the "distance-decay" models found in *Du & Zhou (2022)*.
3.  **Analyze Spatial Heterogeneity:** Use GWR to reveal how local urban morphology (e.g., building density, vegetation cover) interferes with the river's cooling propagation.
4.  **Evaluate Landscape Metrics:** (Optional Expansion) Assess how the shape index (LSI) and fragmentation of the water body influence thermal regulation.

---

## Methodology & Technical Workflow

This project integrates remote sensing inversion with spatial statistics.

### 1. Data Acquisition & Preprocessing
- **Data Source:** USGS EarthExplorer (Landsat 8/9 Collection 2 Level-2).
- **Temporal Scope: Multi-year Monthly Compositing (2020–2025).**
    Instead of a linear time series, images from the same month across five years are aggregated to create a **"Climatological Mean"** (e.g., Representative July = Median of July 2020, 2021... 2025).
    * *Rationale:* This strategy effectively eliminates cloud cover gaps (a major constraint in Tianjin) and highlights stable seasonal thermodynamic trends rather than transient weather events.
- **Study Area:** The 6 central districts of Tianjin (Heping, Nankai, Hexi, Hedong, Hebei, Hongqiao).
- **Preprocessing:**
  - Automated Cloud Masking using `QA_PIXEL`.
  - Geometric Clipping.
  - Radiometric Calibration (DN to Reflectance/Temperature).

### 2. Automated Variable Retrieval (Python/ArcPy)
Scripts are designed to batch-process the following indices:

* **LST (Land Surface Temperature):**
    Single-channel algorithm applied to TIRS Band 10:
    $$LST = (DN \times 0.0034172 + 149.0) - 273.15$$
    *(Corrected for emissivity using fractional vegetation cover method)*
* **NDWI (Normalized Difference Water Index):**
    Used for dynamic extraction of the Haihe River boundary.
    $$NDWI = \frac{Green - NIR}{Green + NIR}$$
* **Covariates (Control Variables):**
    - **NDVI** (Vegetation Density)
    - **NDBI** (Impervious Surface Density)

### 3. Spatial Statistical Modeling
- **Buffer Analysis:** Multi-ring buffers (e.g., 30m intervals up to 1000m) to extract mean LST gradients.
- **Geographically Weighted Regression (GWR):**
  Unlike global OLS models, GWR allows regression coefficients to vary across space, capturing the local impact of the river on LST.
  - **Dependent Variable:** LST
  - **Explanatory Variable:** Euclidean Distance to River ($D_{river}$)
  - **Kernel:** Adaptive Bi-square

---

## Repository Structure

```text
Tianjin_Haihe_Cooling/
├── Data/                          # GIS Datasets (Excluded via .gitignore)
│   ├── Raw_TIF/                   # GEE exported monthly composites (12 TIFs)
│   ├── Processed/                 # Extracted LST/NDWI bands by month
│   ├── Vector/                    # Shapefiles (Haihe_River, Buffers, etc.)
│   ├── GWR_Results/               # GWR output shapefiles
│   ├── Gradient_Month_XX.xlsx     # Zonal statistics per month
│   ├── GWR_Samples_XX.csv         # Regression sample points per month
│   └── All_Months_Gradient.xlsx   # Combined gradient summary
├── Scripts/                       # Processing Modules
│   ├── 00 GEE_data_acquisition.js # Google Earth Engine script
│   ├── 01 preprocessing.py        # Band extraction & water masking
│   ├── 02 LST retrieval.py        # Buffer analysis & zonal stats
│   └── 03 GWR analysis.py         # Spatial regression analysis
├── Maps/                          # Visualization Outputs (60+ charts)
│   ├── Cooling_Gradient_XX.png    # Distance-LST curves
│   ├── Local_Regression_XX.png    # GWR coefficient maps
│   ├── Local_R2_XX.png            # Model fit maps
│   └── Seasonal_Comparison_All.png
├── Docs/                          # Documentation
│   └── OPERATION_GUIDE.md         # Step-by-step workflow guide
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Preliminary Results & Hypotheses

*Based on literature review:*

* **Hypothesis 1:** The cooling effect of the Haihe River is non-linear and follows a logarithmic decay function.
* **Hypothesis 2:** The "Cooling Threshold Distance" is expected to range between **300m and 600m**, varying significantly by river width and adjacent building height.
* **Hypothesis 3:** Areas with higher NDVI (parks along the river) will show a synergistic cooling effect (Interaction of Blue-Green Space).

---

## Study Area

**Location:** Tianjin, China - 6 Central Districts (Heping, Nankai, Hexi, Hedong, Hebei, Hongqiao)

```
Bounding Box (WGS84):
  Northwest: 116.9528°E, 39.3504°N
  Southeast: 117.8853°E, 38.8987°N
  
Projected CRS: EPSG:32650 (WGS 84 / UTM zone 50N)
```

---

## Data Source & Acquisition

### Google Earth Engine Pipeline

The raw data was acquired using a custom GEE script (`Scripts/00 GEE_data_acquisition.js`):

| Parameter | Value |
|-----------|-------|
| **Satellite** | Landsat 8/9 Collection 2 Level-2 |
| **Time Range** | 2020-01-01 to 2025-12-31 |
| **Cloud Filter** | < 30% |
| **Compositing** | Monthly median (all years combined) |
| **Output Bands** | LST_Celsius, NDWI |
| **Resolution** | 30m |
| **CRS** | EPSG:32650 |

**Output:** 12 GeoTIFF files (`Tianjin_Monthly_Median_01.tif` ... `_12.tif`)
* **Hypothesis 3:** Areas with higher NDVI (parks along the river) will show a synergistic cooling effect (Interaction of Blue-Green Space).

---

## References & Literature Base

This project is grounded in the following key studies:

1. **Water Body Landscape Pattern:** *Yang, B., et al. (2015).* "The Impact Analysis of Water Body Landscape Pattern on Urban Heat Island: A Case Study of Wuhan City."
2. **GWR Application:** *Wang, Z., et al. (2020).* "A Geographically Weighted Regression Approach to Understanding Urbanization Impacts... Las Vegas."
3. **Tianjin Context:** *Wang, L., et al. (2023).* "The Regulating Effect of Urban Large Planar Water Bodies on Residential Heat Islands: A Case Study of Meijiang Lake in Tianjin."
4. **Blue-Green Interaction:** *Jiang, Y., et al. (2021).* "Interaction of Urban Rivers and Green Space Morphology to Mitigate the Urban Heat Island Effect."

---

## Author

**Congyuan Zheng (Othello)**
University of Colorado Boulder
*Department of Geography & Applied Mathematics*
**Course:** GEOG 4503: GIS Project Management
**Research Interests:** Remote Sensing, Spatial Statistics, Urban Resilience, Blue-Green Infrastructure.
