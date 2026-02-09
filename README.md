# The Blue Spine: Spatiotemporal Analysis of Urban Cooling Island (UCI) Intensity in Tianjin
### Quantifying the Micro-climatic Regulation of the Haihe River (2020–2025)

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Overview

Tianjin, a megacity in Northern China, faces intensifying **Urban Heat Island (UHI)** effects due to rapid urbanization and surface sealing. The Haihe River, as the city's "Blue Spine," plays a critical role in micro-climatic regulation, yet its thermodynamic interaction with the surrounding urban fabric remains spatially heterogeneous.

This project implements a **multi-temporal composite analysis** using Landsat 8/9 OLI/TIRS imagery, aggregating five years of data (2020–2025) into **12 representative monthly baselines**. Through **Geographically Weighted Regression (GWR)** and spatial autocorrelation analysis, we quantify the seasonal fluctuations in **Urban Cooling Island (UCI)** intensity and identify the **Threshold Value of Efficiency (TVoE)** for urban planning applications.

---

## Key Findings

### Haihe River Cooling Effect (Riverside Corridor: 0-1500m)

| Season | LST Near River (0-300m) | LST Far (750-1500m) | Cooling Intensity | Model R² |
|--------|------------------------|---------------------|-------------------|----------|
| **Summer (Jun-Aug)** | 38.65°C | 41.27°C | **−2.63°C** | 0.659 |
| **Spring (Mar-May)** | 26.34°C | 28.60°C | −2.26°C | 0.649 |
| **Autumn (Sep-Nov)** | 22.10°C | 23.22°C | −1.12°C | 0.530 |
| **Winter (Dec-Feb)** | 5.63°C | 6.32°C | −0.69°C | 0.577 |

### Multivariate GWR Coefficient Interpretation (Summer, 0-300m from river)

| Variable | Coefficient | Interpretation |
|----------|-------------|----------------|
| **Distance to River** | +32.70 | Strong positive effect — farther from river = higher LST |
| **NDBI (Built-up Index)** | +3.60 | Built-up density significantly increases LST |
| **NDVI (Vegetation Index)** | +1.55 | Vegetation effect varies spatially |

### Principal Conclusions

1. The Haihe River provides significant cooling, with **summer cooling reaching 2.63°C** within 300m of the riverbank.
2. The cooling effect follows a **logarithmic distance-decay pattern**, with the strongest influence within **0-500m** and a transition zone at **500-750m**.
3. **Built-up density (NDBI)** is the dominant factor affecting LST near the river, contributing ~45% of explained variance in summer.
4. Seasonal variation is pronounced: **summer cooling is 4× stronger than winter**.
5. The multivariate GWR model achieves **R² = 0.57–0.66** in the riverside corridor.

---

## Methodology

### Data Acquisition
- **Satellite:** Landsat 8/9 Collection 2 Level-2
- **Temporal Range:** 2020-01-01 to 2025-12-31
- **Compositing Strategy:** Monthly median (5-year aggregation per month)
- **Resolution:** 30m | **CRS:** EPSG:32650 (UTM 50N)

### Variable Retrieval

| Variable | Formula | Purpose |
|----------|---------|---------|
| **LST** | Single-channel algorithm (TIRS Band 10) | Land Surface Temperature |
| **NDWI** | (Green − NIR) / (Green + NIR) | Water body extraction |
| **NDVI** | (NIR − Red) / (NIR + Red) | Vegetation density |
| **NDBI** | (SWIR − NIR) / (SWIR + NIR) | Built-up density |

### Spatial Analysis

1. **Buffer Analysis:** Multi-ring buffers (30m–1500m) for LST gradient extraction
2. **Geographically Weighted Regression (GWR):**
   - Single-variable: LST ~ Distance
   - Multivariate: LST ~ Distance + NDVI + NDBI
   - Kernel: Gaussian (bandwidth = 500m)
3. **Spatial Autocorrelation:** Global Moran's I, Local Moran's I (LISA), Getis-Ord Gi*
4. **Seasonal Analysis:** Sinusoidal model fitting, phase analysis

---

## Repository Structure

```
Tianjin_Haihe_Cooling/
├── Data/
│   ├── Raw_TIF/                    # GEE monthly composites (v1: 2-band, v2: 4-band)
│   ├── Processed/Month_XX/         # Extracted single-band TIFs
│   ├── Vector/                     # Haihe_River.shp, buffer shapefiles
│   ├── GWR_Results/                # Single-variable GWR sample data
│   ├── GWR_Multivariate/           # Multi-variable GWR outputs (12 months)
│   └── Spatial_Stats/              # Moran's I & LISA results
├── Scripts/
│   ├── config.py                   # Shared configuration (paths, constants)
│   ├── 00 GEE_data_acquisition.js  # Google Earth Engine export
│   ├── 01 preprocessing.py         # Band extraction & water masking
│   ├── 02 LST retrieval.py         # Buffer analysis & zonal statistics
│   ├── 03 GWR analysis.py          # Single-variable GWR
│   ├── 04 spatial_autocorrelation.py
│   ├── 05 seasonal_analysis.py
│   ├── 06 multivariate_GWR.py      # LST ~ Distance + NDVI + NDBI
│   └── 07 riverside_analysis.py    # Corridor analysis (0-1500m)
├── Maps/
│   ├── Buffer_Analysis/            # Cooling gradient charts (Script 02)
│   ├── GWR_SingleVar/              # Local regression maps (Script 03)
│   ├── GWR_Multivariate/           # Coefficient distribution maps
│   ├── Spatial_Autocorrelation/    # LISA & hot spot maps
│   ├── Seasonal_Analysis/          # Temporal pattern charts
│   └── Riverside_Analysis/         # Corridor-focused visualizations
├── Docs/
│   └── OPERATION_GUIDE.md
├── requirements.txt
└── README.md
```

---

## Study Area

**Location:** Tianjin, China — 6 Central Districts  
(Heping, Nankai, Hexi, Hedong, Hebei, Hongqiao)

```
Bounding Box (WGS84):
  Northwest: 116.9528°E, 39.3504°N
  Southeast: 117.8853°E, 38.8987°N
```

---

## References

1. Yang, B., et al. (2015). "The Impact Analysis of Water Body Landscape Pattern on Urban Heat Island: A Case Study of Wuhan City."
2. Wang, Z., et al. (2020). "A Geographically Weighted Regression Approach to Understanding Urbanization Impacts... Las Vegas."
3. Wang, L., et al. (2023). "The Regulating Effect of Urban Large Planar Water Bodies on Residential Heat Islands: A Case Study of Meijiang Lake in Tianjin."
4. Jiang, Y., et al. (2021). "Interaction of Urban Rivers and Green Space Morphology to Mitigate the Urban Heat Island Effect."
5. Du, H. & Zhou, X. (2022). "Distance-decay models for urban blue space cooling effects."

---

## Author

**Congyuan Zheng (Othello)**  
University of Colorado Boulder  
Department of Geography & Applied Mathematics  
GEOG 4503: GIS Project Management

---

## License

This project is licensed under the MIT License.

