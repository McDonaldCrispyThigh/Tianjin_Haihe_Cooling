# 🌊 The Blue Spine: Spatiotemporal Analysis of Urban Cooling Island Intensity

## 📖 Project Overview
Tianjin, a megacity in Northern China, faces intensifying Urban Heat Island (UHI) effects due to rapid urbanization. While the Haihe River ("The Mother River") acts as a critical blue space, its specific cooling capacity remains unquantified for recent years.

This project utilizes **Landsat 8/9 satellite imagery (2020–2025)** to quantify the micro-climatic regulation of the Haihe River. Instead of traditional global statistics, this repository implements an automated **Python workflow** and **Geographically Weighted Regression (GWR)** to model the spatial heterogeneity of the cooling effect, providing evidence-based metrics for urban resilience planning.

---

## 🚀 Key Features (Technical Highlights)

- **Automated LST Retrieval**  
  A custom Python script (using ArcPy) to batch-process Level-2 Landsat thermal bands, converting DN values to Celsius and applying cloud masking.

- **Spatial Heterogeneity Modeling**  
  Implementation of **GWR (Geographically Weighted Regression)** to identify local variations in the cooling effect, overcoming the limitations of **OLS (Ordinary Least Squares)**.

- **Water Body Extraction**  
  Automated **NDWI** calculation to dynamically delineate the Haihe River boundary across different years.

- **Threshold Calculation**  
  Statistical analysis to determine the maximum **Cooling Threshold Distance** (e.g., how many meters inland the cooling effect reaches).

---

## 🛠️ Methodology & Workflow

### Data Acquisition
- **Source:** USGS EarthExplorer (Landsat 8/9 OLI/TIRS Level-2 Science Products)  
- **Timeframe:** Summer months (June–September), 2020–2025  
- **Study Area:** The 6 central districts of Tianjin (Heping, Nankai, Hexi, etc.)

### Preprocessing (Python / ArcPy)
- Batch cloud masking using the `QA_PIXEL` band  
- Geometric clipping to the study area boundary  

### Analysis (Spatial Statistics)
- **LST Retrieval:**  
  \[
  Temperature = (ST\_B10 \times 0.0034172 + 149.0) - 273.15
  \]

- **GWR Modeling:**  
  - Dependent Variable: LST  
  - Explanatory Variable: Euclidean Distance to River  

### Visualization
- Generation of Cooling Coefficient Maps  
- Generation of Threshold Curves  

---

## 📂 Repository Structure

```text
Tianjin_Haihe_Cooling/
Tianjin_Haihe_Cooling/
├── Data/                 # All GIS Data (Ignored by Git)
├── Scripts/              # Python code (.py or .ipynb)
│   ├── 01_preprocessing.py
│   ├── 02_LST_retrieval.py
│   └── 03_GWR_analysis.py
├── Maps/                 # Final output maps (JPG, PDF) - Deliverables [5]
├── Docs/                 # Course assignment documents
│   ├── Vision_Statement.pdf
│   ├── Gantt_Chart.xlsx
│   └── Final_Report.docx
├── .gitignore            # Git ignore configuration
└── README.md             # Project documentation (Portfolio landing page)
```


---

## 📊 Preliminary Results (Coming Soon)

- LST Distribution Map (2025) – *[Placeholder]*  
- GWR Coefficient Map – *[Placeholder]*  
- Cooling Distance Threshold Chart – *[Placeholder]*  

---

## 💻 Requirements

- **ArcGIS Pro 3.0+** (Advanced License required for Spatial Statistics tools)  
- **Python 3.x** (Standard ArcPy environment)  
- **Storage Space:** ~5GB for raw satellite imagery  

---

## 👤 Author

**Congyuan Zheng**  
University of Colorado Boulder  
GEOG 4503: GIS Project Management  
**Focus:** Remote Sensing, Spatial Statistics, Urban Resilience
