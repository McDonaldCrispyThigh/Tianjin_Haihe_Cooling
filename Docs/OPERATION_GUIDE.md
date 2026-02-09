# The Blue Spine - 完整操作指南
## Tianjin Haihe Cooling Project - Step-by-Step Guide

---

## 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [Phase 0: GEE 数据获取](#phase-0-gee-数据获取)
4. [Phase 1: 数据预处理](#phase-1-数据预处理)
5. [Phase 2: LST 缓冲区分析](#phase-2-lst-缓冲区分析)
6. [Phase 3: GWR 单变量回归](#phase-3-gwr-单变量回归)
7. [Phase 4: 空间自相关分析](#phase-4-空间自相关分析)
8. [Phase 5: 季节性时序分析](#phase-5-季节性时序分析)
9. [Phase 6: 多变量 GWR](#phase-6-多变量-gwr)
10. [Phase 7: 河岸廊道分析](#phase-7-河岸廊道分析)
11. [GitHub 版本控制](#github-版本控制)

---

## 项目概述

### 当前进度

| Phase | 脚本 | 状态 | 说明 |
|-------|------|------|------|
| 0 | `00 GEE_data_acquisition.js` | ✅ 完成 | GEE 数据导出 (v1 + v2) |
| 1 | `01 preprocessing.py` | ✅ 完成 | 波段提取、水体掩膜 |
| 2 | `02 LST retrieval.py` | ✅ 完成 | 多环缓冲区、分区统计 |
| 3 | `03 GWR analysis.py` | ✅ 完成 | 单变量 GWR 分析 |
| 4 | `04 spatial_autocorrelation.py` | ✅ 完成 | Moran's I, LISA, Gi* |
| 5 | `05 seasonal_analysis.py` | ✅ 完成 | 正弦模型、季节聚合 |
| 6 | `06 multivariate_GWR.py` | ✅ 完成 | 多变量 GWR (LST ~ Distance + NDVI + NDBI) |
| 7 | `07 riverside_analysis.py` | ✅ 完成 | 河岸廊道 0-1500m 聚焦分析 |

---

## 环境配置

### 系统要求
- **Python 3.10+**（推荐 3.13）
- **macOS / Windows / Linux** 均可
- **Google Earth Engine 账号**（用于数据下载）

### 虚拟环境设置

```bash
# 1. 进入项目目录
cd /path/to/Tianjin_Haihe_Cooling

# 2. 创建虚拟环境
python3 -m venv Tianjin

# 3. 激活虚拟环境
source Tianjin/bin/activate     # macOS/Linux
# Tianjin\Scripts\activate      # Windows

# 4. 安装依赖
pip install -r requirements.txt
```

### 依赖清单
```
numpy, pandas, rasterio, geopandas, shapely, scipy,
matplotlib, openpyxl, libpysal, esda, splot
```

### 运行脚本
```bash
# 确保在项目根目录运行（config.py 依赖相对路径）
cd /path/to/Tianjin_Haihe_Cooling

# 按顺序运行
python "Scripts/01 preprocessing.py"
python "Scripts/02 LST retrieval.py"
python "Scripts/03 GWR analysis.py"
python "Scripts/04 spatial_autocorrelation.py"
python "Scripts/05 seasonal_analysis.py"
python "Scripts/06 multivariate_GWR.py"
python "Scripts/07 riverside_analysis.py"
```

> **注意**：所有脚本共享 `Scripts/config.py` 配置文件。路径、参数、月份常量统一在此管理。

---

## Phase 0: GEE 数据获取

### 脚本：`Scripts/00 GEE_data_acquisition.js`

### 运行方式
1. 打开 [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. 粘贴脚本内容
3. 点击 Run → 在 Tasks 面板逐个点击 Run 导出
4. 从 Google Drive 下载到 `Data/Raw_TIF/`

### 数据版本

| 版本 | 文件名格式 | 波段数 | 用途 |
|------|-----------|--------|------|
| v1 | `Tianjin_Monthly_Median_XX.tif` | 2 (LST, NDWI) | Scripts 01-05 |
| v2 | `Tianjin_Monthly_v2_XX.tif` | 4 (LST, NDVI, NDBI, NDWI) | Scripts 06-07 |

---

## Phase 1: 数据预处理

### 脚本：`Scripts/01 preprocessing.py`

### 功能
- 验证 12 个月 TIF 数据完整性
- 提取 LST (Band 1) 和 NDWI (Band 2)
- 基于 NDWI 阈值 (0.1) 生成水体二值掩膜

### 输出
```
Data/Processed/Month_XX/    → LST_XX.tif, NDWI_XX.tif, Water_Binary_XX.tif
Data/Vector/                → Water_Polygon_07.shp (7月参考)
```

---

## Phase 2: LST 缓冲区分析

### 脚本：`Scripts/02 LST retrieval.py`

### 功能
- 以 `Haihe_River.shp` 为中心创建多环缓冲区 (30m-1000m)
- 计算各环带平均 LST → 提取冷却梯度
- 确定 TVoE（冷却效率阈值距离）

### 输出
```
Data/Vector/Haihe_Buffers_Analysis.shp     → 多环缓冲区矢量
Data/Gradient_Month_XX.xlsx                → 各月梯度数据
Data/All_Months_Gradient.xlsx              → 汇总
Maps/Buffer_Analysis/                      → 梯度曲线图、散点图
```

---

## Phase 3: GWR 单变量回归

### 脚本：`Scripts/03 GWR analysis.py`

### 功能
- 创建 100m 间距采样网格 (~27,500 点/月)
- 全局 OLS 回归：LST ~ Distance
- 局部加权回归（GWR-like）：空间变化的系数
- 高斯核，带宽 500m

### 输出
```
Data/GWR_Results/GWR_Samples_XX.csv        → 采样点数据
Maps/GWR_SingleVar/                        → 回归散点图、系数图、R²图
```

---

## Phase 4: 空间自相关分析

### 脚本：`Scripts/04 spatial_autocorrelation.py`

### 功能
- **Global Moran's I**：检测 LST 是否存在空间聚类
- **Local Moran's I (LISA)**：识别 HH/LL/HL/LH 聚类
- **Getis-Ord Gi\***：热点/冷点统计显著性

### 输出
```
Data/Spatial_Stats/Spatial_Stats_XX.csv/.shp  → 各月结果
Data/Spatial_Stats/Spatial_Autocorrelation_Summary.csv
Maps/Spatial_Autocorrelation/                 → Moran散点图、LISA聚类图、Gi*热点图
```

---

## Phase 5: 季节性时序分析

### 脚本：`Scripts/05 seasonal_analysis.py`

### 功能
- 提取 12 个月 ΔT、TVoE、梯度斜率
- 拟合正弦模型：ΔT(t) = A·sin(2πt/12 + φ) + C
- 计算变异系数 (CV) 评价降温稳定性
- 按四季聚合

### 输出
```
Data/Seasonal_Metrics_Summary.csv
Maps/Seasonal_Analysis/         → 季节周期图、箱线图、相位分析图
```

---

## Phase 6: 多变量 GWR

### 脚本：`Scripts/06 multivariate_GWR.py`

### 功能
- **模型**：LST = β₀ + β₁(Distance) + β₂(NDVI) + β₃(NDBI) + ε
- 需要 v2 数据（4 波段）
- 各系数空间可视化
- 与全局 OLS 对比

### 输出
```
Data/GWR_Multivariate/GWR_Multivariate_XX.csv/.shp
Data/GWR_Multivariate/GWR_Multivariate_Summary.csv
Maps/GWR_Multivariate/        → 系数空间分布图、月度对比图
```

---

## Phase 7: 河岸廊道分析

### 脚本：`Scripts/07 riverside_analysis.py`

### 功能
- 聚焦河岸 0-1500m 内的 GWR 系数
- 按距离带 (0-100, 100-200, ..., 1000-1500m) 分析变量影响力
- 季节性对比

### 输出
```
Maps/Riverside_Analysis/    → 廊道系数图、变量重要性图、季节对比图
```

---

## 项目文件结构

```
Tianjin_Haihe_Cooling/
├── Data/
│   ├── Raw_TIF/                    # GEE 原始 TIF (v1 + v2)
│   ├── Processed/Month_XX/         # 单波段提取结果
│   ├── Vector/                     # Haihe_River.shp, Buffers
│   ├── GWR_Results/                # 单变量 GWR 采样数据
│   ├── GWR_Multivariate/           # 多变量 GWR 结果
│   ├── Spatial_Stats/              # 空间自相关结果
│   ├── Gradient_Month_XX.xlsx      # 各月梯度数据
│   └── Seasonal_Metrics_Summary.csv
│
├── Maps/
│   ├── Buffer_Analysis/            # 缓冲区梯度图 (Script 02)
│   ├── GWR_SingleVar/              # 单变量 GWR 图 (Script 03)
│   ├── Spatial_Autocorrelation/    # Moran, LISA, Gi* (Script 04)
│   ├── Seasonal_Analysis/          # 季节性分析图 (Script 05)
│   ├── GWR_Multivariate/           # 多变量系数图 (Script 06)
│   └── Riverside_Analysis/         # 河岸廊道分析 (Script 07)
│
├── Scripts/
│   ├── config.py                   # ★ 共享配置（路径、常量、参数）
│   ├── 00 GEE_data_acquisition.js
│   ├── 01 preprocessing.py
│   ├── 02 LST retrieval.py
│   ├── 03 GWR analysis.py
│   ├── 04 spatial_autocorrelation.py
│   ├── 05 seasonal_analysis.py
│   ├── 06 multivariate_GWR.py
│   └── 07 riverside_analysis.py
│
├── Docs/
│   └── OPERATION_GUIDE.md          # 本文件
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## GitHub 版本控制

### .gitignore 策略
以下文件/文件夹 **不会** 上传到 GitHub：
- `*.tif` — 栅格数据（太大）
- `*.gdb/` — ArcGIS 地理数据库
- `*.aprx`, `*.atbx` — ArcGIS 工程文件
- `Tianjin/` — Python 虚拟环境
- `__pycache__/` — Python 缓存

### 常用 Git 命令

```bash
# 查看状态
git status

# 添加并提交
git add -A
git commit -m "Update: description of changes"

# 推送到 GitHub
git push origin main

# 拉取最新
git pull origin main
```

---

## 常见问题

### Q: 脚本报错 `ModuleNotFoundError`？
**A:** 确保激活了虚拟环境：`source Tianjin/bin/activate`

### Q: matplotlib 报 UTF-8 错误？
**A:** macOS 上删除 AppleDouble 文件：`find Tianjin/ -name "._*" -delete`

### Q: GWR 运行很慢？
**A:** 在 `config.py` 中增大 `SAMPLE_SPACING_GWR`（如 100 → 200），减少采样点

### Q: 多变量 GWR 找不到数据？
**A:** 需要 v2 数据。在 GEE 运行更新后的脚本导出 `Tianjin_Monthly_v2_XX.tif`

### Q: GitHub 推送被拒绝？
**A:** 可能有大文件未被忽略，检查 `git status` 和 `.gitignore`

---

**Congyuan Zheng | CU Boulder | GEOG 4503**
