# 🚀 The Blue Spine - 完整操作指南
## Tianjin Haihe Cooling Project - Step-by-Step Guide

---

## 📋 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [Phase 1: 数据预处理](#phase-1-数据预处理)
4. [Phase 2: LST 反演与缓冲区分析](#phase-2-lst-反演与缓冲区分析)
5. [Phase 3: GWR 地理加权回归](#phase-3-gwr-地理加权回归)
6. [Phase 4: 可视化与出图](#phase-4-可视化与出图)
7. [Phase 5: GitHub 版本控制](#phase-5-github-版本控制)

---

## 项目概述

### 你已经完成了什么？
✅ GEE 代码运行，生成 12 个月度中位数合成影像  
✅ 数据已下载到本地 (`Data/Raw_TIF/`)  
✅ 海河矢量边界已创建 (`Haihe_River.shp`)  

### 接下来要做什么？
⏳ 运行 Python 脚本进行空间分析  
⏳ 完成 GWR 分析  
⏳ 同步到 GitHub  

---

## 环境配置

### 必需软件
- **ArcGIS Pro 3.0+** (需要 Advanced License 用于 GWR)
- **Python 3.x** (使用 ArcGIS Pro 自带的 Python 环境)

### 脚本运行方式

⚠️ **重要**：这些脚本使用 `arcpy` 模块，**必须在 ArcGIS Pro 环境中运行**！

**方法 1：ArcGIS Pro Python 窗口**
1. 打开 ArcGIS Pro
2. 点击 `Analysis` → `Python` → `Python Window`
3. 使用 `exec(open('脚本路径').read())` 运行

**方法 2：ArcGIS Pro Notebook**
1. 打开 ArcGIS Pro
2. 点击 `Insert` → `New Notebook`
3. 复制粘贴代码到单元格运行

**方法 3：命令行（推荐）**
```powershell
# 找到 ArcGIS Pro 的 Python 路径
# 通常在: C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe

& "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" "D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling\Scripts\01 preprocessing.py"
```

---

## Phase 1: 数据预处理

### 脚本：`Scripts/01 preprocessing.py`

### 功能说明
1. **验证数据完整性**：检查 12 个月的 TIF 是否都存在
2. **提取波段**：从多波段 TIF 中分离 LST 和 NDWI
3. **生成水体掩膜**：利用 NDWI 阈值提取水体

### 运行前检查
```
确保以下文件存在:
Data/Raw_TIF/
├── Tianjin_Monthly_Median_01.tif
├── Tianjin_Monthly_Median_02.tif
├── ...
└── Tianjin_Monthly_Median_12.tif
```

### 配置修改
打开脚本，修改第 23-28 行的路径配置：
```python
PROJECT_ROOT = r"D:\你的实际路径\Tianjin_Haihe_Cooling"
NDWI_THRESHOLD = 0.1  # 可根据目视效果调整 (0.05-0.2)
```

### 输出结果
```
Data/Processed/
├── Month_01/
│   ├── LST_01.tif
│   ├── NDWI_01.tif
│   └── Water_Binary_01.tif
├── Month_02/
│   └── ...
└── Month_12/

Data/Vector/
├── Water_Polygon_07.shp  (仅 7 月)
└── Water_Only_07.shp
```

---

## Phase 2: LST 反演与缓冲区分析

### 脚本：`Scripts/02 LST retrieval.py`

### 功能说明
1. **多环缓冲区**：以海河为中心，建立 30m-1000m 的同心环带
2. **分区统计**：计算每个环带内的平均 LST
3. **冷却阈值分析**：确定 TVoE（效率阈值距离）

### 运行前检查
```
确保以下文件存在:
Data/Vector/Haihe_River.shp  ← 海河矢量边界
```

### 配置修改
```python
# 第 29-32 行：缓冲区配置
BUFFER_DISTANCES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 
                    350, 400, 450, 500, 600, 700, 800, 900, 1000]
```

### 输出结果
```
Data/Vector/
└── Haihe_Buffers.shp  (多环缓冲区)

Data/
├── Gradient_Month_01.xlsx
├── Gradient_Month_02.xlsx
├── ...
├── Gradient_Month_12.xlsx
└── All_Months_Gradient.xlsx  ← 汇总文件

Tianjin_Haihe_Cooling.gdb/
├── Stats_LST_Month_01
├── Stats_LST_Month_02
└── ...
```

### 结果解读
Excel 文件包含以下列：
| distance | MEAN | COUNT | AREA | Month |
|----------|------|-------|------|-------|
| 30 | 28.5 | 1234 | 50000 | 07 |
| 60 | 29.2 | 2345 | 80000 | 07 |
| ... | ... | ... | ... | ... |

- **distance**: 距河边的距离 (米)
- **MEAN**: 该环带内的平均温度 (°C)
- **ΔT**: 用最远距离的温度减去最近距离的温度 = 冷却强度

---

## Phase 3: GWR 地理加权回归

### 脚本：`Scripts/03 GWR analysis.py`

### 功能说明
1. **创建采样点网格**：在研究区建立规则点阵
2. **提取变量值**：将 LST 和距离提取到点上
3. **运行 GWR**：分析降温效应的空间异质性

### 配置修改
```python
# 第 32-33 行：GWR 配置
CELL_SIZE = 150  # 采样点间距 (米)，越小越精细但计算越慢
STUDY_AREA_BUFFER = 1500  # 研究范围 (河边向外延伸)
```

### 输出结果
```
Tianjin_Haihe_Cooling.gdb/
├── Study_Area_Extent    (研究区范围)
├── Sample_Points        (采样点)
├── GWR_Results_07       (GWR 输出点要素)
└── GWR_Coeff_Raster_07  (系数栅格)
```

### 结果解读
GWR 输出包含以下关键字段：
| 字段名 | 含义 |
|--------|------|
| `Coeff_Dist_River` | 距离系数 (正值=离河越远越热=有冷却效应) |
| `LocalR2` | 局部 R² (模型拟合度) |
| `Residual` | 残差 |

**系数地图解读**：
- **高系数区域（深色）**: 河流冷却效应强
- **低系数区域（浅色）**: 河流冷却效应弱（可能有高楼阻挡）

---

## Phase 4: 可视化与出图

### 在 ArcGIS Pro 中操作

#### 1. LST 分布图
1. 加载 `Tianjin_Monthly_Median_07.tif`
2. 右键 → `Symbology` → `Classify`
3. 选择 `Natural Breaks` 分类
4. 配色：蓝-绿-黄-红 (冷到热)

#### 2. 冷却梯度曲线
1. 打开 `All_Months_Gradient.xlsx`
2. 筛选 7 月数据
3. 创建散点图：X = distance, Y = MEAN
4. 添加趋势线（对数函数拟合）

#### 3. GWR 系数图
1. 加载 `GWR_Results_07`
2. 按 `Coeff_Dist_River` 字段符号化
3. 使用渐变色（红-白-蓝）

#### 4. 导出地图
1. `Insert` → `New Layout`
2. 添加地图框、图例、比例尺、北指针
3. `Share` → `Export Layout` → PNG/PDF

---

## Phase 5: GitHub 版本控制

### 已配置的 .gitignore
以下文件会被自动忽略（不会上传到 GitHub）：
- 所有 `.tif` 栅格文件
- `.gdb` 地理数据库
- `.aprx` ArcGIS 工程文件
- `.lock` 锁文件

### 提交代码更改

```powershell
# 1. 进入项目目录
cd "D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling"

# 2. 查看状态
git status

# 3. 添加所有更改
git add .

# 4. 提交（写有意义的提交信息）
git commit -m "Add: Python scripts for preprocessing, LST analysis, and GWR"

# 5. 推送到 GitHub
git push origin main
```

### 推荐的提交节点

| 完成阶段 | 提交信息示例 |
|----------|--------------|
| Phase 1 | `Add: Preprocessing script - band extraction and water masking` |
| Phase 2 | `Add: Buffer analysis script - zonal statistics for LST gradient` |
| Phase 3 | `Add: GWR analysis script - spatial heterogeneity modeling` |
| 出图完成 | `Add: Maps - LST distribution and cooling gradient charts` |
| README 更新 | `Update: README with preliminary results` |

### 查看提交历史
```powershell
git log --oneline
```

---

## 📊 预期成果

完成所有步骤后，你应该有：

### 数据产品
- [ ] 12 个月的 LST 提取结果
- [ ] 多环缓冲区矢量
- [ ] 12 个月的梯度数据表
- [ ] GWR 系数空间分布

### 可视化产品
- [ ] LST 分布图 (7 月)
- [ ] 冷却梯度曲线图
- [ ] GWR 系数热力图
- [ ] 季节性对比图 (可选)

### 关键指标
- [ ] 冷却强度 ΔT (°C)
- [ ] 冷却阈值距离 TVoE (m)
- [ ] GWR 系数范围

---

## ❓ 常见问题

### Q: arcpy 导入失败？
**A:** 确保使用 ArcGIS Pro 的 Python 环境，而不是系统 Python。

### Q: 内存不足？
**A:** 减小 `CELL_SIZE` 或分批处理月份。

### Q: GWR 运行很慢？
**A:** 增大 `CELL_SIZE` (150→300)，或只处理单月数据。

### Q: GitHub 推送被拒绝？
**A:** 检查是否有大文件未被 .gitignore 忽略。

---

**祝分析顺利！如有问题随时询问。** 🎓
