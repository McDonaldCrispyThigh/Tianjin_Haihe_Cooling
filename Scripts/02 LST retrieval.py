"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 02 LST Retrieval & Buffer Analysis (Open Source Version)
DESCRIPTION: 
    - Use existing Haihe_River.shp boundary (NO water extraction needed)
    - Create multi-ring buffers around Haihe River
    - Calculate zonal statistics for ALL 12 months
    - Generate cooling gradient charts with regression equations
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: rasterio, geopandas, pandas, numpy, matplotlib, scipy
=============================================================================
"""

import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Import shared configuration
from config import (PROJECT_ROOT, RAW_TIF_DIR, VECTOR_DIR, HAIHE_RIVER,
                    DATA_DIR, MAPS_BUFFER, MONTH_NAMES, BUFFER_DISTANCES,
                    ensure_dirs)

# ============================================================================
# CONFIGURATION
# ============================================================================

STATS_OUTPUT = DATA_DIR
MAPS_DIR = MAPS_BUFFER

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Create output directories."""
    ensure_dirs(STATS_OUTPUT, MAPS_DIR)

# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def logarithmic_func(x, a, b):
    """Logarithmic decay: T = a * ln(x) + b"""
    return a * np.log(x + 1) + b

def exponential_func(x, a, b, c):
    """Exponential decay: T = a * (1 - e^(-x/b)) + c"""
    return a * (1 - np.exp(-x / b)) + c

def polynomial_func(x, a, b, c):
    """Quadratic polynomial: T = ax² + bx + c"""
    return a * x**2 + b * x + c

# ============================================================================
# MULTI-RING BUFFER CREATION
# ============================================================================

def create_multi_ring_buffer(input_shp, distances):
    """
    Create multiple ring buffers around the Haihe River.
    Uses YOUR existing Haihe_River.shp boundary!
    """
    print("\n" + "="*60)
    print("STEP 1: Creating Multi-Ring Buffers from Haihe_River.shp")
    print("="*60)
    
    # Read YOUR river shapefile
    river = gpd.read_file(input_shp)
    print(f"  ✓ Loaded YOUR river boundary: {input_shp}")
    print(f"  ✓ Features: {len(river)}")
    print(f"  ✓ CRS: {river.crs}")
    
    # Dissolve to single geometry
    river_dissolved = river.dissolve()
    river_geom = river_dissolved.geometry.iloc[0]
    
    # Create ring buffers
    rings = []
    for i, dist in enumerate(distances):
        current_buffer = river_geom.buffer(dist)
        
        if i == 0:
            ring = current_buffer.difference(river_geom)
        else:
            prev_dist = distances[i-1]
            prev_buffer = river_geom.buffer(prev_dist)
            ring = current_buffer.difference(prev_buffer)
        
        rings.append({
            'distance': dist,
            'geometry': ring
        })
        print(f"    Creating buffer: {dist}m")
    
    # Create GeoDataFrame
    buffers_gdf = gpd.GeoDataFrame(rings, crs=river.crs)
    
    # Save to shapefile
    output_shp = os.path.join(VECTOR_DIR, "Haihe_Buffers_Analysis.shp")
    buffers_gdf.to_file(output_shp)
    print(f"\n  ✓ Buffers saved: {output_shp}")
    print(f"  ✓ Total rings: {len(buffers_gdf)}")
    
    return buffers_gdf, river_geom

# ============================================================================
# ZONAL STATISTICS
# ============================================================================

def calculate_zonal_statistics(buffers_gdf, lst_raster_path, month_str):
    """Calculate mean LST for each buffer zone."""
    print(f"\n  Calculating zonal statistics for {MONTH_NAMES.get(month_str, month_str)}...")
    
    results = []
    
    with rasterio.open(lst_raster_path) as src:
        for idx, row in buffers_gdf.iterrows():
            try:
                geom = [mapping(row.geometry)]
                out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)
                
                data = out_image[0]
                valid_data = data[~np.isnan(data) & (data > -100) & (data < 100)]
                
                if len(valid_data) > 0:
                    mean_lst = np.nanmean(valid_data)
                    std_lst = np.nanstd(valid_data)
                    min_lst = np.nanmin(valid_data)
                    max_lst = np.nanmax(valid_data)
                    count = len(valid_data)
                else:
                    mean_lst = std_lst = min_lst = max_lst = np.nan
                    count = 0
                
                results.append({
                    'distance': row['distance'],
                    'MEAN': mean_lst,
                    'STD': std_lst,
                    'MIN': min_lst,
                    'MAX': max_lst,
                    'COUNT': count
                })
            except Exception as e:
                results.append({
                    'distance': row['distance'],
                    'MEAN': np.nan, 'STD': np.nan, 'MIN': np.nan, 'MAX': np.nan, 'COUNT': 0
                })
    
    df = pd.DataFrame(results)
    df['Month'] = month_str
    df = df.sort_values('distance')
    
    print(f"    ✓ Statistics calculated for {len(df)} zones")
    
    return df

# ============================================================================
# PROCESS ALL 12 MONTHS
# ============================================================================

def process_all_months(buffers_gdf):
    """Run zonal statistics for all 12 monthly LST composites."""
    print("\n" + "="*60)
    print("STEP 2: Calculating Zonal Statistics for ALL 12 Months")
    print("="*60)
    
    all_results = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        lst_raster = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month_str}.tif")
        
        if not os.path.exists(lst_raster):
            print(f"\n  ⚠ Month {month_str}: LST raster not found, skipping...")
            continue
        
        print(f"\n{'─'*40}")
        print(f"Processing {MONTH_NAMES[month_str]} ({month_str})")
        print(f"{'─'*40}")
        
        df = calculate_zonal_statistics(buffers_gdf, lst_raster, month_str)
        
        # Export individual month Excel
        excel_output = os.path.join(STATS_OUTPUT, f"Gradient_Month_{month_str}.xlsx")
        df.to_excel(excel_output, index=False)
        print(f"    ✓ Excel saved: {excel_output}")
        
        all_results.append(df)
    
    # Combine all months
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        master_excel = os.path.join(STATS_OUTPUT, "All_Months_Gradient.xlsx")
        combined_df.to_excel(master_excel, index=False)
        print(f"\n✓ Master gradient file: {master_excel}")
        return combined_df
    
    return None

# ============================================================================
# COOLING THRESHOLD ANALYSIS
# ============================================================================

def analyze_cooling_threshold(df, month_str):
    """Determine the Threshold Value of Efficiency (TVoE)."""
    month_data = df[df['Month'] == month_str].copy()
    month_data = month_data.sort_values('distance')
    
    # Calculate temperature gradient
    month_data['temp_gradient'] = month_data['MEAN'].diff() / month_data['distance'].diff()
    
    # Find where gradient drops below threshold
    gradient_threshold = 0.005  # °C/m
    threshold_rows = month_data[abs(month_data['temp_gradient']) < gradient_threshold]
    
    tvoe = threshold_rows.iloc[0]['distance'] if not threshold_rows.empty else None
    
    water_temp = month_data.iloc[0]['MEAN']
    urban_temp = month_data.iloc[-1]['MEAN']
    delta_t = urban_temp - water_temp
    
    return {
        'month': month_str,
        'month_name': MONTH_NAMES.get(month_str, month_str),
        'tvoe': tvoe,
        'water_temp': water_temp,
        'urban_temp': urban_temp,
        'delta_t': delta_t
    }

# ============================================================================
# VISUALIZATION - COOLING GRADIENT WITH EQUATION
# ============================================================================

def plot_cooling_gradient_with_equation(df, month_str, output_path):
    """Create cooling gradient chart WITH regression equation displayed."""
    month_data = df[df['Month'] == month_str].sort_values('distance')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = month_data['distance'].values
    y = month_data['MEAN'].values
    yerr = month_data['STD'].values
    
    # Plot data points with error bars
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, capthick=1.5,
                color='#e74c3c', ecolor='gray', markersize=8,
                label='Mean LST ± Std Dev')
    
    # Fit logarithmic function
    try:
        popt_log, _ = curve_fit(logarithmic_func, x, y, p0=[1, 30], maxfev=5000)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_log = logarithmic_func(x_smooth, *popt_log)
        
        # Calculate R²
        y_pred = logarithmic_func(x, *popt_log)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_log = 1 - (ss_res / ss_tot)
        
        # Plot fitted curve
        ax.plot(x_smooth, y_log, '-', color='#3498db', linewidth=2.5,
                label=f'Logarithmic Fit (R² = {r2_log:.4f})')
        
        # Display equation on plot
        equation_text = f'T = {popt_log[0]:.4f} × ln(d+1) + {popt_log[1]:.2f}'
        ax.text(0.95, 0.05, f'Fitted Equation:\n{equation_text}\nR² = {r2_log:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    except Exception as e:
        print(f"    ⚠ Logarithmic fit failed: {e}")
    
    # Calculate cooling metrics
    delta_t = y[-1] - y[0]
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=13)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=13)
    ax.set_title(f'Urban Cooling Island Effect - {MONTH_NAMES[month_str]}\n'
                 f'Tianjin Haihe River | ΔT = {delta_t:.2f}°C', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, x.max() * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Gradient chart saved: {output_path}")
    return popt_log if 'popt_log' in dir() else None

# ============================================================================
# VISUALIZATION - SCATTER PLOT WITH EQUATION
# ============================================================================

def plot_scatter_with_equation(df, month_str, output_path):
    """Create scatter plot with polynomial fit and equation."""
    month_data = df[df['Month'] == month_str].sort_values('distance')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = month_data['distance'].values
    y = month_data['MEAN'].values
    
    # Scatter plot
    ax.scatter(x, y, s=100, c='#3498db', alpha=0.7, edgecolors='white', linewidth=1.5)
    
    # Polynomial fit
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    
    # Calculate R²
    y_pred = p(x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    ax.plot(x_smooth, p(x_smooth), 'r-', linewidth=2.5, 
            label=f'Polynomial Fit (R² = {r2:.4f})')
    
    # Display equation
    equation_text = f'T = {z[0]:.2e}d² + {z[1]:.4f}d + {z[2]:.2f}'
    ax.text(0.95, 0.05, f'Fitted Equation:\n{equation_text}\nR² = {r2:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=13)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=13)
    ax.set_title(f'Distance-Temperature Relationship - {MONTH_NAMES[month_str]}\n'
                 f'Tianjin Haihe River', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Scatter plot saved: {output_path}")

# ============================================================================
# VISUALIZATION - SEASONAL COMPARISON
# ============================================================================

def plot_seasonal_comparison(df, output_path):
    """Create chart comparing all 12 months."""
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Color maps for seasons
    season_colors = {
        '12': '#1f77b4', '01': '#2ca02c', '02': '#17becf',  # Winter (blue/green)
        '03': '#bcbd22', '04': '#98df8a', '05': '#7f7f7f',  # Spring
        '06': '#ff7f0e', '07': '#d62728', '08': '#e377c2',  # Summer (warm)
        '09': '#9467bd', '10': '#8c564b', '11': '#aec7e8',  # Fall
    }
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        month_data = df[df['Month'] == month_str].sort_values('distance')
        
        if not month_data.empty:
            color = season_colors.get(month_str, '#333333')
            ax.plot(month_data['distance'], month_data['MEAN'], 
                   'o-', color=color, label=f'{MONTH_NAMES[month_str]}', 
                   alpha=0.8, markersize=5, linewidth=1.5)
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=13)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=13)
    ax.set_title('Seasonal Variation of Urban Cooling Island Effect\n'
                 'Tianjin Haihe River (2020-2025 Monthly Medians)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Seasonal comparison chart: {output_path}")

# ============================================================================
# VISUALIZATION - COOLING INTENSITY SUMMARY
# ============================================================================

def plot_monthly_cooling_intensity(df, output_path):
    """Create bar chart of monthly cooling intensity (ΔT)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = []
    delta_ts = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        month_data = df[df['Month'] == month_str].sort_values('distance')
        
        if not month_data.empty:
            water_temp = month_data.iloc[0]['MEAN']
            urban_temp = month_data.iloc[-1]['MEAN']
            delta_t = urban_temp - water_temp
            
            months.append(MONTH_NAMES[month_str][:3])
            delta_ts.append(delta_t)
    
    colors = ['#3498db' if dt > 0 else '#e74c3c' for dt in delta_ts]
    bars = ax.bar(months, delta_ts, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, dt in zip(bars, delta_ts):
        height = bar.get_height()
        ax.annotate(f'{dt:.1f}°C',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cooling Intensity ΔT (°C)', fontsize=12)
    ax.set_title('Monthly Cooling Intensity of Haihe River\n'
                 '(Temperature Difference: Urban 1000m - River Edge)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Monthly cooling intensity chart: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main LST retrieval and buffer analysis workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - LST RETRIEVAL & BUFFER ANALYSIS")
    print("Using YOUR Haihe_River.shp boundary")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check if river boundary exists
    if not os.path.exists(HAIHE_RIVER):
        print(f"\nERROR: Haihe River shapefile not found at:")
        print(f"  {HAIHE_RIVER}")
        return
    
    print(f"\n✓ Using your river boundary: {HAIHE_RIVER}")
    
    # Step 1: Create multi-ring buffers from YOUR shapefile
    buffers_gdf, river_geom = create_multi_ring_buffer(HAIHE_RIVER, BUFFER_DISTANCES)
    
    # Step 2: Process all 12 months
    combined_df = process_all_months(buffers_gdf)
    
    if combined_df is None:
        print("ERROR: No data processed.")
        return
    
    # Step 3: Generate visualizations for ALL months
    print("\n" + "="*60)
    print("STEP 3: Generating Visualizations for ALL Months")
    print("="*60)
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        if month_str in combined_df['Month'].values:
            print(f"\n  {MONTH_NAMES[month_str]}:")
            
            # Cooling gradient chart with equation
            gradient_path = os.path.join(MAPS_DIR, f"Cooling_Gradient_{month_str}.png")
            plot_cooling_gradient_with_equation(combined_df, month_str, gradient_path)
            
            # Scatter plot with equation
            scatter_path = os.path.join(MAPS_DIR, f"Scatter_Plot_{month_str}.png")
            plot_scatter_with_equation(combined_df, month_str, scatter_path)
    
    # Seasonal comparison
    seasonal_path = os.path.join(MAPS_DIR, "Seasonal_Comparison_All.png")
    plot_seasonal_comparison(combined_df, seasonal_path)
    
    # Monthly cooling intensity bar chart
    intensity_path = os.path.join(MAPS_DIR, "Monthly_Cooling_Intensity.png")
    plot_monthly_cooling_intensity(combined_df, intensity_path)
    
    # Step 4: Summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nCooling Intensity by Month:")
    print("-" * 50)
    for month in range(1, 13):
        month_str = f"{month:02d}"
        if month_str in combined_df['Month'].values:
            result = analyze_cooling_threshold(combined_df, month_str)
            print(f"  {result['month_name']:12s} | ΔT = {result['delta_t']:+.2f}°C | "
                  f"TVoE = {result['tvoe']}m")
    
    print("\n" + "="*60)
    print("LST RETRIEVAL COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

