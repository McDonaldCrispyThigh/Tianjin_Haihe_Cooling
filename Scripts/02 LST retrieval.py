"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 02 LST Retrieval & Buffer Analysis (Open Source Version)
DESCRIPTION: 
    - Create multi-ring buffers around Haihe River
    - Calculate zonal statistics (mean LST per buffer zone)
    - Export results for gradient analysis
    - Generate cooling gradient charts
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: rasterio, geopandas, pandas, numpy, matplotlib (open source)
=============================================================================
"""

import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling"

# Input paths
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")
HAIHE_RIVER = os.path.join(VECTOR_DIR, "Haihe_River.shp")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
STATS_OUTPUT = os.path.join(PROJECT_ROOT, "Data")
MAPS_DIR = os.path.join(PROJECT_ROOT, "Maps")

# Buffer configuration (in meters)
BUFFER_DISTANCES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 
                    350, 400, 450, 500, 600, 700, 800, 900, 1000]

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Create output directories."""
    for directory in [OUTPUT_DIR, STATS_OUTPUT, MAPS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created: {directory}")

# ============================================================================
# MULTI-RING BUFFER CREATION
# ============================================================================

def create_multi_ring_buffer(input_shp, distances):
    """
    Create multiple ring buffers around the Haihe River.
    Each ring represents a distance zone (e.g., 0-30m, 30-60m, etc.)
    """
    print("\n" + "="*60)
    print("STEP 1: Creating Multi-Ring Buffers")
    print("="*60)
    
    # Read river shapefile
    river = gpd.read_file(input_shp)
    print(f"  ✓ Loaded river shapefile: {len(river)} features")
    print(f"  ✓ CRS: {river.crs}")
    
    # Dissolve to single geometry
    river_dissolved = river.dissolve()
    river_geom = river_dissolved.geometry.iloc[0]
    
    # Create ring buffers
    rings = []
    prev_buffer = river_geom
    
    for i, dist in enumerate(distances):
        # Create buffer at this distance
        current_buffer = river_geom.buffer(dist)
        
        if i == 0:
            # First ring: from river edge to first distance
            ring = current_buffer.difference(river_geom)
        else:
            # Subsequent rings: difference between current and previous buffer
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
    output_shp = os.path.join(VECTOR_DIR, "Haihe_Buffers_Rings.shp")
    buffers_gdf.to_file(output_shp)
    print(f"\n  ✓ Buffers saved: {output_shp}")
    print(f"  ✓ Total rings: {len(buffers_gdf)}")
    
    return buffers_gdf, output_shp

# ============================================================================
# ZONAL STATISTICS
# ============================================================================

def calculate_zonal_statistics(buffers_gdf, lst_raster_path, month_str):
    """
    Calculate mean LST for each buffer zone.
    This creates the data needed for the "Distance vs Temperature" curve.
    """
    print(f"\n  Calculating zonal statistics for Month {month_str}...")
    
    results = []
    
    with rasterio.open(lst_raster_path) as src:
        for idx, row in buffers_gdf.iterrows():
            try:
                # Mask raster with buffer geometry
                geom = [mapping(row.geometry)]
                out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)
                
                # Calculate statistics (ignore NoData)
                data = out_image[0]
                valid_data = data[~np.isnan(data)]
                
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
                print(f"    ⚠ Error at distance {row['distance']}: {e}")
                results.append({
                    'distance': row['distance'],
                    'MEAN': np.nan,
                    'STD': np.nan,
                    'MIN': np.nan,
                    'MAX': np.nan,
                    'COUNT': 0
                })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df['Month'] = month_str
    df = df.sort_values('distance')
    
    print(f"    ✓ Statistics calculated for {len(df)} zones")
    
    return df

# ============================================================================
# BATCH PROCESSING - ALL MONTHS
# ============================================================================

def process_all_months(buffers_gdf):
    """
    Run zonal statistics for all 12 monthly LST composites.
    """
    print("\n" + "="*60)
    print("STEP 2: Calculating Zonal Statistics for All Months")
    print("="*60)
    
    all_results = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        
        # Input LST raster (from GEE export)
        lst_raster = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month_str}.tif")
        
        if not os.path.exists(lst_raster):
            print(f"\n  ⚠ Month {month_str}: LST raster not found, skipping...")
            continue
        
        print(f"\n{'─'*40}")
        print(f"Processing Month {month_str}")
        print(f"{'─'*40}")
        
        # Calculate zonal statistics
        df = calculate_zonal_statistics(buffers_gdf, lst_raster, month_str)
        
        # Export to Excel
        excel_output = os.path.join(STATS_OUTPUT, f"Gradient_Month_{month_str}.xlsx")
        df.to_excel(excel_output, index=False)
        print(f"    ✓ Excel saved: {excel_output}")
        
        all_results.append(df)
    
    # Combine all months into one master file
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
    """
    Determine the Threshold Value of Efficiency (TVoE).
    """
    print(f"\n  Analyzing cooling threshold for Month {month_str}...")
    
    month_data = df[df['Month'] == month_str].copy()
    month_data = month_data.sort_values('distance')
    
    # Calculate temperature gradient
    month_data['temp_gradient'] = month_data['MEAN'].diff() / month_data['distance'].diff()
    
    # Find where gradient drops below threshold
    gradient_threshold = 0.005  # °C/m
    threshold_rows = month_data[abs(month_data['temp_gradient']) < gradient_threshold]
    
    if not threshold_rows.empty:
        tvoe = threshold_rows.iloc[0]['distance']
        print(f"    ✓ Cooling Threshold Distance (TVoE): {tvoe} meters")
    else:
        tvoe = None
        print(f"    ⚠ Could not determine TVoE")
    
    # Calculate cooling intensity
    water_temp = month_data.iloc[0]['MEAN']
    urban_temp = month_data.iloc[-1]['MEAN']
    delta_t = urban_temp - water_temp
    
    print(f"    ✓ Water Edge Temperature: {water_temp:.2f}°C")
    print(f"    ✓ Urban Matrix Temperature (1000m): {urban_temp:.2f}°C")
    print(f"    ✓ Cooling Intensity (ΔT): {delta_t:.2f}°C")
    
    return {
        'month': month_str,
        'tvoe': tvoe,
        'water_temp': water_temp,
        'urban_temp': urban_temp,
        'delta_t': delta_t
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cooling_gradient(df, month_str, output_path):
    """
    Create a cooling gradient chart for a specific month.
    """
    month_data = df[df['Month'] == month_str].sort_values('distance')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars
    ax.errorbar(month_data['distance'], month_data['MEAN'], 
                yerr=month_data['STD'], 
                fmt='o-', capsize=3, capthick=1,
                color='#e74c3c', ecolor='gray', 
                label=f'Month {month_str} Mean LST')
    
    # Add trend line
    z = np.polyfit(month_data['distance'], month_data['MEAN'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(month_data['distance'].min(), month_data['distance'].max(), 100)
    ax.plot(x_smooth, p(x_smooth), '--', color='#3498db', 
            label='Polynomial Fit', linewidth=2)
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=12)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=12)
    ax.set_title(f'Urban Cooling Island Effect - Month {month_str}\nTianjin Haihe River', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Chart saved: {output_path}")

def plot_seasonal_comparison(df, output_path):
    """
    Create a chart comparing all 12 months.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for months
    colors = plt.cm.viridis(np.linspace(0, 1, 12))
    
    for i, month in enumerate(range(1, 13)):
        month_str = f"{month:02d}"
        month_data = df[df['Month'] == month_str].sort_values('distance')
        
        if not month_data.empty:
            ax.plot(month_data['distance'], month_data['MEAN'], 
                   'o-', color=colors[i], label=f'Month {month_str}', 
                   alpha=0.7, markersize=4)
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=12)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=12)
    ax.set_title('Seasonal Variation of Urban Cooling Island Effect\nTianjin Haihe River (2020-2025)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Seasonal comparison chart: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main LST retrieval and buffer analysis workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - LST RETRIEVAL & BUFFER ANALYSIS")
    print("(Open Source Version)")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check if river boundary exists
    if not os.path.exists(HAIHE_RIVER):
        print(f"\nERROR: Haihe River shapefile not found at:")
        print(f"  {HAIHE_RIVER}")
        print("\nPlease ensure you have the river boundary file.")
        return
    
    # Step 1: Create multi-ring buffers
    buffers_gdf, buffer_shp = create_multi_ring_buffer(HAIHE_RIVER, BUFFER_DISTANCES)
    
    # Step 2: Process all months
    combined_df = process_all_months(buffers_gdf)
    
    if combined_df is None:
        print("ERROR: No data processed.")
        return
    
    # Step 3: Analyze cooling threshold for July
    print("\n" + "="*60)
    print("STEP 3: Cooling Threshold Analysis")
    print("="*60)
    
    threshold_results = analyze_cooling_threshold(combined_df, "07")
    
    # Step 4: Generate visualizations
    print("\n" + "="*60)
    print("STEP 4: Generating Visualizations")
    print("="*60)
    
    # July gradient chart
    july_chart = os.path.join(MAPS_DIR, "Cooling_Gradient_July.png")
    plot_cooling_gradient(combined_df, "07", july_chart)
    
    # Seasonal comparison
    seasonal_chart = os.path.join(MAPS_DIR, "Seasonal_Comparison.png")
    plot_seasonal_comparison(combined_df, seasonal_chart)
    
    # Final summary
    print("\n" + "="*60)
    print("LST RETRIEVAL COMPLETE")
    print("="*60)
    print(f"\nKey Results (July):")
    print(f"  • Cooling Intensity (ΔT): {threshold_results['delta_t']:.2f}°C")
    print(f"  • Threshold Distance: {threshold_results['tvoe']} m")
    print(f"\nOutput Files:")
    print(f"  • Excel Data: {os.path.join(STATS_OUTPUT, 'All_Months_Gradient.xlsx')}")
    print(f"  • Charts: {MAPS_DIR}")


if __name__ == "__main__":
    main()

