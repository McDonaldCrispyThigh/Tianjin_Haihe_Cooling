"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 03 GWR Analysis (Open Source Version)
DESCRIPTION: 
    - Create sample point grid for GWR analysis
    - Extract LST and distance values to points
    - Run GWR to analyze spatial heterogeneity of cooling effect
    - Visualize results
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: rasterio, geopandas, scipy, numpy, matplotlib (open source)
NOTE: Uses mgwr library for GWR (pip install mgwr)
=============================================================================
"""

import rasterio
from rasterio.transform import rowcol
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import mgwr for GWR analysis
try:
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    MGWR_AVAILABLE = True
except ImportError:
    MGWR_AVAILABLE = False
    print("⚠ mgwr library not installed. GWR analysis will use OLS fallback.")
    print("  Install with: pip install mgwr")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling"

# Input paths
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")
HAIHE_RIVER = os.path.join(VECTOR_DIR, "Haihe_River.shp")

# Output paths
GDB_DIR = os.path.join(PROJECT_ROOT, "Data", "GWR_Results")
MAPS_DIR = os.path.join(PROJECT_ROOT, "Maps")

# GWR Configuration
CELL_SIZE = 150  # meters - grid cell size for sample points
STUDY_AREA_BUFFER = 1500  # meters - analysis extent around river

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Create output directories."""
    for directory in [GDB_DIR, MAPS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created: {directory}")

# ============================================================================
# STEP 1: CREATE SAMPLE POINT GRID
# ============================================================================

def create_study_area_extent(river_shp, buffer_distance):
    """
    Create a study area polygon by buffering the river.
    """
    print("\n" + "="*60)
    print("STEP 1: Creating Study Area Extent")
    print("="*60)
    
    river = gpd.read_file(river_shp)
    river_dissolved = river.dissolve()
    
    # Buffer the river
    study_area = river_dissolved.buffer(buffer_distance)
    study_area_gdf = gpd.GeoDataFrame(geometry=study_area, crs=river.crs)
    
    print(f"  ✓ Study area: {buffer_distance}m buffer around river")
    print(f"  ✓ CRS: {river.crs}")
    
    return study_area_gdf, river_dissolved.geometry.iloc[0]

def create_sample_points(study_area_gdf, river_geom, cell_size):
    """
    Create a regular grid of sample points within the study area.
    """
    print("\n" + "="*60)
    print("STEP 2: Creating Sample Point Grid")
    print("="*60)
    
    study_area = study_area_gdf.geometry.iloc[0]
    bounds = study_area.bounds  # (minx, miny, maxx, maxy)
    
    # Generate grid points
    points = []
    x = bounds[0]
    while x <= bounds[2]:
        y = bounds[1]
        while y <= bounds[3]:
            point = Point(x, y)
            # Only keep points inside study area and outside river
            if study_area.contains(point) and not river_geom.contains(point):
                points.append(point)
            y += cell_size
        x += cell_size
    
    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=study_area_gdf.crs)
    points_gdf['point_id'] = range(len(points_gdf))
    
    print(f"  ✓ Grid resolution: {cell_size}m x {cell_size}m")
    print(f"  ✓ Sample points created: {len(points_gdf)}")
    
    # Save to shapefile
    output_shp = os.path.join(GDB_DIR, "Sample_Points.shp")
    points_gdf.to_file(output_shp)
    print(f"  ✓ Saved: {output_shp}")
    
    return points_gdf

# ============================================================================
# STEP 2: CALCULATE DISTANCE TO RIVER
# ============================================================================

def calculate_distance_to_river(points_gdf, river_shp):
    """
    Calculate Euclidean distance from each sample point to the nearest river edge.
    """
    print("\n" + "="*60)
    print("STEP 3: Calculating Distance to River")
    print("="*60)
    
    river = gpd.read_file(river_shp)
    river_dissolved = river.dissolve()
    river_geom = river_dissolved.geometry.iloc[0]
    
    # Calculate distance for each point
    distances = []
    for idx, row in points_gdf.iterrows():
        dist = row.geometry.distance(river_geom)
        distances.append(dist)
    
    points_gdf['Dist_River'] = distances
    
    print(f"  ✓ Distance calculated for {len(points_gdf)} points")
    print(f"  ✓ Distance range: {min(distances):.1f}m - {max(distances):.1f}m")
    
    return points_gdf

# ============================================================================
# STEP 3: EXTRACT LST VALUES
# ============================================================================

def extract_lst_to_points(points_gdf, lst_raster_path, month_str):
    """
    Extract LST values from raster to sample points.
    """
    print(f"\n  Extracting LST values for Month {month_str}...")
    
    with rasterio.open(lst_raster_path) as src:
        # Get coordinates
        coords = [(point.x, point.y) for point in points_gdf.geometry]
        
        # Sample raster at point locations
        lst_values = []
        for coord in coords:
            try:
                row, col = rowcol(src.transform, coord[0], coord[1])
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = src.read(1)[row, col]
                    lst_values.append(value if not np.isnan(value) else np.nan)
                else:
                    lst_values.append(np.nan)
            except:
                lst_values.append(np.nan)
    
    points_gdf[f'LST_{month_str}'] = lst_values
    
    valid_count = sum(1 for v in lst_values if not np.isnan(v))
    print(f"    ✓ LST extracted: {valid_count}/{len(lst_values)} valid values")
    
    return points_gdf

# ============================================================================
# STEP 4: GWR ANALYSIS
# ============================================================================

def run_gwr_analysis(points_gdf, dependent_var, explanatory_vars, month_str):
    """
    Run Geographically Weighted Regression (GWR).
    
    If mgwr is not available, falls back to simple OLS regression.
    """
    print("\n" + "="*60)
    print(f"STEP 4: Running GWR Analysis (Month {month_str})")
    print("="*60)
    
    # Prepare data
    df = points_gdf.copy()
    df = df.dropna(subset=[dependent_var] + explanatory_vars)
    
    if len(df) < 50:
        print(f"  ⚠ Not enough valid points ({len(df)}). Skipping GWR.")
        return None
    
    print(f"  ✓ Valid points: {len(df)}")
    print(f"  ✓ Dependent Variable: {dependent_var}")
    print(f"  ✓ Explanatory Variables: {explanatory_vars}")
    
    # Get coordinates and variables
    coords = np.array([(geom.x, geom.y) for geom in df.geometry])
    y = df[dependent_var].values.reshape(-1, 1)
    X = df[explanatory_vars].values
    
    if MGWR_AVAILABLE:
        print("\n  Running GWR with mgwr library...")
        try:
            # Select bandwidth using AICc
            selector = Sel_BW(coords, y, X)
            bw = selector.search(criterion='AICc')
            print(f"    ✓ Optimal bandwidth: {bw}")
            
            # Run GWR
            gwr_model = GWR(coords, y, X, bw)
            gwr_results = gwr_model.fit()
            
            # Add results to dataframe
            df['GWR_Predicted'] = gwr_results.predy.flatten()
            df['GWR_Residual'] = gwr_results.resid_response.flatten()
            df['LocalR2'] = gwr_results.localR2.flatten()
            
            # Add coefficient for distance (first explanatory variable)
            df['Coeff_Dist_River'] = gwr_results.params[:, 0]
            
            print(f"    ✓ GWR completed!")
            print(f"    ✓ Global R²: {gwr_results.R2:.4f}")
            print(f"    ✓ AICc: {gwr_results.aicc:.2f}")
            
        except Exception as e:
            print(f"    ⚠ GWR failed: {e}")
            print("    Falling back to OLS...")
            df = run_ols_fallback(df, y, X, explanatory_vars)
    else:
        print("\n  Running OLS regression (mgwr not available)...")
        df = run_ols_fallback(df, y, X, explanatory_vars)
    
    # Save results
    output_shp = os.path.join(GDB_DIR, f"GWR_Results_{month_str}.shp")
    gdf_result = gpd.GeoDataFrame(df, geometry='geometry', crs=points_gdf.crs)
    gdf_result.to_file(output_shp)
    print(f"  ✓ Results saved: {output_shp}")
    
    return gdf_result

def run_ols_fallback(df, y, X, explanatory_vars):
    """
    Run simple OLS regression as fallback.
    """
    from scipy import stats
    
    # Add constant for intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # OLS: beta = (X'X)^-1 X'y
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    
    # Predictions and residuals
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    df['OLS_Predicted'] = y_pred.flatten()
    df['OLS_Residual'] = residuals.flatten()
    df['Coeff_Dist_River'] = beta[1, 0]  # Distance coefficient (constant for OLS)
    
    print(f"    ✓ OLS completed!")
    print(f"    ✓ R²: {r2:.4f}")
    print(f"    ✓ Distance Coefficient: {beta[1, 0]:.6f}")
    
    return df

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

def visualize_gwr_results(gdf, month_str):
    """
    Create visualization of GWR results.
    """
    print(f"\n  Creating visualizations for Month {month_str}...")
    
    # Check which coefficient field exists
    if 'Coeff_Dist_River' in gdf.columns:
        coeff_field = 'Coeff_Dist_River'
    else:
        print("    ⚠ No coefficient field found for visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: LST Distribution
    ax1 = axes[0]
    lst_field = f'LST_{month_str}'
    if lst_field in gdf.columns:
        scatter1 = ax1.scatter(
            [p.x for p in gdf.geometry],
            [p.y for p in gdf.geometry],
            c=gdf[lst_field],
            cmap='RdYlBu_r',
            s=10,
            alpha=0.7
        )
        plt.colorbar(scatter1, ax=ax1, label='LST (°C)')
        ax1.set_title(f'Land Surface Temperature\nMonth {month_str}')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
    
    # Plot 2: Coefficient Distribution (or Residuals)
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        [p.x for p in gdf.geometry],
        [p.y for p in gdf.geometry],
        c=gdf[coeff_field],
        cmap='coolwarm',
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter2, ax=ax2, label='Coefficient')
    ax2.set_title(f'Distance-Temperature Coefficient\nMonth {month_str}')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    
    plt.tight_layout()
    
    output_path = os.path.join(MAPS_DIR, f"GWR_Results_{month_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Visualization saved: {output_path}")

def plot_distance_lst_relationship(gdf, month_str):
    """
    Create scatter plot of Distance vs LST with regression line.
    """
    lst_field = f'LST_{month_str}'
    if lst_field not in gdf.columns or 'Dist_River' not in gdf.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(gdf['Dist_River'], gdf[lst_field], 
               alpha=0.3, s=5, c='#3498db', label='Sample Points')
    
    # Fit polynomial
    valid_mask = ~(gdf['Dist_River'].isna() | gdf[lst_field].isna())
    x = gdf.loc[valid_mask, 'Dist_River'].values
    y = gdf.loc[valid_mask, lst_field].values
    
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_smooth, p(x_smooth), 'r-', linewidth=2, label='Polynomial Fit')
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=12)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=12)
    ax.set_title(f'Distance-Temperature Relationship\nMonth {month_str} (n={len(gdf)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(MAPS_DIR, f"Distance_LST_Scatter_{month_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Scatter plot saved: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main GWR analysis workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - GWR ANALYSIS MODULE")
    print("(Open Source Version)")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check inputs
    if not os.path.exists(HAIHE_RIVER):
        print(f"\nERROR: River shapefile not found: {HAIHE_RIVER}")
        return
    
    # Step 1: Create study area and sample points
    study_area_gdf, river_geom = create_study_area_extent(HAIHE_RIVER, STUDY_AREA_BUFFER)
    points_gdf = create_sample_points(study_area_gdf, river_geom, CELL_SIZE)
    
    # Step 2: Calculate distance to river
    points_gdf = calculate_distance_to_river(points_gdf, HAIHE_RIVER)
    
    # Step 3: Extract LST for July (peak summer)
    july_lst = os.path.join(RAW_TIF_DIR, "Tianjin_Monthly_Median_07.tif")
    
    if os.path.exists(july_lst):
        points_gdf = extract_lst_to_points(points_gdf, july_lst, "07")
        
        # Step 4: Run GWR
        gwr_result = run_gwr_analysis(
            points_gdf=points_gdf,
            dependent_var="LST_07",
            explanatory_vars=["Dist_River"],
            month_str="07"
        )
        
        # Step 5: Visualizations
        if gwr_result is not None:
            visualize_gwr_results(gwr_result, "07")
            plot_distance_lst_relationship(gwr_result, "07")
    else:
        print(f"\nWARNING: July LST raster not found: {july_lst}")
    
    # Final summary
    print("\n" + "="*60)
    print("GWR ANALYSIS COMPLETE")
    print("="*60)
    print("\nOutput Files:")
    print(f"  • Sample Points: {os.path.join(GDB_DIR, 'Sample_Points.shp')}")
    print(f"  • GWR Results: {os.path.join(GDB_DIR, 'GWR_Results_07.shp')}")
    print(f"  • Maps: {MAPS_DIR}")
    print("\nInterpretation:")
    print("  • Positive coefficient: Temperature increases with distance")
    print("    → River has COOLING effect")
    print("  • Higher coefficient areas: Stronger cooling influence")


if __name__ == "__main__":
    main()

