"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 06 Multivariate GWR Analysis
DESCRIPTION: 
    - Multi-variable Geographically Weighted Regression
    - Model: LST = f(Distance, NDVI, NDBI)
    - Analyze spatial heterogeneity of cooling drivers
    - Compare with single-variable model
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: rasterio, geopandas, pandas, numpy, scipy, matplotlib

NOTE: This script requires v2 data with 4 bands (LST, NDVI, NDBI, NDWI)
      Run GEE script first to download: Tianjin_Monthly_v2_XX.tif
=============================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import rasterio
from shapely.geometry import Point
from scipy import stats
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

# Import shared configuration
from config import (RAW_TIF_DIR, VECTOR_DIR, HAIHE_RIVER,
                    GWR_MULTI_DIR, MAPS_GWR_MULTI, MONTH_NAMES,
                    SAMPLE_SPACING_MULTI, GWR_BANDWIDTH, ensure_dirs)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = GWR_MULTI_DIR
MAPS_DIR = MAPS_GWR_MULTI
SAMPLE_SPACING = SAMPLE_SPACING_MULTI

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create output directories."""
    for d in [OUTPUT_DIR, MAPS_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"[OK] Directory ready: {d}")

def check_v2_data():
    """Check if v2 data (4-band) exists."""
    print("\n" + "="*60)
    print("CHECKING DATA AVAILABILITY")
    print("="*60)
    
    v2_files = []
    v1_files = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        
        # Check for v2 files
        v2_path = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_v2_{month_str}.tif")
        v1_path = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month_str}.tif")
        
        if os.path.exists(v2_path):
            with rasterio.open(v2_path) as src:
                if src.count >= 4:
                    v2_files.append(month_str)
                    print(f"  [OK] Month {month_str}: v2 data found (4 bands)")
                else:
                    print(f"  [WARNING] Month {month_str}: v2 file has only {src.count} bands")
        elif os.path.exists(v1_path):
            v1_files.append(month_str)
            print(f"  [WARNING] Month {month_str}: Only v1 data (2 bands) - need to download v2")
        else:
            print(f"  [FAIL] Month {month_str}: No data found")
    
    if len(v2_files) == 0:
        print("\n" + "!"*60)
        print("  NO v2 DATA FOUND!")
        print("  Please run the updated GEE script to download 4-band TIFs:")
        print("  - File naming: Tianjin_Monthly_v2_XX.tif")
        print("  - Bands: LST_Celsius, NDVI, NDBI, NDWI")
        print("!"*60)
        return None
    
    print(f"\n  Summary: {len(v2_files)}/12 months with v2 data")
    return v2_files

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def create_sample_points(raster_path, spacing=SAMPLE_SPACING):
    """Create regular grid of sample points."""
    print(f"  Creating sample grid (spacing = {spacing}m)...")
    
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        
        x_coords = np.arange(bounds.left + spacing/2, bounds.right, spacing)
        y_coords = np.arange(bounds.bottom + spacing/2, bounds.top, spacing)
        
        points = []
        for x in x_coords:
            for y in y_coords:
                points.append(Point(x, y))
        
        gdf = gpd.GeoDataFrame({'geometry': points}, crs=crs)
        print(f"    [OK] Created {len(gdf)} sample points")
        return gdf

def extract_all_bands(gdf, raster_path):
    """Extract LST, NDVI, NDBI, NDWI values at sample points."""
    print(f"  Extracting multi-band values...")
    
    coords = [(p.x, p.y) for p in gdf.geometry]
    
    with rasterio.open(raster_path) as src:
        # Band order: LST, NDVI, NDBI, NDWI
        band_names = ['LST', 'NDVI', 'NDBI', 'NDWI']
        
        for band_idx, band_name in enumerate(band_names, start=1):
            values = list(src.sample(coords, indexes=band_idx))
            gdf[band_name] = [v[0] if len(v) > 0 else np.nan for v in values]
    
    # Filter valid values
    valid_mask = (
        (gdf['LST'] > -50) & (gdf['LST'] < 80) &
        (gdf['NDVI'] > -1) & (gdf['NDVI'] < 1) &
        (gdf['NDBI'] > -1) & (gdf['NDBI'] < 1) &
        (gdf['NDWI'] > -1) & (gdf['NDWI'] < 1)
    )
    gdf = gdf[valid_mask].dropna()
    
    print(f"    [OK] Valid samples: {len(gdf)}")
    print(f"    [OK] LST range: {gdf['LST'].min():.1f} - {gdf['LST'].max():.1f} °C")
    print(f"    [OK] NDVI range: {gdf['NDVI'].min():.3f} - {gdf['NDVI'].max():.3f}")
    print(f"    [OK] NDBI range: {gdf['NDBI'].min():.3f} - {gdf['NDBI'].max():.3f}")
    
    return gdf

def calculate_distance_to_river(gdf, river_shp=HAIHE_RIVER):
    """Calculate distance from each point to Haihe River."""
    print(f"  Calculating distance to river...")
    
    river = gpd.read_file(river_shp)
    river_dissolved = river.dissolve()
    river_geom = river_dissolved.geometry.iloc[0]
    
    distances = [point.distance(river_geom) for point in gdf.geometry]
    gdf['Distance'] = distances
    
    print(f"    [OK] Distance range: {min(distances):.0f} - {max(distances):.0f} m")
    return gdf

# ============================================================================
# MULTIVARIATE GWR
# ============================================================================

def run_multivariate_gwr(gdf, bandwidth=GWR_BANDWIDTH):
    """
    Run Geographically Weighted Regression with multiple explanatory variables.
    
    Model: LST = β₀ + β₁(Distance) + β₂(NDVI) + β₃(NDBI) + ε
    
    Each coefficient varies spatially, allowing us to see:
    - Where does distance to river matter most? (β₁)
    - Where does vegetation help cooling? (β₂)  
    - Where does built-up area cause warming? (β₃)
    """
    print(f"\n  Running Multivariate GWR (bandwidth = {bandwidth}m)...")
    
    # Prepare data
    coords = np.array([[p.x, p.y] for p in gdf.geometry])
    y = gdf['LST'].values
    
    # Explanatory variables (standardized for comparison)
    X_raw = gdf[['Distance', 'NDVI', 'NDBI']].values
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_standardized = (X_raw - X_mean) / X_std
    
    n = len(gdf)
    n_vars = X_standardized.shape[1]
    
    # Storage for local coefficients
    local_intercepts = np.zeros(n)
    local_coef_distance = np.zeros(n)
    local_coef_ndvi = np.zeros(n)
    local_coef_ndbi = np.zeros(n)
    local_r2 = np.zeros(n)
    local_residuals = np.zeros(n)
    
    # For each point, fit local weighted regression
    for i in range(n):
        if i % 5000 == 0:
            print(f"    Processing point {i}/{n}...")
        
        # Calculate distances to all other points
        dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
        
        # Gaussian kernel weights
        weights = np.exp(-(dists**2) / (2 * bandwidth**2))
        weights[dists > bandwidth * 3] = 0  # Truncate far points
        
        # Need enough weighted observations
        valid_mask = weights > 0.01
        if np.sum(valid_mask) > 15:  # Need more points for multivariate
            w = weights[valid_mask]
            X_local = X_standardized[valid_mask]
            y_local = y[valid_mask]
            
            # Add constant term
            X_design = np.column_stack([np.ones(len(X_local)), X_local])
            W = np.diag(w)
            
            try:
                # Weighted OLS: (X'WX)^-1 X'Wy
                XtW = X_design.T @ W
                beta = np.linalg.solve(XtW @ X_design, XtW @ y_local)
                
                local_intercepts[i] = beta[0]
                local_coef_distance[i] = beta[1]
                local_coef_ndvi[i] = beta[2]
                local_coef_ndbi[i] = beta[3]
                
                # Prediction at this point
                y_pred_i = X_design @ beta
                
                # Find index of current point in valid_mask
                orig_indices = np.where(valid_mask)[0]
                self_idx = np.where(orig_indices == i)[0]
                if len(self_idx) > 0:
                    local_residuals[i] = y_local[self_idx[0]] - y_pred_i[self_idx[0]]
                
                # Local R²
                ss_res = np.sum(w * (y_local - y_pred_i)**2)
                ss_tot = np.sum(w * (y_local - np.average(y_local, weights=w))**2)
                local_r2[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except:
                local_intercepts[i] = np.nan
                local_coef_distance[i] = np.nan
                local_coef_ndvi[i] = np.nan
                local_coef_ndbi[i] = np.nan
                local_r2[i] = np.nan
        else:
            local_intercepts[i] = np.nan
            local_coef_distance[i] = np.nan
            local_coef_ndvi[i] = np.nan
            local_coef_ndbi[i] = np.nan
            local_r2[i] = np.nan
    
    # Add results to GeoDataFrame
    result = gdf.copy()
    result['Intercept'] = local_intercepts
    result['Coef_Distance'] = local_coef_distance
    result['Coef_NDVI'] = local_coef_ndvi
    result['Coef_NDBI'] = local_coef_ndbi
    result['Local_R2'] = local_r2
    result['Residual'] = local_residuals
    
    # Summary statistics
    print("\n  GWR Results Summary:")
    print(f"    Coef_Distance: {np.nanmean(local_coef_distance):.4f} ± {np.nanstd(local_coef_distance):.4f}")
    print(f"    Coef_NDVI:     {np.nanmean(local_coef_ndvi):.4f} ± {np.nanstd(local_coef_ndvi):.4f}")
    print(f"    Coef_NDBI:     {np.nanmean(local_coef_ndbi):.4f} ± {np.nanstd(local_coef_ndbi):.4f}")
    print(f"    Mean Local R²: {np.nanmean(local_r2):.4f}")
    
    return result

def run_global_ols(gdf):
    """Run global OLS for comparison."""
    print("\n  Running Global OLS (for comparison)...")
    
    y = gdf['LST'].values
    X = gdf[['Distance', 'NDVI', 'NDBI']].values
    
    # Add intercept
    X_design = np.column_stack([np.ones(len(X)), X])
    
    # OLS
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    y_pred = X_design @ beta
    
    # R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    results = {
        'Intercept': beta[0],
        'Coef_Distance': beta[1],
        'Coef_NDVI': beta[2],
        'Coef_NDBI': beta[3],
        'R2': r2
    }
    
    print(f"    Global OLS R² = {r2:.4f}")
    print(f"    Intercept:     {beta[0]:.4f}")
    print(f"    Coef_Distance: {beta[1]:.6f}")
    print(f"    Coef_NDVI:     {beta[2]:.4f}")
    print(f"    Coef_NDBI:     {beta[3]:.4f}")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_coefficient_maps(gdf, month_str):
    """Create maps showing spatial distribution of GWR coefficients."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    coefficients = [
        ('Coef_Distance', 'Distance to River Effect\n(+: farther = hotter)', 'RdYlBu_r'),
        ('Coef_NDVI', 'Vegetation Effect\n(-: more green = cooler)', 'RdYlGn'),
        ('Coef_NDBI', 'Built-up Effect\n(+: more built-up = hotter)', 'RdYlBu_r'),
        ('Local_R2', 'Local Model Fit (R²)', 'viridis')
    ]
    
    for ax, (col, title, cmap) in zip(axes.flat, coefficients):
        values = gdf[col].values
        valid = ~np.isnan(values)
        
        if col == 'Local_R2':
            vmin, vmax = 0, 1
            norm = None
        else:
            # Center colormap at 0
            vmax = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
            vmin = -vmax
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        scatter = ax.scatter(
            gdf.geometry.x[valid], gdf.geometry.y[valid],
            c=values[valid], cmap=cmap, s=2, alpha=0.7,
            norm=norm if col != 'Local_R2' else None,
            vmin=vmin if col == 'Local_R2' else None,
            vmax=vmax if col == 'Local_R2' else None
        )
        
        plt.colorbar(scatter, ax=ax, shrink=0.8)
        ax.set_title(f'{title}', fontsize=11)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_aspect('equal')
    
    plt.suptitle(f'Multivariate GWR Coefficients - {MONTH_NAMES.get(month_str, month_str)}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(MAPS_DIR, f"GWR_Coefficients_{month_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

def plot_coefficient_comparison(all_results):
    """Compare coefficient distributions across months."""
    if not all_results:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    months = list(all_results.keys())
    coefs = ['Coef_Distance', 'Coef_NDVI', 'Coef_NDBI']
    titles = ['Distance Effect', 'NDVI Effect', 'NDBI Effect']
    colors = ['steelblue', 'forestgreen', 'firebrick']
    
    for ax, coef, title, color in zip(axes, coefs, titles, colors):
        means = [all_results[m]['mean_' + coef] for m in months]
        stds = [all_results[m]['std_' + coef] for m in months]
        
        x = range(len(months))
        ax.bar(x, means, yerr=stds, capsize=3, color=color, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MONTH_NAMES[m][:3] for m in months], rotation=45)
        ax.set_ylabel('Coefficient (standardized)')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Monthly Variation in GWR Coefficients', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(MAPS_DIR, "Monthly_Coefficient_Comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

def plot_variable_importance(all_results):
    """Plot relative importance of each variable by month."""
    if not all_results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = list(all_results.keys())
    x = np.arange(len(months))
    width = 0.25
    
    # Use absolute values for importance comparison
    dist_imp = [abs(all_results[m]['mean_Coef_Distance']) for m in months]
    ndvi_imp = [abs(all_results[m]['mean_Coef_NDVI']) for m in months]
    ndbi_imp = [abs(all_results[m]['mean_Coef_NDBI']) for m in months]
    
    ax.bar(x - width, dist_imp, width, label='Distance', color='steelblue')
    ax.bar(x, ndvi_imp, width, label='NDVI (Vegetation)', color='forestgreen')
    ax.bar(x + width, ndbi_imp, width, label='NDBI (Built-up)', color='firebrick')
    
    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_NAMES[m][:3] for m in months], rotation=45)
    ax.set_ylabel('Absolute Coefficient Value')
    ax.set_title('Relative Variable Importance by Month', fontsize=14)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Variable_Importance_Monthly.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def analyze_single_month(month_str):
    """Run complete multivariate GWR for one month."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {MONTH_NAMES.get(month_str, month_str)} (Month {month_str})")
    print('='*60)
    
    # Look for v2 data first, then fall back to v1
    v2_path = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_v2_{month_str}.tif")
    v1_path = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month_str}.tif")
    
    if os.path.exists(v2_path):
        raster_path = v2_path
        print(f"  Using v2 data (4 bands)")
    else:
        print(f"  [FAIL] v2 data not found: {v2_path}")
        print(f"    Please download from GEE first!")
        return None
    
    # Create sample points
    gdf = create_sample_points(raster_path)
    
    # Extract all band values
    gdf = extract_all_bands(gdf, raster_path)
    
    # Calculate distance to river
    gdf = calculate_distance_to_river(gdf)
    
    if len(gdf) < 100:
        print(f"  [FAIL] Too few valid samples")
        return None
    
    # Run global OLS
    ols_results = run_global_ols(gdf)
    
    # Run multivariate GWR
    gdf = run_multivariate_gwr(gdf)
    
    # Visualize
    plot_coefficient_maps(gdf, month_str)
    
    # Save results
    output_csv = os.path.join(OUTPUT_DIR, f"GWR_Multivariate_{month_str}.csv")
    gdf.drop(columns=['geometry']).to_csv(output_csv, index=False)
    print(f"  [OK] Results saved: {output_csv}")
    
    output_shp = os.path.join(OUTPUT_DIR, f"GWR_Multivariate_{month_str}.shp")
    gdf.to_file(output_shp)
    print(f"  [OK] Shapefile saved: {output_shp}")
    
    return {
        'month': month_str,
        'n_samples': len(gdf),
        'global_r2': ols_results['R2'],
        'mean_local_r2': np.nanmean(gdf['Local_R2']),
        'mean_Coef_Distance': np.nanmean(gdf['Coef_Distance']),
        'std_Coef_Distance': np.nanstd(gdf['Coef_Distance']),
        'mean_Coef_NDVI': np.nanmean(gdf['Coef_NDVI']),
        'std_Coef_NDVI': np.nanstd(gdf['Coef_NDVI']),
        'mean_Coef_NDBI': np.nanmean(gdf['Coef_NDBI']),
        'std_Coef_NDBI': np.nanstd(gdf['Coef_NDBI']),
    }

def run_all_months():
    """Run multivariate GWR for all available months."""
    print("\n" + "="*60)
    print("MULTIVARIATE GWR ANALYSIS")
    print("Model: LST = f(Distance, NDVI, NDBI)")
    print("="*60)
    
    setup_directories()
    
    # Check data availability
    available_months = check_v2_data()
    
    if not available_months:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Go to https://code.earthengine.google.com/")
        print("2. Paste the updated GEE script (Scripts/00 GEE_data_acquisition.js)")
        print("3. Run all 12 export tasks")
        print("4. Download files to Data/Raw_TIF/")
        print("5. Re-run this script")
        print("="*60)
        return
    
    all_results = {}
    
    for month_str in available_months:
        result = analyze_single_month(month_str)
        if result:
            all_results[month_str] = result
    
    if all_results:
        # Summary visualizations
        plot_coefficient_comparison(all_results)
        plot_variable_importance(all_results)
        
        # Save summary
        summary_df = pd.DataFrame(all_results.values())
        summary_path = os.path.join(OUTPUT_DIR, "GWR_Multivariate_Summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[OK] Summary saved: {summary_path}")
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON: GWR vs OLS")
        print("="*60)
        for month, res in all_results.items():
            improvement = res['mean_local_r2'] - res['global_r2']
            print(f"  {MONTH_NAMES[month]}: OLS R²={res['global_r2']:.4f}, "
                  f"GWR R²={res['mean_local_r2']:.4f} (Δ={improvement:+.4f})")
    
    print("\n" + "="*60)
    print("MULTIVARIATE GWR ANALYSIS COMPLETE!")
    print("="*60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_all_months()
