"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 03 GWR Analysis (Open Source Version - Fixed)
DESCRIPTION: 
    - Use existing Haihe_River.shp boundary
    - Create sample points, calculate distance to river
    - Run OLS regression analysis (with GWR-like spatial visualization)
    - Generate professional charts WITH equations
    - Process ALL 12 months
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
from shapely.geometry import Point, mapping
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import stats
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

# Import shared configuration
from config import (PROJECT_ROOT, RAW_TIF_DIR, VECTOR_DIR, HAIHE_RIVER,
                    DATA_DIR, MAPS_GWR_SINGLE, MONTH_NAMES,
                    SAMPLE_SPACING_GWR, ensure_dirs)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = DATA_DIR
MAPS_DIR = MAPS_GWR_SINGLE
SAMPLE_SPACING = SAMPLE_SPACING_GWR

# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def create_sample_points(raster_path, spacing=100):
    """Create regular grid sample points within raster extent."""
    print(f"\n  Creating sample grid (spacing = {spacing}m)...")
    
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        
        # Create grid
        x_coords = np.arange(bounds.left + spacing/2, bounds.right, spacing)
        y_coords = np.arange(bounds.bottom + spacing/2, bounds.top, spacing)
        
        points = []
        for x in x_coords:
            for y in y_coords:
                points.append(Point(x, y))
        
        gdf = gpd.GeoDataFrame({'geometry': points}, crs=crs)
        print(f"    ✓ Created {len(gdf)} sample points")
        
        return gdf

def extract_lst_values(sample_points, raster_path):
    """Extract LST values at sample points."""
    print(f"  Extracting LST values...")
    
    coords = [(p.x, p.y) for p in sample_points.geometry]
    
    with rasterio.open(raster_path) as src:
        lst_values = list(src.sample(coords))
        lst_values = [v[0] if len(v) > 0 else np.nan for v in lst_values]
    
    sample_points = sample_points.copy()
    sample_points['LST'] = lst_values
    
    # Filter valid values
    valid_mask = (sample_points['LST'] > -50) & (sample_points['LST'] < 100)
    sample_points = sample_points[valid_mask]
    
    print(f"    ✓ Valid samples: {len(sample_points)}")
    return sample_points

def calculate_distance_to_river(sample_points, river_gdf):
    """Calculate distance from each point to nearest river segment."""
    print(f"  Calculating distances to Haihe River...")
    
    # Dissolve river to single geometry
    river_dissolved = river_gdf.dissolve()
    river_geom = river_dissolved.geometry.iloc[0]
    
    distances = []
    for point in sample_points.geometry:
        dist = point.distance(river_geom)
        distances.append(dist)
    
    sample_points = sample_points.copy()
    sample_points['Distance'] = distances
    
    print(f"    ✓ Distance range: {min(distances):.0f} - {max(distances):.0f} m")
    return sample_points

# ============================================================================
# LOCAL REGRESSION (GWR-like approach)
# ============================================================================

def local_weighted_regression(sample_df, bandwidth=500):
    """
    Perform local weighted regression to get spatially varying coefficients.
    This simulates GWR by fitting local regressions at each point.
    """
    print(f"\n  Running Local Weighted Regression (bandwidth = {bandwidth}m)...")
    
    coords = np.array([[p.x, p.y] for p in sample_df.geometry])
    X = sample_df['Distance'].values
    y = sample_df['LST'].values
    
    n = len(sample_df)
    local_slopes = np.zeros(n)
    local_intercepts = np.zeros(n)
    local_r2 = np.zeros(n)
    
    for i in range(n):
        # Calculate distances to all other points
        dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
        
        # Gaussian weights
        weights = np.exp(-(dists**2) / (2 * bandwidth**2))
        weights[dists > bandwidth * 3] = 0  # Truncate far points
        
        # Weighted least squares
        valid_mask = weights > 0.01
        if np.sum(valid_mask) > 10:  # Need enough points
            w = weights[valid_mask]
            X_local = X[valid_mask]
            y_local = y[valid_mask]
            
            # Add constant term
            X_design = np.column_stack([np.ones(len(X_local)), X_local])
            W = np.diag(w)
            
            try:
                # Weighted OLS: (X'WX)^-1 X'Wy
                XtW = X_design.T @ W
                beta = np.linalg.solve(XtW @ X_design, XtW @ y_local)
                
                local_intercepts[i] = beta[0]
                local_slopes[i] = beta[1]
                
                # Local R²
                y_pred = X_design @ beta
                ss_res = np.sum(w * (y_local - y_pred)**2)
                ss_tot = np.sum(w * (y_local - np.mean(y_local))**2)
                local_r2[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except:
                local_slopes[i] = np.nan
                local_intercepts[i] = np.nan
                local_r2[i] = np.nan
        else:
            local_slopes[i] = np.nan
            local_intercepts[i] = np.nan
            local_r2[i] = np.nan
    
    result_df = sample_df.copy()
    result_df['Local_Slope'] = local_slopes
    result_df['Local_Intercept'] = local_intercepts
    result_df['Local_R2'] = local_r2
    
    valid_slopes = local_slopes[~np.isnan(local_slopes)]
    print(f"    ✓ Coefficient range: {valid_slopes.min():.6f} to {valid_slopes.max():.6f}")
    print(f"    ✓ Mean slope: {np.nanmean(local_slopes):.6f}")
    
    return result_df

def run_ols_regression(sample_df):
    """Run global OLS regression."""
    print("\n  Running Global OLS Regression...")
    
    X = sample_df['Distance'].values
    y = sample_df['LST'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    print(f"    ✓ Equation: LST = {slope:.6f} × Distance + {intercept:.2f}")
    print(f"    ✓ R² = {r_value**2:.4f}")
    print(f"    ✓ p-value = {p_value:.2e}")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

# ============================================================================
# VISUALIZATION WITH EQUATION
# ============================================================================

def plot_scatter_regression(sample_df, ols_result, month_str, output_path):
    """Create scatter plot with regression line and equation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = sample_df['Distance'].values
    y = sample_df['LST'].values
    
    # Sample for plotting (avoid overplotting)
    n_plot = min(5000, len(x))
    idx = np.random.choice(len(x), n_plot, replace=False)
    
    # Scatter plot with density coloring
    ax.scatter(x[idx], y[idx], s=10, alpha=0.3, c='#3498db', label='Sample Points')
    
    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = ols_result['slope'] * x_line + ols_result['intercept']
    ax.plot(x_line, y_line, 'r-', linewidth=3, label='OLS Regression')
    
    # Display equation
    sign = '+' if ols_result['intercept'] >= 0 else '-'
    equation = f"LST = {ols_result['slope']:.6f} × Distance {sign} {abs(ols_result['intercept']):.2f}"
    stats_text = f"R² = {ols_result['r_squared']:.4f}\np-value = {ols_result['p_value']:.2e}\nn = {len(x)}"
    
    ax.text(0.98, 0.05, f"Regression Equation:\n{equation}\n\n{stats_text}",
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=13)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=13)
    ax.set_title(f'Distance-LST Regression Analysis - {MONTH_NAMES[month_str]}\n'
                 f'Tianjin Haihe River Urban Cooling Effect', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Scatter plot saved: {output_path}")

def plot_local_regression_results(result_df, month_str, output_path):
    """
    Create GWR-style map showing spatially varying coefficients.
    Left: LST values, Right: Local slope coefficients
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Sample for faster plotting
    n_plot = min(10000, len(result_df))
    plot_df = result_df.sample(n=n_plot, random_state=42).copy()
    
    x = np.array([p.x for p in plot_df.geometry])
    y = np.array([p.y for p in plot_df.geometry])
    
    # Left plot: LST values
    ax1 = axes[0]
    lst_values = plot_df['LST'].values
    scatter1 = ax1.scatter(x, y, c=lst_values, cmap='RdYlBu_r', 
                           s=15, alpha=0.8, edgecolors='none')
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='LST (°C)', pad=0.02)
    ax1.set_title(f'Land Surface Temperature - {MONTH_NAMES[month_str]}', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Easting (m)', fontsize=11)
    ax1.set_ylabel('Northing (m)', fontsize=11)
    ax1.set_aspect('equal')
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    # Right plot: Local coefficients (slopes)
    ax2 = axes[1]
    slopes = plot_df['Local_Slope'].values
    
    # Remove NaN for visualization
    valid_mask = ~np.isnan(slopes)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    slopes_valid = slopes[valid_mask]
    
    # Use diverging colormap centered at 0
    if len(slopes_valid) > 0:
        vmin, vmax = np.percentile(slopes_valid, [5, 95])
        
        # Check if there's actual variation
        if abs(vmax - vmin) < 1e-8:
            # No significant variation - use single color but explain why
            scatter2 = ax2.scatter(x_valid, y_valid, c='#2ecc71', 
                                   s=15, alpha=0.8, edgecolors='none')
            ax2.text(0.5, 0.02, f'Coefficient ≈ {np.mean(slopes_valid):.6f}\n(Spatially uniform)',
                    transform=ax2.transAxes, fontsize=11, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Use diverging norm centered at mean or 0
            center = 0 if vmin < 0 < vmax else np.mean(slopes_valid)
            try:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            except:
                norm = Normalize(vmin=vmin, vmax=vmax)
            
            scatter2 = ax2.scatter(x_valid, y_valid, c=slopes_valid, 
                                   cmap='RdBu_r', norm=norm,
                                   s=15, alpha=0.8, edgecolors='none')
            cbar2 = plt.colorbar(scatter2, ax=ax2, label='Local Slope (°C/m)', pad=0.02)
        
        # Add statistics
        stats_text = f"Mean: {np.mean(slopes_valid):.6f}\nStd: {np.std(slopes_valid):.6f}"
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title(f'Local Regression Coefficient (Slope) - {MONTH_NAMES[month_str]}\n'
                  f'Spatial Variation of Distance Effect', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Easting (m)', fontsize=11)
    ax2.set_ylabel('Northing (m)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Local regression map saved: {output_path}")

def plot_local_r2_map(result_df, month_str, output_path):
    """Create map showing local R² values."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    n_plot = min(10000, len(result_df))
    plot_df = result_df.sample(n=n_plot, random_state=42).copy()
    
    x = np.array([p.x for p in plot_df.geometry])
    y = np.array([p.y for p in plot_df.geometry])
    r2_values = plot_df['Local_R2'].values
    
    valid_mask = ~np.isnan(r2_values)
    scatter = ax.scatter(x[valid_mask], y[valid_mask], 
                        c=r2_values[valid_mask], cmap='YlGnBu',
                        s=15, alpha=0.8, vmin=0, vmax=1, edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax, label='Local R²', pad=0.02)
    
    ax.set_title(f'Local Model Fit (R²) - {MONTH_NAMES[month_str]}\n'
                 f'Tianjin Haihe River Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Local R² map saved: {output_path}")

# ============================================================================
# MONTHLY SUMMARY
# ============================================================================

def create_monthly_summary(all_results):
    """Create summary chart comparing all months."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    months = []
    slopes = []
    r2_values = []
    
    for result in all_results:
        months.append(MONTH_NAMES[result['month']][:3])
        slopes.append(result['ols']['slope'] * 1000)  # Convert to °C/km
        r2_values.append(result['ols']['r_squared'])
    
    # Left: Slopes
    ax1 = axes[0]
    colors1 = ['#e74c3c' if s > 0 else '#3498db' for s in slopes]
    bars1 = ax1.bar(months, slopes, color=colors1, edgecolor='black')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Slope (°C/km)', fontsize=12)
    ax1.set_title('Distance Coefficient by Month\n(Positive = warming with distance)', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Right: R² values
    ax2 = axes[1]
    bars2 = ax2.bar(months, r2_values, color='#2ecc71', edgecolor='black')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Model Fit (R²) by Month', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(r2_values) * 1.2 if r2_values else 1)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars2, r2_values):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MAPS_DIR, "Monthly_Regression_Summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Monthly summary chart saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_single_month(month_str, sample_points, river_gdf):
    """Process a single month's data."""
    print(f"\n{'='*60}")
    print(f"Processing {MONTH_NAMES[month_str]} ({month_str})")
    print(f"{'='*60}")
    
    lst_raster = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month_str}.tif")
    
    if not os.path.exists(lst_raster):
        print(f"  ⚠ LST raster not found: {lst_raster}")
        return None
    
    # Extract LST values
    sample_with_lst = extract_lst_values(sample_points.copy(), lst_raster)
    
    # Calculate distance to river
    sample_with_dist = calculate_distance_to_river(sample_with_lst, river_gdf)
    
    # Filter to study area (within 1500m of river)
    sample_filtered = sample_with_dist[sample_with_dist['Distance'] <= 1500].copy()
    print(f"  ✓ Samples within 1500m: {len(sample_filtered)}")
    
    if len(sample_filtered) < 100:
        print("  ⚠ Not enough samples for analysis")
        return None
    
    # Run global OLS
    ols_result = run_ols_regression(sample_filtered)
    
    # Run local weighted regression
    local_result = local_weighted_regression(sample_filtered, bandwidth=500)
    
    # Save data
    output_csv = os.path.join(OUTPUT_DIR, "GWR_Results", f"GWR_Samples_{month_str}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    local_result.drop(columns='geometry').to_csv(output_csv, index=False)
    print(f"  ✓ Sample data saved: {output_csv}")
    
    # Generate visualizations
    print(f"\n  Generating visualizations...")
    
    # Scatter plot with equation
    scatter_path = os.path.join(MAPS_DIR, f"Regression_Scatter_{month_str}.png")
    plot_scatter_regression(sample_filtered, ols_result, month_str, scatter_path)
    
    # Local regression maps
    local_path = os.path.join(MAPS_DIR, f"Local_Regression_{month_str}.png")
    plot_local_regression_results(local_result, month_str, local_path)
    
    # Local R² map
    r2_path = os.path.join(MAPS_DIR, f"Local_R2_{month_str}.png")
    plot_local_r2_map(local_result, month_str, r2_path)
    
    return {
        'month': month_str,
        'ols': ols_result,
        'n_samples': len(sample_filtered)
    }

def main():
    """Main GWR analysis workflow."""
    print("\n" + "="*70)
    print("THE BLUE SPINE - LOCAL REGRESSION ANALYSIS (GWR-like)")
    print("Using YOUR Haihe_River.shp boundary")
    print("="*70)
    
    # Verify input files
    if not os.path.exists(HAIHE_RIVER):
        print(f"\nERROR: River shapefile not found: {HAIHE_RIVER}")
        return
    
    # Load river boundary
    print(f"\n✓ Loading your river boundary: {HAIHE_RIVER}")
    river_gdf = gpd.read_file(HAIHE_RIVER)
    print(f"  Features: {len(river_gdf)}, CRS: {river_gdf.crs}")
    
    # Find first available raster for creating sample grid
    first_raster = None
    for month in range(1, 13):
        raster_path = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month:02d}.tif")
        if os.path.exists(raster_path):
            first_raster = raster_path
            break
    
    if first_raster is None:
        print("ERROR: No LST rasters found!")
        return
    
    # Create sample points
    sample_points = create_sample_points(first_raster, spacing=SAMPLE_SPACING)
    
    # Process all 12 months
    all_results = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        result = process_single_month(month_str, sample_points, river_gdf)
        if result:
            all_results.append(result)
    
    # Create summary
    if all_results:
        create_monthly_summary(all_results)
        
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"\n{'Month':<12} {'Slope (°C/m)':<15} {'R²':<10} {'p-value':<15}")
        print("-" * 55)
        for r in all_results:
            print(f"{MONTH_NAMES[r['month']]:<12} {r['ols']['slope']:<15.6f} "
                  f"{r['ols']['r_squared']:<10.4f} {r['ols']['p_value']:<15.2e}")
    
    print("\n" + "="*70)
    print("LOCAL REGRESSION ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

