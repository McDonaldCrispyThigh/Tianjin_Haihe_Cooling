"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 04 Spatial Autocorrelation Analysis
DESCRIPTION: 
    - Global Moran's I: Test if LST shows spatial clustering
    - Local Moran's I (LISA): Identify hot/cold spot clusters
    - Getis-Ord Gi*: Statistically significant hot/cold spots
    - Generate publication-ready maps
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: libpysal, esda, geopandas, numpy, matplotlib
=============================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Spatial statistics libraries
try:
    from libpysal.weights import Queen, KNN, DistanceBand
    from esda.moran import Moran, Moran_Local
    from esda.getisord import G_Local
    from splot.esda import moran_scatterplot, lisa_cluster, plot_local_autocorrelation
    SPATIAL_LIBS_AVAILABLE = True
except ImportError:
    print("⚠️  Spatial libraries not installed. Run:")
    print("    pip install libpysal esda splot")
    SPATIAL_LIBS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Data", "Spatial_Stats")
MAPS_DIR = os.path.join(PROJECT_ROOT, "Maps", "Spatial_Autocorrelation")

# Sampling configuration
SAMPLE_SPACING = 150  # meters (balance between resolution and computation)

# Analysis parameters
SIGNIFICANCE_LEVEL = 0.05
N_PERMUTATIONS = 999  # For Monte Carlo significance testing

MONTH_NAMES = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December'
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create output directories."""
    for d in [OUTPUT_DIR, MAPS_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Directory ready: {d}")

# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def create_sample_grid(raster_path, spacing=SAMPLE_SPACING):
    """Create regular grid of sample points."""
    print(f"\n  Creating sample grid (spacing = {spacing}m)...")
    
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
        print(f"    ✓ Created {len(gdf)} sample points")
        return gdf

def extract_lst_at_points(gdf, raster_path):
    """Extract LST values at sample points."""
    coords = [(p.x, p.y) for p in gdf.geometry]
    
    with rasterio.open(raster_path) as src:
        values = list(src.sample(coords))
        lst = [v[0] if len(v) > 0 else np.nan for v in values]
    
    gdf = gdf.copy()
    gdf['LST'] = lst
    
    # Filter invalid values
    gdf = gdf[(gdf['LST'] > -50) & (gdf['LST'] < 80)]
    gdf = gdf.dropna(subset=['LST'])
    
    print(f"    ✓ Valid samples: {len(gdf)}")
    return gdf

# ============================================================================
# GLOBAL MORAN'S I
# ============================================================================

def calculate_global_morans_i(gdf, variable='LST', k_neighbors=8):
    """
    Calculate Global Moran's I statistic.
    
    Interpretation:
    - I > 0: Positive spatial autocorrelation (clustering)
    - I < 0: Negative spatial autocorrelation (dispersion)
    - I ≈ 0: Random spatial pattern
    """
    print(f"\n  Calculating Global Moran's I...")
    
    # Create spatial weights matrix (K-Nearest Neighbors)
    w = KNN.from_dataframe(gdf, k=k_neighbors)
    w.transform = 'R'  # Row standardization
    
    # Calculate Moran's I
    y = gdf[variable].values
    moran = Moran(y, w, permutations=N_PERMUTATIONS)
    
    results = {
        'Morans_I': moran.I,
        'Expected_I': moran.EI,
        'Variance': moran.VI_norm,
        'Z_score': moran.z_norm,
        'P_value': moran.p_norm,
        'P_value_sim': moran.p_sim,  # Simulation-based p-value
        'Significant': moran.p_sim < SIGNIFICANCE_LEVEL
    }
    
    print(f"    ✓ Moran's I = {moran.I:.4f}")
    print(f"    ✓ Z-score = {moran.z_norm:.4f}")
    print(f"    ✓ P-value = {moran.p_sim:.4f}")
    
    if moran.p_sim < SIGNIFICANCE_LEVEL:
        if moran.I > 0:
            print(f"    → SIGNIFICANT positive spatial autocorrelation (clustering)")
        else:
            print(f"    → SIGNIFICANT negative spatial autocorrelation (dispersion)")
    else:
        print(f"    → No significant spatial autocorrelation")
    
    return moran, w, results

# ============================================================================
# LOCAL MORAN'S I (LISA)
# ============================================================================

def calculate_lisa(gdf, w, variable='LST'):
    """
    Calculate Local Indicators of Spatial Association (LISA).
    
    Cluster types:
    - HH (High-High): Hot spot - high value surrounded by high values
    - LL (Low-Low): Cold spot - low value surrounded by low values  
    - HL (High-Low): Spatial outlier - high value surrounded by low values
    - LH (Low-High): Spatial outlier - low value surrounded by high values
    - NS: Not significant
    """
    print(f"\n  Calculating Local Moran's I (LISA)...")
    
    y = gdf[variable].values
    lisa = Moran_Local(y, w, permutations=N_PERMUTATIONS)
    
    gdf = gdf.copy()
    gdf['LISA_I'] = lisa.Is
    gdf['LISA_q'] = lisa.q  # Quadrant (1=HH, 2=LH, 3=LL, 4=HL)
    gdf['LISA_p'] = lisa.p_sim
    gdf['LISA_sig'] = lisa.p_sim < SIGNIFICANCE_LEVEL
    
    # Create cluster labels
    cluster_labels = []
    for i in range(len(gdf)):
        if lisa.p_sim[i] >= SIGNIFICANCE_LEVEL:
            cluster_labels.append('Not Significant')
        elif lisa.q[i] == 1:
            cluster_labels.append('High-High (Hot Spot)')
        elif lisa.q[i] == 2:
            cluster_labels.append('Low-High (Outlier)')
        elif lisa.q[i] == 3:
            cluster_labels.append('Low-Low (Cold Spot)')
        elif lisa.q[i] == 4:
            cluster_labels.append('High-Low (Outlier)')
        else:
            cluster_labels.append('Not Significant')
    
    gdf['LISA_Cluster'] = cluster_labels
    
    # Summary statistics
    cluster_counts = gdf['LISA_Cluster'].value_counts()
    print(f"    ✓ Cluster distribution:")
    for cluster, count in cluster_counts.items():
        pct = count / len(gdf) * 100
        print(f"      - {cluster}: {count} ({pct:.1f}%)")
    
    return lisa, gdf

# ============================================================================
# GETIS-ORD Gi*
# ============================================================================

def calculate_getis_ord(gdf, variable='LST', distance_threshold=500):
    """
    Calculate Getis-Ord Gi* statistic.
    
    Interpretation:
    - High positive Gi*: Clustering of high values (hot spot)
    - High negative Gi*: Clustering of low values (cold spot)
    - Near zero: No significant clustering
    """
    print(f"\n  Calculating Getis-Ord Gi*...")
    
    # Distance-based weights for Gi*
    w = DistanceBand.from_dataframe(gdf, threshold=distance_threshold, binary=False)
    w.transform = 'R'
    
    y = gdf[variable].values
    g_local = G_Local(y, w, star=True, permutations=N_PERMUTATIONS)
    
    gdf = gdf.copy()
    gdf['Gi_Z'] = g_local.Zs
    gdf['Gi_p'] = g_local.p_sim
    
    # Classify hot/cold spots
    spot_labels = []
    for i in range(len(gdf)):
        z = g_local.Zs[i]
        p = g_local.p_sim[i]
        
        if p >= SIGNIFICANCE_LEVEL:
            spot_labels.append('Not Significant')
        elif z > 2.58:  # 99% confidence
            spot_labels.append('Hot Spot (99%)')
        elif z > 1.96:  # 95% confidence
            spot_labels.append('Hot Spot (95%)')
        elif z > 1.65:  # 90% confidence
            spot_labels.append('Hot Spot (90%)')
        elif z < -2.58:
            spot_labels.append('Cold Spot (99%)')
        elif z < -1.96:
            spot_labels.append('Cold Spot (95%)')
        elif z < -1.65:
            spot_labels.append('Cold Spot (90%)')
        else:
            spot_labels.append('Not Significant')
    
    gdf['Gi_Spot'] = spot_labels
    
    # Summary
    spot_counts = gdf['Gi_Spot'].value_counts()
    print(f"    ✓ Hot/Cold spot distribution:")
    for spot, count in spot_counts.items():
        pct = count / len(gdf) * 100
        print(f"      - {spot}: {count} ({pct:.1f}%)")
    
    return g_local, gdf

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_moran_scatterplot(moran, gdf, month_str, variable='LST'):
    """Create Moran scatterplot with regression line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Standardize values
    y = gdf[variable].values
    y_std = (y - y.mean()) / y.std()
    
    # Spatial lag
    lag = moran.z
    
    ax.scatter(y_std, lag, alpha=0.5, s=10, c='steelblue')
    
    # Regression line
    slope, intercept = moran.I, 0
    x_line = np.linspace(y_std.min(), y_std.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f"Moran's I = {moran.I:.4f}")
    
    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # Quadrant labels
    ax.text(0.95, 0.95, 'HH', transform=ax.transAxes, fontsize=12, va='top', ha='right', color='red')
    ax.text(0.05, 0.95, 'LH', transform=ax.transAxes, fontsize=12, va='top', ha='left', color='lightblue')
    ax.text(0.05, 0.05, 'LL', transform=ax.transAxes, fontsize=12, va='bottom', ha='left', color='blue')
    ax.text(0.95, 0.05, 'HL', transform=ax.transAxes, fontsize=12, va='bottom', ha='right', color='pink')
    
    ax.set_xlabel('Standardized LST', fontsize=12)
    ax.set_ylabel('Spatial Lag of LST', fontsize=12)
    ax.set_title(f"Moran's I Scatterplot - {MONTH_NAMES.get(month_str, month_str)}\n"
                 f"I = {moran.I:.4f}, p = {moran.p_sim:.4f}", fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, f"Moran_Scatter_{month_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")

def plot_lisa_cluster_map(gdf, month_str):
    """Create LISA cluster map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color mapping for clusters
    colors = {
        'High-High (Hot Spot)': '#d7191c',
        'Low-Low (Cold Spot)': '#2c7bb6',
        'High-Low (Outlier)': '#fdae61',
        'Low-High (Outlier)': '#abd9e9',
        'Not Significant': '#eeeeee'
    }
    
    gdf['color'] = gdf['LISA_Cluster'].map(colors)
    
    # Plot
    gdf.plot(ax=ax, color=gdf['color'], markersize=5, alpha=0.7)
    
    # Legend
    legend_elements = [Patch(facecolor=color, label=label) 
                       for label, color in colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_title(f"LISA Cluster Map - {MONTH_NAMES.get(month_str, month_str)}\n"
                 f"(Local Indicators of Spatial Association)", fontsize=14)
    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    
    # Add summary stats as text
    sig_count = (gdf['LISA_sig']).sum()
    total = len(gdf)
    ax.text(0.02, 0.98, f"Significant clusters: {sig_count}/{total} ({sig_count/total*100:.1f}%)",
            transform=ax.transAxes, fontsize=10, va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, f"LISA_Cluster_{month_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")

def plot_getis_ord_map(gdf, month_str):
    """Create Getis-Ord Gi* hot/cold spot map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color mapping for spots
    colors = {
        'Hot Spot (99%)': '#d7191c',
        'Hot Spot (95%)': '#fdae61',
        'Hot Spot (90%)': '#fee08b',
        'Not Significant': '#eeeeee',
        'Cold Spot (90%)': '#d0d1e6',
        'Cold Spot (95%)': '#74a9cf',
        'Cold Spot (99%)': '#2c7bb6'
    }
    
    gdf['color'] = gdf['Gi_Spot'].map(colors)
    gdf['color'] = gdf['color'].fillna('#eeeeee')
    
    # Plot
    gdf.plot(ax=ax, color=gdf['color'], markersize=5, alpha=0.7)
    
    # Legend (in order)
    order = ['Hot Spot (99%)', 'Hot Spot (95%)', 'Hot Spot (90%)', 
             'Not Significant', 
             'Cold Spot (90%)', 'Cold Spot (95%)', 'Cold Spot (99%)']
    legend_elements = [Patch(facecolor=colors[label], label=label) 
                       for label in order if label in gdf['Gi_Spot'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_title(f"Getis-Ord Gi* Hot/Cold Spot Analysis - {MONTH_NAMES.get(month_str, month_str)}", 
                 fontsize=14)
    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, f"Getis_Ord_{month_str}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_single_month(month_str):
    """Run complete spatial autocorrelation analysis for one month."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {MONTH_NAMES.get(month_str, month_str)} (Month {month_str})")
    print('='*60)
    
    # Load raster
    raster_path = os.path.join(RAW_TIF_DIR, f"Tianjin_Monthly_Median_{month_str}.tif")
    if not os.path.exists(raster_path):
        print(f"  ✗ Raster not found: {raster_path}")
        return None
    
    # Create sample points and extract LST
    gdf = create_sample_grid(raster_path)
    gdf = extract_lst_at_points(gdf, raster_path)
    
    if len(gdf) < 100:
        print(f"  ✗ Too few valid samples ({len(gdf)})")
        return None
    
    # 1. Global Moran's I
    moran, w, global_results = calculate_global_morans_i(gdf)
    plot_moran_scatterplot(moran, gdf, month_str)
    
    # 2. Local Moran's I (LISA)
    lisa, gdf = calculate_lisa(gdf, w)
    plot_lisa_cluster_map(gdf, month_str)
    
    # 3. Getis-Ord Gi*
    gi, gdf = calculate_getis_ord(gdf)
    plot_getis_ord_map(gdf, month_str)
    
    # Save results
    output_csv = os.path.join(OUTPUT_DIR, f"Spatial_Stats_{month_str}.csv")
    gdf.drop(columns=['geometry', 'color'], errors='ignore').to_csv(output_csv, index=False)
    print(f"  ✓ Results saved: {output_csv}")
    
    # Save as shapefile
    output_shp = os.path.join(OUTPUT_DIR, f"Spatial_Stats_{month_str}.shp")
    gdf_save = gdf.drop(columns=['color'], errors='ignore')
    gdf_save.to_file(output_shp)
    print(f"  ✓ Shapefile saved: {output_shp}")
    
    return {
        'month': month_str,
        'n_samples': len(gdf),
        **global_results,
        'HH_count': (gdf['LISA_Cluster'] == 'High-High (Hot Spot)').sum(),
        'LL_count': (gdf['LISA_Cluster'] == 'Low-Low (Cold Spot)').sum(),
        'HotSpot_count': gdf['Gi_Spot'].str.contains('Hot').sum(),
        'ColdSpot_count': gdf['Gi_Spot'].str.contains('Cold').sum()
    }

def analyze_all_months():
    """Run analysis for all 12 months and create summary."""
    print("\n" + "="*60)
    print("SPATIAL AUTOCORRELATION ANALYSIS - ALL MONTHS")
    print("="*60)
    
    setup_directories()
    
    if not SPATIAL_LIBS_AVAILABLE:
        print("\n❌ Cannot proceed without spatial libraries.")
        print("   Install with: pip install libpysal esda splot")
        return
    
    all_results = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        result = analyze_single_month(month_str)
        if result:
            all_results.append(result)
    
    # Create summary DataFrame
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(OUTPUT_DIR, "Spatial_Autocorrelation_Summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved: {summary_path}")
        
        # Plot summary
        plot_monthly_summary(summary_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

def plot_monthly_summary(summary_df):
    """Plot monthly Moran's I values."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    months = [MONTH_NAMES[m] for m in summary_df['month']]
    
    # Plot 1: Moran's I over months
    ax1 = axes[0]
    colors = ['#d7191c' if sig else '#999999' for sig in summary_df['Significant']]
    ax1.bar(months, summary_df['Morans_I'], color=colors, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel("Moran's I", fontsize=12)
    ax1.set_title("Global Spatial Autocorrelation of LST by Month", fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add significance indicators
    for i, (sig, p) in enumerate(zip(summary_df['Significant'], summary_df['P_value_sim'])):
        if sig:
            ax1.text(i, summary_df['Morans_I'].iloc[i] + 0.01, '*', 
                    ha='center', fontsize=14, color='red')
    
    # Plot 2: Hot/Cold spot counts
    ax2 = axes[1]
    x = np.arange(len(months))
    width = 0.35
    
    ax2.bar(x - width/2, summary_df['HotSpot_count'], width, label='Hot Spots', color='#d7191c')
    ax2.bar(x + width/2, summary_df['ColdSpot_count'], width, label='Cold Spots', color='#2c7bb6')
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Number of Significant Hot/Cold Spots by Month', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Monthly_Spatial_Autocorrelation_Summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Summary plot saved: {output_path}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run single month (for testing)
    # analyze_single_month('07')
    
    # Run all months
    analyze_all_months()
