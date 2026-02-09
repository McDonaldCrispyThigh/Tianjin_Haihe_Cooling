"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 07 Riverside GWR Analysis
DESCRIPTION: 
    - Focus analysis on Haihe River corridor only
    - Analyze GWR coefficients by distance bands from river
    - Show how Distance, NDVI, NDBI influence LST near the river
    - Generate publication-ready figures for river cooling study
AUTHOR: Congyuan Zheng
DATE: 2026-02
=============================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches
from shapely.geometry import Point
import os
import warnings
warnings.filterwarnings('ignore')

# Import shared configuration
from config import (GWR_MULTI_DIR, VECTOR_DIR, HAIHE_RIVER,
                    MAPS_RIVERSIDE, MONTH_NAMES, ensure_dirs)

# ============================================================================
# CONFIGURATION
# ============================================================================

GWR_RESULTS_DIR = GWR_MULTI_DIR
OUTPUT_DIR = MAPS_RIVERSIDE

# Analysis configuration
MAX_DISTANCE = 1500  # Only analyze within 1.5km of river
DISTANCE_BANDS = [0, 100, 200, 300, 500, 750, 1000, 1500]  # Distance bins

# Season definitions (string month keys for this script)
SEASONS = {
    'Spring': ['03', '04', '05'],
    'Summer': ['06', '07', '08'],
    'Autumn': ['09', '10', '11'],
    'Winter': ['12', '01', '02']
}

# ============================================================================
# SETUP
# ============================================================================

def setup():
    """Create output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output directory: {OUTPUT_DIR}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_gwr_results(month_str):
    """Load GWR results for a specific month."""
    csv_path = os.path.join(GWR_RESULTS_DIR, f"GWR_Multivariate_{month_str}.csv")
    
    if not os.path.exists(csv_path):
        print(f"  [WARNING] No data for month {month_str}")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def filter_riverside_points(df, max_distance=MAX_DISTANCE):
    """Filter points within specified distance from river."""
    df_filtered = df[df['Distance'] <= max_distance].copy()
    print(f"    Filtered: {len(df_filtered)}/{len(df)} points within {max_distance}m of river")
    return df_filtered

def assign_distance_bands(df, bands=DISTANCE_BANDS):
    """Assign each point to a distance band."""
    df['Distance_Band'] = pd.cut(
        df['Distance'], 
        bins=bands, 
        labels=[f"{bands[i]}-{bands[i+1]}m" for i in range(len(bands)-1)]
    )
    return df

# ============================================================================
# ANALYSIS BY DISTANCE BANDS
# ============================================================================

def analyze_by_distance_bands(df, month_str):
    """Calculate mean coefficients for each distance band."""
    results = df.groupby('Distance_Band').agg({
        'Coef_Distance': ['mean', 'std'],
        'Coef_NDVI': ['mean', 'std'],
        'Coef_NDBI': ['mean', 'std'],
        'Local_R2': ['mean', 'std'],
        'LST': ['mean', 'std'],
        'Distance': 'count'
    }).reset_index()
    
    # Flatten column names
    results.columns = ['Distance_Band', 
                       'Coef_Distance_mean', 'Coef_Distance_std',
                       'Coef_NDVI_mean', 'Coef_NDVI_std',
                       'Coef_NDBI_mean', 'Coef_NDBI_std',
                       'R2_mean', 'R2_std',
                       'LST_mean', 'LST_std',
                       'Sample_Count']
    
    results['Month'] = month_str
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_coefficient_by_distance(all_results, season='Summer'):
    """Plot how GWR coefficients change with distance from river."""
    
    # Filter for specific season
    season_months = SEASONS[season]
    season_data = all_results[all_results['Month'].isin(season_months)]
    
    if len(season_data) == 0:
        print(f"  No data for {season}")
        return
    
    # Calculate season average
    season_avg = season_data.groupby('Distance_Band').agg({
        'Coef_Distance_mean': 'mean',
        'Coef_NDVI_mean': 'mean',
        'Coef_NDBI_mean': 'mean',
        'R2_mean': 'mean',
        'LST_mean': 'mean'
    }).reset_index()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'GWR Coefficients by Distance from Haihe River ({season})', 
                 fontsize=14, fontweight='bold')
    
    x = range(len(season_avg))
    x_labels = season_avg['Distance_Band'].astype(str)
    
    # Plot 1: Distance Coefficient
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, season_avg['Coef_Distance_mean'], color='steelblue', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Coefficient (standardized)', fontsize=10)
    ax1.set_title('Distance to River Effect on LST', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    # Add value labels
    for bar, val in zip(bars1, season_avg['Coef_Distance_mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: NDVI Coefficient (Vegetation effect)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, season_avg['Coef_NDVI_mean'], color='forestgreen', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Coefficient (standardized)', fontsize=10)
    ax2.set_title('Vegetation (NDVI) Effect on LST', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars2, season_avg['Coef_NDVI_mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: NDBI Coefficient (Built-up effect)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, season_avg['Coef_NDBI_mean'], color='coral', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax3.set_ylabel('Coefficient (standardized)', fontsize=10)
    ax3.set_title('Built-up Area (NDBI) Effect on LST', fontsize=11)
    ax3.set_xlabel('Distance Band from River', fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars3, season_avg['Coef_NDBI_mean']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Model R² and LST
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1, = ax4.plot(x, season_avg['R2_mean'], 'b-o', linewidth=2, markersize=8, label='Local R²')
    line2, = ax4_twin.plot(x, season_avg['LST_mean'], 'r-s', linewidth=2, markersize=8, label='Mean LST')
    
    ax4.set_ylabel('Local R²', color='blue', fontsize=10)
    ax4_twin.set_ylabel('Mean LST (°C)', color='red', fontsize=10)
    ax4.set_title('Model Fit (R²) and Mean LST', fontsize=11)
    ax4.set_xlabel('Distance Band from River', fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.legend([line1, line2], ['Local R²', 'Mean LST'], loc='upper right')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'Coefficients_by_Distance_{season}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

def plot_seasonal_comparison(all_results):
    """Compare coefficients across seasons."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Seasonal Comparison of GWR Coefficients\n(Haihe Riverside, 0-1500m)', 
                 fontsize=13, fontweight='bold')
    
    colors = {'Spring': 'mediumseagreen', 'Summer': 'orangered', 
              'Autumn': 'goldenrod', 'Winter': 'steelblue'}
    
    coef_cols = ['Coef_Distance_mean', 'Coef_NDVI_mean', 'Coef_NDBI_mean']
    titles = ['Distance Effect', 'Vegetation (NDVI) Effect', 'Built-up (NDBI) Effect']
    
    for idx, (coef_col, title) in enumerate(zip(coef_cols, titles)):
        ax = axes[idx]
        
        for season, months in SEASONS.items():
            season_data = all_results[all_results['Month'].isin(months)]
            if len(season_data) == 0:
                continue
            
            season_avg = season_data.groupby('Distance_Band')[coef_col].mean().reset_index()
            x = range(len(season_avg))
            ax.plot(x, season_avg[coef_col], '-o', color=colors[season], 
                    label=season, linewidth=2, markersize=6)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Distance Band from River', fontsize=10)
        ax.set_ylabel('Coefficient (standardized)', fontsize=10)
        
        # Use first season's data for x-axis labels
        first_season_data = all_results[all_results['Month'].isin(SEASONS['Summer'])]
        if len(first_season_data) > 0:
            labels = first_season_data.groupby('Distance_Band').size().index.astype(str)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'Seasonal_Comparison_Riverside.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

def plot_variable_importance_riverside(all_results):
    """Show relative importance of each variable by distance band."""
    
    # Focus on summer months
    summer_months = ['06', '07', '08']
    summer_data = all_results[all_results['Month'].isin(summer_months)]
    
    if len(summer_data) == 0:
        print("  No summer data available")
        return
    
    # Calculate absolute mean coefficients for importance
    importance = summer_data.groupby('Distance_Band').agg({
        'Coef_Distance_mean': lambda x: np.abs(x).mean(),
        'Coef_NDVI_mean': lambda x: np.abs(x).mean(),
        'Coef_NDBI_mean': lambda x: np.abs(x).mean()
    }).reset_index()
    
    # Normalize to show relative importance
    total = importance['Coef_Distance_mean'] + importance['Coef_NDVI_mean'] + importance['Coef_NDBI_mean']
    importance['Distance_pct'] = importance['Coef_Distance_mean'] / total * 100
    importance['NDVI_pct'] = importance['Coef_NDVI_mean'] / total * 100
    importance['NDBI_pct'] = importance['Coef_NDBI_mean'] / total * 100
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(importance))
    width = 0.6
    
    p1 = ax.bar(x, importance['Distance_pct'], width, label='Distance to River', color='steelblue')
    p2 = ax.bar(x, importance['NDVI_pct'], width, bottom=importance['Distance_pct'], 
                label='Vegetation (NDVI)', color='forestgreen')
    p3 = ax.bar(x, importance['NDBI_pct'], width, 
                bottom=importance['Distance_pct'] + importance['NDVI_pct'],
                label='Built-up (NDBI)', color='coral')
    
    ax.set_ylabel('Relative Importance (%)', fontsize=11)
    ax.set_xlabel('Distance Band from Haihe River', fontsize=11)
    ax.set_title('Relative Importance of Cooling Factors by Distance (Summer)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(importance['Distance_Band'].astype(str), rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    # Add percentage labels
    for i in range(len(importance)):
        # Distance label
        if importance['Distance_pct'].iloc[i] > 5:
            ax.text(i, importance['Distance_pct'].iloc[i]/2, 
                    f"{importance['Distance_pct'].iloc[i]:.0f}%", 
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        # NDVI label
        if importance['NDVI_pct'].iloc[i] > 5:
            ax.text(i, importance['Distance_pct'].iloc[i] + importance['NDVI_pct'].iloc[i]/2,
                    f"{importance['NDVI_pct'].iloc[i]:.0f}%",
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        # NDBI label
        if importance['NDBI_pct'].iloc[i] > 5:
            ax.text(i, importance['Distance_pct'].iloc[i] + importance['NDVI_pct'].iloc[i] + importance['NDBI_pct'].iloc[i]/2,
                    f"{importance['NDBI_pct'].iloc[i]:.0f}%",
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'Variable_Importance_by_Distance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

def plot_cooling_gradient(all_results):
    """Show LST gradient from river and model R²."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'Spring': 'mediumseagreen', 'Summer': 'orangered', 
              'Autumn': 'goldenrod', 'Winter': 'steelblue'}
    
    # Plot 1: LST by distance (cooling gradient)
    ax1 = axes[0]
    for season, months in SEASONS.items():
        season_data = all_results[all_results['Month'].isin(months)]
        if len(season_data) == 0:
            continue
        
        season_avg = season_data.groupby('Distance_Band')['LST_mean'].mean().reset_index()
        x = range(len(season_avg))
        ax1.plot(x, season_avg['LST_mean'], '-o', color=colors[season], 
                 label=season, linewidth=2, markersize=8)
    
    ax1.set_title('LST Cooling Gradient from Haihe River', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance Band from River', fontsize=11)
    ax1.set_ylabel('Mean LST (°C)', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis labels
    first_data = all_results[all_results['Month'].isin(SEASONS['Summer'])]
    if len(first_data) > 0:
        labels = first_data.groupby('Distance_Band').size().index.astype(str)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    # Plot 2: Model R² by distance
    ax2 = axes[1]
    for season, months in SEASONS.items():
        season_data = all_results[all_results['Month'].isin(months)]
        if len(season_data) == 0:
            continue
        
        season_avg = season_data.groupby('Distance_Band')['R2_mean'].mean().reset_index()
        x = range(len(season_avg))
        ax2.plot(x, season_avg['R2_mean'], '-s', color=colors[season], 
                 label=season, linewidth=2, markersize=8)
    
    ax2.set_title('GWR Model Explanatory Power (R²)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance Band from River', fontsize=11)
    ax2.set_ylabel('Mean Local R²', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    if len(first_data) > 0:
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'Cooling_Gradient_and_R2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output_path}")

def create_summary_table(all_results):
    """Create a summary table of key findings."""
    
    # Overall summary by season
    summary_list = []
    
    for season, months in SEASONS.items():
        season_data = all_results[all_results['Month'].isin(months)]
        if len(season_data) == 0:
            continue
        
        # Near river (0-300m) vs far (750-1500m)
        near = season_data[season_data['Distance_Band'].isin(['0-100m', '100-200m', '200-300m'])]
        far = season_data[season_data['Distance_Band'].isin(['750-1000m', '1000-1500m'])]
        
        if len(near) > 0 and len(far) > 0:
            summary_list.append({
                'Season': season,
                'LST_Near_River': near['LST_mean'].mean(),
                'LST_Far': far['LST_mean'].mean(),
                'Cooling_Effect_C': far['LST_mean'].mean() - near['LST_mean'].mean(),
                'R2_Near_River': near['R2_mean'].mean(),
                'R2_Far': far['R2_mean'].mean(),
                'Coef_Distance_Near': near['Coef_Distance_mean'].mean(),
                'Coef_NDVI_Near': near['Coef_NDVI_mean'].mean(),
                'Coef_NDBI_Near': near['Coef_NDBI_mean'].mean()
            })
    
    summary_df = pd.DataFrame(summary_list)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'Riverside_Analysis_Summary.csv')
    summary_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n  [OK] Summary saved: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("RIVERSIDE ANALYSIS SUMMARY (0-1500m from Haihe River)")
    print("="*70)
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['Season']}:")
        print(f"  LST near river (0-300m): {row['LST_Near_River']:.2f}°C")
        print(f"  LST far from river (750-1500m): {row['LST_Far']:.2f}°C")
        print(f"  Cooling effect: {row['Cooling_Effect_C']:.2f}°C")
        print(f"  Model R² near river: {row['R2_Near_River']:.3f}")
        print(f"  Key coefficients (near river):")
        print(f"    - Distance: {row['Coef_Distance_Near']:.4f}")
        print(f"    - NDVI: {row['Coef_NDVI_Near']:.4f}")
        print(f"    - NDBI: {row['Coef_NDBI_Near']:.4f}")
    
    return summary_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("RIVERSIDE GWR ANALYSIS")
    print("Focus: Haihe River Corridor (0-1500m)")
    print("="*60)
    
    setup()
    
    # Process all months
    all_band_results = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        print(f"\nProcessing {MONTH_NAMES[month_str]} (Month {month_str})...")
        
        # Load data
        df = load_gwr_results(month_str)
        if df is None:
            continue
        
        # Filter to riverside
        df_riverside = filter_riverside_points(df, MAX_DISTANCE)
        
        # Assign distance bands
        df_riverside = assign_distance_bands(df_riverside, DISTANCE_BANDS)
        
        # Analyze by distance bands
        band_results = analyze_by_distance_bands(df_riverside, month_str)
        all_band_results.append(band_results)
    
    # Combine all results
    all_results = pd.concat(all_band_results, ignore_index=True)
    
    # Save detailed results
    detail_path = os.path.join(OUTPUT_DIR, 'Riverside_GWR_by_Distance_Band.csv')
    all_results.to_csv(detail_path, index=False, float_format='%.4f')
    print(f"\n[OK] Detailed results saved: {detail_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_coefficient_by_distance(all_results, 'Summer')
    plot_coefficient_by_distance(all_results, 'Winter')
    plot_seasonal_comparison(all_results)
    plot_variable_importance_riverside(all_results)
    plot_cooling_gradient(all_results)
    
    # Create summary
    summary = create_summary_table(all_results)
    
    print("\n" + "="*60)
    print("RIVERSIDE ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
