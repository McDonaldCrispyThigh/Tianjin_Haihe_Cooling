"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 05 Seasonal Time Series Analysis
DESCRIPTION: 
    - Analyze seasonal patterns in UCI (Urban Cooling Island) intensity
    - Fit sinusoidal model to capture annual cycle
    - Calculate cooling stability index (CV)
    - Identify peak cooling months and phase relationships
    - Generate publication-ready seasonal charts
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: pandas, numpy, scipy, matplotlib
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Import shared configuration
from config import (DATA_DIR, MAPS_SEASONAL, MONTH_ABBR as MONTH_NAMES,
                    MONTH_FULL, SEASONS, SEASON_COLORS, ensure_dirs)

# ============================================================================
# CONFIGURATION
# ============================================================================

MAPS_DIR = MAPS_SEASONAL

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create output directories."""
    os.makedirs(MAPS_DIR, exist_ok=True)
    print(f"✓ Directory ready: {MAPS_DIR}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_gradient_data():
    """Load all monthly gradient data."""
    print("\n" + "="*60)
    print("LOADING MONTHLY GRADIENT DATA")
    print("="*60)
    
    all_data = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filepath = os.path.join(DATA_DIR, f"Gradient_Month_{month_str}.xlsx")
        
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            df['Month'] = month
            df['Month_Name'] = MONTH_NAMES[month-1]
            all_data.append(df)
            print(f"  ✓ Loaded: Month {month_str} ({len(df)} records)")
        else:
            print(f"  ✗ Missing: {filepath}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n  Total records: {len(combined)}")
        return combined
    else:
        return None

def calculate_monthly_metrics(df):
    """Calculate key cooling metrics for each month."""
    print("\n" + "="*60)
    print("CALCULATING MONTHLY COOLING METRICS")
    print("="*60)
    
    metrics = []
    
    for month in range(1, 13):
        month_data = df[df['Month'] == month].copy()
        
        if len(month_data) == 0:
            continue
        
        # Sort by distance
        month_data = month_data.sort_values('distance')
        
        # Get LST at different distances
        lst_near = month_data[month_data['distance'] <= 60]['MEAN'].mean()  # Near river
        lst_far = month_data[month_data['distance'] >= 800]['MEAN'].mean()  # Far from river
        lst_mid = month_data[(month_data['distance'] >= 200) & 
                             (month_data['distance'] <= 400)]['MEAN'].mean()  # Mid-range
        
        # Cooling intensity (ΔT): difference between far and near
        delta_t = lst_far - lst_near
        
        # Calculate gradient slope (LST change per 100m)
        if len(month_data) >= 3:
            slope, intercept, r, p, se = stats.linregress(
                month_data['distance'], month_data['MEAN']
            )
            gradient_per_100m = slope * 100
        else:
            gradient_per_100m = np.nan
            r = np.nan
        
        # Threshold distance estimation (where cooling becomes < 0.5°C from baseline)
        # Using a simple approach: find where LST reaches 90% of max
        lst_range = lst_far - lst_near
        threshold_lst = lst_near + 0.9 * lst_range
        
        above_threshold = month_data[month_data['MEAN'] >= threshold_lst]
        if len(above_threshold) > 0:
            tvoe = above_threshold['distance'].min()
        else:
            tvoe = month_data['distance'].max()
        
        metrics.append({
            'Month': month,
            'Month_Name': MONTH_NAMES[month-1],
            'LST_Near': lst_near,
            'LST_Far': lst_far,
            'LST_Mid': lst_mid,
            'Delta_T': delta_t,
            'Gradient_per_100m': gradient_per_100m,
            'TVoE': tvoe,  # Threshold Value of Efficiency
            'R_squared': r**2 if not np.isnan(r) else np.nan
        })
        
        print(f"  Month {month:02d}: ΔT = {delta_t:.2f}°C, TVoE = {tvoe:.0f}m, "
              f"Gradient = {gradient_per_100m:.3f}°C/100m")
    
    return pd.DataFrame(metrics)

# ============================================================================
# SINUSOIDAL MODEL FITTING
# ============================================================================

def sinusoidal_model(t, amplitude, phase, offset, trend):
    """
    Sinusoidal function for seasonal pattern:
    T(t) = A * sin(2π*t/12 + φ) + C + m*t
    
    Parameters:
    - amplitude (A): Half the range between max and min
    - phase (φ): Phase shift (peak timing)
    - offset (C): Mean value
    - trend (m): Linear trend (optional, usually ~0)
    """
    return amplitude * np.sin(2 * np.pi * t / 12 + phase) + offset + trend * t

def fit_seasonal_model(metrics_df, variable='Delta_T'):
    """Fit sinusoidal model to seasonal data."""
    print(f"\n  Fitting sinusoidal model to {variable}...")
    
    t = metrics_df['Month'].values
    y = metrics_df[variable].values
    
    # Initial guess
    amplitude_guess = (y.max() - y.min()) / 2
    offset_guess = y.mean()
    phase_guess = 0  # Will be optimized
    trend_guess = 0
    
    try:
        popt, pcov = curve_fit(
            sinusoidal_model, t, y,
            p0=[amplitude_guess, phase_guess, offset_guess, trend_guess],
            maxfev=10000
        )
        
        amplitude, phase, offset, trend = popt
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate R²
        y_pred = sinusoidal_model(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate peak month
        # Peak occurs when sin() = 1, i.e., when 2π*t/12 + φ = π/2
        peak_month = (np.pi/2 - phase) * 12 / (2 * np.pi)
        peak_month = ((peak_month - 1) % 12) + 1  # Normalize to 1-12
        
        results = {
            'amplitude': amplitude,
            'phase': phase,
            'offset': offset,
            'trend': trend,
            'r_squared': r_squared,
            'peak_month': peak_month,
            'peak_month_name': MONTH_NAMES[int(round(peak_month)) - 1],
            'popt': popt,
            'perr': perr
        }
        
        print(f"    ✓ Amplitude = {amplitude:.3f}")
        print(f"    ✓ Mean = {offset:.3f}")
        print(f"    ✓ Peak month ≈ {results['peak_month_name']}")
        print(f"    ✓ R² = {r_squared:.4f}")
        
        return results
        
    except Exception as e:
        print(f"    ✗ Fitting failed: {e}")
        return None

# ============================================================================
# STABILITY ANALYSIS
# ============================================================================

def calculate_stability_metrics(metrics_df):
    """Calculate cooling stability metrics."""
    print("\n" + "="*60)
    print("CALCULATING STABILITY METRICS")
    print("="*60)
    
    stability = {}
    
    for var in ['Delta_T', 'TVoE', 'Gradient_per_100m']:
        values = metrics_df[var].dropna()
        
        mean_val = values.mean()
        std_val = values.std()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
        range_val = values.max() - values.min()
        
        stability[var] = {
            'mean': mean_val,
            'std': std_val,
            'cv': cv,  # Coefficient of Variation (%)
            'min': values.min(),
            'max': values.max(),
            'range': range_val
        }
        
        print(f"\n  {var}:")
        print(f"    Mean ± SD: {mean_val:.3f} ± {std_val:.3f}")
        print(f"    CV: {cv:.1f}%")
        print(f"    Range: {values.min():.3f} - {values.max():.3f}")
    
    return stability

def seasonal_aggregation(metrics_df):
    """Aggregate metrics by season."""
    print("\n" + "="*60)
    print("SEASONAL AGGREGATION")
    print("="*60)
    
    def get_season(month):
        for season, months in SEASONS.items():
            if month in months:
                return season
        return None
    
    metrics_df['Season'] = metrics_df['Month'].apply(get_season)
    
    seasonal_stats = metrics_df.groupby('Season').agg({
        'Delta_T': ['mean', 'std'],
        'TVoE': ['mean', 'std'],
        'Gradient_per_100m': ['mean', 'std'],
        'LST_Near': 'mean',
        'LST_Far': 'mean'
    }).round(3)
    
    print("\n  Seasonal Summary:")
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        if season in metrics_df['Season'].values:
            season_data = metrics_df[metrics_df['Season'] == season]
            print(f"\n  {season}:")
            print(f"    ΔT = {season_data['Delta_T'].mean():.2f} ± {season_data['Delta_T'].std():.2f} °C")
            print(f"    TVoE = {season_data['TVoE'].mean():.0f} ± {season_data['TVoE'].std():.0f} m")
    
    return seasonal_stats

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_seasonal_cycle(metrics_df, fit_results):
    """Plot annual cycle of cooling intensity with fitted curve."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = metrics_df['Month'].values
    delta_t = metrics_df['Delta_T'].values
    
    # Scatter plot of observed values
    ax.scatter(months, delta_t, s=100, c='steelblue', zorder=5, 
               label='Observed ΔT', edgecolor='black')
    
    # Fitted curve
    if fit_results:
        t_smooth = np.linspace(1, 12, 100)
        y_smooth = sinusoidal_model(t_smooth, *fit_results['popt'])
        ax.plot(t_smooth, y_smooth, 'r-', linewidth=2, 
                label=f"Sinusoidal Fit (R² = {fit_results['r_squared']:.3f})")
        
        # Mark peak
        peak_month = fit_results['peak_month']
        peak_value = sinusoidal_model(peak_month, *fit_results['popt'])
        ax.axvline(peak_month, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'Peak: {fit_results["peak_month_name"]}',
                    xy=(peak_month, peak_value), xytext=(peak_month+0.5, peak_value+0.3),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Formatting
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cooling Intensity ΔT (°C)', fontsize=12)
    ax.set_title('Seasonal Cycle of Haihe River Cooling Effect', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add equation text
    if fit_results:
        eq_text = (f"ΔT(t) = {fit_results['amplitude']:.2f}·sin(2πt/12 + {fit_results['phase']:.2f}) "
                   f"+ {fit_results['offset']:.2f}")
        ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=10,
                va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Seasonal_Cycle_DeltaT.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_seasonal_heatmap(metrics_df):
    """Create heatmap of cooling metrics across months and distances."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = [
        ('Delta_T', 'Cooling Intensity (ΔT, °C)', 'RdYlBu_r'),
        ('TVoE', 'Cooling Threshold (TVoE, m)', 'viridis'),
        ('Gradient_per_100m', 'Temperature Gradient (°C/100m)', 'RdYlBu_r')
    ]
    
    for ax, (metric, title, cmap) in zip(axes, metrics_to_plot):
        values = metrics_df.set_index('Month_Name')[metric]
        
        # Create bar chart (easier to read than heatmap for 1D data)
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.8, 12))
        bars = ax.bar(range(12), values.values, color=colors, edgecolor='black')
        
        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Highlight max and min
        max_idx = values.values.argmax()
        min_idx = values.values.argmin()
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(2)
        bars[min_idx].set_edgecolor('blue')
        bars[min_idx].set_linewidth(2)
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Seasonal_Metrics_Comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_seasonal_boxplot(metrics_df):
    """Create boxplot comparing seasons."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    def get_season(month):
        for season, months in SEASONS.items():
            if month in months:
                return season
        return None
    
    metrics_df['Season'] = metrics_df['Month'].apply(get_season)
    
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    colors = [SEASON_COLORS[s] for s in season_order]
    
    for ax, metric in zip(axes, ['Delta_T', 'TVoE', 'Gradient_per_100m']):
        data = [metrics_df[metrics_df['Season'] == s][metric].values for s in season_order]
        
        bp = ax.boxplot(data, labels=season_order, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric.replace('_', ' '), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ")} by Season', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Seasonal_Boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_monthly_gradient_profiles(df):
    """Plot cooling gradient profiles for each month (overlaid)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color gradient from winter (blue) to summer (red)
    colors = plt.cm.coolwarm(np.linspace(0, 1, 12))
    
    for month in range(1, 13):
        month_data = df[df['Month'] == month].sort_values('distance')
        ax.plot(month_data['distance'], month_data['MEAN'], 
                color=colors[month-1], linewidth=1.5, alpha=0.8,
                label=MONTH_NAMES[month-1])
    
    ax.set_xlabel('Distance from Haihe River (m)', fontsize=12)
    ax.set_ylabel('Land Surface Temperature (°C)', fontsize=12)
    ax.set_title('Monthly LST Gradient Profiles (Cooling Decay Curves)', fontsize=14)
    ax.legend(loc='lower right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Cooling Zone', xy=(100, df['MEAN'].min()+2), fontsize=10,
                arrowprops=dict(arrowstyle='<-', color='blue'),
                xytext=(200, df['MEAN'].min()+5))
    ax.annotate('Urban Background', xy=(800, df['MEAN'].max()-2), fontsize=10,
                arrowprops=dict(arrowstyle='<-', color='red'),
                xytext=(600, df['MEAN'].max()-5))
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Monthly_Gradient_Profiles_Overlay.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_phase_analysis(metrics_df):
    """Analyze phase relationship between LST and cooling effect."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    months = metrics_df['Month'].values
    
    # Plot 1: LST (Near and Far)
    ax1 = axes[0]
    ax1.plot(months, metrics_df['LST_Near'], 'b-o', linewidth=2, 
             markersize=8, label='LST Near River (0-60m)')
    ax1.plot(months, metrics_df['LST_Far'], 'r-o', linewidth=2, 
             markersize=8, label='LST Far from River (>800m)')
    ax1.fill_between(months, metrics_df['LST_Near'], metrics_df['LST_Far'], 
                      alpha=0.3, color='gray', label='Cooling Effect')
    ax1.set_ylabel('LST (°C)', fontsize=12)
    ax1.set_title('Land Surface Temperature: Near vs Far from Haihe River', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ΔT
    ax2 = axes[1]
    ax2.bar(months, metrics_df['Delta_T'], color='steelblue', edgecolor='black', alpha=0.8)
    ax2.plot(months, metrics_df['Delta_T'], 'ko-', linewidth=2)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_NAMES)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('ΔT (°C)', fontsize=12)
    ax2.set_title('Cooling Intensity (ΔT = LST_far - LST_near)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Mark peak
    peak_month = metrics_df.loc[metrics_df['Delta_T'].idxmax(), 'Month']
    peak_value = metrics_df['Delta_T'].max()
    ax2.annotate(f'Peak: {MONTH_NAMES[peak_month-1]}\nΔT = {peak_value:.2f}°C',
                 xy=(peak_month, peak_value), xytext=(peak_month+1.5, peak_value-0.5),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(MAPS_DIR, "Phase_Analysis_LST_DeltaT.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_seasonal_analysis():
    """Execute complete seasonal analysis pipeline."""
    print("\n" + "="*60)
    print("SEASONAL TIME SERIES ANALYSIS")
    print("The Blue Spine - Haihe River Cooling Effect")
    print("="*60)
    
    setup_directories()
    
    # Load data
    df = load_gradient_data()
    if df is None:
        print("\n❌ No data found. Exiting.")
        return
    
    # Calculate monthly metrics
    metrics_df = calculate_monthly_metrics(df)
    
    # Fit seasonal model
    fit_results = fit_seasonal_model(metrics_df, 'Delta_T')
    
    # Stability analysis
    stability = calculate_stability_metrics(metrics_df)
    
    # Seasonal aggregation
    seasonal_stats = seasonal_aggregation(metrics_df)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_seasonal_cycle(metrics_df, fit_results)
    plot_seasonal_heatmap(metrics_df)
    plot_seasonal_boxplot(metrics_df)
    plot_monthly_gradient_profiles(df)
    plot_phase_analysis(metrics_df)
    
    # Save results
    output_csv = os.path.join(DATA_DIR, "Seasonal_Metrics_Summary.csv")
    metrics_df.to_csv(output_csv, index=False)
    print(f"\n✓ Metrics saved: {output_csv}")
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n  Annual Mean ΔT: {metrics_df['Delta_T'].mean():.2f} ± {metrics_df['Delta_T'].std():.2f} °C")
    print(f"  Annual Mean TVoE: {metrics_df['TVoE'].mean():.0f} ± {metrics_df['TVoE'].std():.0f} m")
    
    if fit_results:
        print(f"\n  Seasonal Pattern:")
        print(f"    Peak cooling month: {fit_results['peak_month_name']}")
        print(f"    Amplitude: {fit_results['amplitude']:.2f}°C")
        print(f"    Model fit (R²): {fit_results['r_squared']:.4f}")
    
    max_month = metrics_df.loc[metrics_df['Delta_T'].idxmax()]
    min_month = metrics_df.loc[metrics_df['Delta_T'].idxmin()]
    print(f"\n  Strongest cooling: {max_month['Month_Name']} (ΔT = {max_month['Delta_T']:.2f}°C)")
    print(f"  Weakest cooling: {min_month['Month_Name']} (ΔT = {min_month['Delta_T']:.2f}°C)")
    
    print("\n" + "="*60)
    print("SEASONAL ANALYSIS COMPLETE!")
    print("="*60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_seasonal_analysis()
