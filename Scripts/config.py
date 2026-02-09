"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
MODULE: Shared Configuration
DESCRIPTION:
    Central configuration file for all analysis scripts.
    All paths, constants, and shared parameters are defined here.
AUTHOR: Congyuan Zheng
DATE: 2026-02
=============================================================================
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")
HAIHE_RIVER = os.path.join(VECTOR_DIR, "Haihe_River.shp")

# Output data
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
GWR_RESULTS_DIR = os.path.join(PROJECT_ROOT, "Data", "GWR_Results")
GWR_MULTI_DIR = os.path.join(PROJECT_ROOT, "Data", "GWR_Multivariate")
SPATIAL_STATS_DIR = os.path.join(PROJECT_ROOT, "Data", "Spatial_Stats")

# Map outputs (organized by analysis type)
MAPS_DIR = os.path.join(PROJECT_ROOT, "Maps")
MAPS_BUFFER = os.path.join(MAPS_DIR, "Buffer_Analysis")
MAPS_GWR_SINGLE = os.path.join(MAPS_DIR, "GWR_SingleVar")
MAPS_GWR_MULTI = os.path.join(MAPS_DIR, "GWR_Multivariate")
MAPS_SPATIAL = os.path.join(MAPS_DIR, "Spatial_Autocorrelation")
MAPS_SEASONAL = os.path.join(MAPS_DIR, "Seasonal_Analysis")
MAPS_RIVERSIDE = os.path.join(MAPS_DIR, "Riverside_Analysis")

# ============================================================================
# MONTH LABELS (shared across all scripts)
# ============================================================================

MONTH_NAMES = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December'
}

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

MONTH_FULL = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

# ============================================================================
# SEASON DEFINITIONS (Northern Hemisphere)
# ============================================================================

SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

SEASON_COLORS = {
    'Winter': '#2c7bb6',
    'Spring': '#7fbc41',
    'Summer': '#d7191c',
    'Autumn': '#fdae61'
}

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Buffer analysis (Script 02)
BUFFER_DISTANCES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300,
                    350, 400, 450, 500, 600, 700, 800, 900, 1000]

# GWR parameters
GWR_BANDWIDTH = 500        # meters
SAMPLE_SPACING_GWR = 100   # Script 03
SAMPLE_SPACING_SPATIAL = 150  # Script 04
SAMPLE_SPACING_MULTI = 120   # Script 06

# Spatial autocorrelation
SIGNIFICANCE_LEVEL = 0.05
N_PERMUTATIONS = 999

# Preprocessing
NDWI_THRESHOLD = 0.1

# ============================================================================
# STUDY AREA
# ============================================================================

STUDY_AREA = {
    'name': 'Tianjin Central Districts',
    'districts': ['Heping', 'Nankai', 'Hexi', 'Hedong', 'Hebei', 'Hongqiao'],
    'bbox': {
        'west': 116.9528,
        'east': 117.8853,
        'south': 38.8987,
        'north': 39.3504
    },
    'crs': 'EPSG:32650',
    'resolution': 30  # meters
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_month_name(month_str):
    """Get full month name from '01'-'12' string."""
    return MONTH_NAMES.get(month_str, month_str)
