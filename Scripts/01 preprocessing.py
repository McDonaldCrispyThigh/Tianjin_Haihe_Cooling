"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 01 Preprocessing (Open Source Version - No ArcPy Required)
DESCRIPTION: 
    - Validate downloaded GEE monthly composites
    - Extract LST and NDWI bands from multi-band TIFs
    - Create water body binary mask from NDWI
AUTHOR: Congyuan Zheng
DATE: 2026-02
LIBRARIES: rasterio, numpy, geopandas (open source)
=============================================================================
"""

import rasterio
from rasterio.features import shapes
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS TO MATCH YOUR SYSTEM
# ============================================================================

# Project workspace
PROJECT_ROOT = r"D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling"

# Input/Output directories
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")

# NDWI threshold for water extraction (adjust based on visual inspection)
NDWI_THRESHOLD = 0.1

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Create output directories if they don't exist."""
    print("\n" + "="*60)
    print("Setting up environment...")
    print("="*60)
    
    for directory in [OUTPUT_DIR, VECTOR_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_monthly_composites():
    """
    Check if all 12 monthly composite TIFs exist and are valid.
    Returns a list of valid file paths.
    """
    print("\n" + "="*60)
    print("STEP 1: Validating Monthly Composite Files")
    print("="*60)
    
    valid_files = []
    missing_files = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filename = f"Tianjin_Monthly_Median_{month_str}.tif"
        filepath = os.path.join(RAW_TIF_DIR, filename)
        
        if os.path.exists(filepath):
            # Validate raster properties using rasterio
            with rasterio.open(filepath) as src:
                band_count = src.count
                print(f"  ✓ Month {month_str}: Found ({band_count} bands, {src.width}x{src.height} pixels)")
            valid_files.append(filepath)
        else:
            print(f"  ✗ Month {month_str}: MISSING")
            missing_files.append(filename)
    
    print(f"\nSummary: {len(valid_files)}/12 files found")
    
    if missing_files:
        print(f"Warning: Missing files - {missing_files}")
    
    return valid_files

# ============================================================================
# BAND EXTRACTION
# ============================================================================

def extract_bands(input_tif, month_str):
    """
    Extract LST (Band 1) and NDWI (Band 2) from multi-band GEE output.
    
    GEE Export Structure:
        Band 1: LST_Celsius (Land Surface Temperature in Celsius)
        Band 2: NDWI (Normalized Difference Water Index)
    """
    print(f"\n  Extracting bands for Month {month_str}...")
    
    # Create month-specific output folder
    month_output = os.path.join(OUTPUT_DIR, f"Month_{month_str}")
    if not os.path.exists(month_output):
        os.makedirs(month_output)
    
    # Read the multi-band TIF
    with rasterio.open(input_tif) as src:
        # Get metadata for output files
        meta = src.meta.copy()
        meta.update(count=1)  # Single band output
        
        # Extract LST (Band 1)
        lst_output = os.path.join(month_output, f"LST_{month_str}.tif")
        lst_data = src.read(1)
        with rasterio.open(lst_output, 'w', **meta) as dst:
            dst.write(lst_data, 1)
        print(f"    ✓ LST extracted: {lst_output}")
        
        # Extract NDWI (Band 2)
        ndwi_output = os.path.join(month_output, f"NDWI_{month_str}.tif")
        ndwi_data = src.read(2)
        with rasterio.open(ndwi_output, 'w', **meta) as dst:
            dst.write(ndwi_data, 1)
        print(f"    ✓ NDWI extracted: {ndwi_output}")
    
    return lst_output, ndwi_output

# ============================================================================
# WATER BODY EXTRACTION
# ============================================================================

def extract_water_body(ndwi_raster, month_str, threshold=NDWI_THRESHOLD):
    """
    Convert NDWI raster to binary water mask.
    
    Logic: If NDWI > threshold, pixel = 1 (water), else = 0 (non-water)
    """
    print(f"\n  Creating water binary mask (threshold={threshold})...")
    
    month_output = os.path.join(OUTPUT_DIR, f"Month_{month_str}")
    water_binary_output = os.path.join(month_output, f"Water_Binary_{month_str}.tif")
    
    # Read NDWI and apply threshold
    with rasterio.open(ndwi_raster) as src:
        ndwi_data = src.read(1)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8)  # Binary output
        
        # Apply conditional: NDWI > threshold = 1, else = 0
        water_binary = np.where(ndwi_data > threshold, 1, 0).astype(np.uint8)
        
        # Save binary water mask
        with rasterio.open(water_binary_output, 'w', **meta) as dst:
            dst.write(water_binary, 1)
    
    print(f"    ✓ Water binary saved: {water_binary_output}")
    return water_binary_output

def water_raster_to_polygon(water_binary_raster, month_str):
    """
    Convert binary water raster to polygon shapefile.
    """
    print(f"\n  Converting water raster to polygon...")
    
    output_shp = os.path.join(VECTOR_DIR, f"Water_Polygon_{month_str}.shp")
    
    # Read raster and convert to polygons
    with rasterio.open(water_binary_raster) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Extract shapes (vectorize)
        results = (
            {'properties': {'gridcode': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, transform=transform))
            if v == 1  # Only keep water pixels (value = 1)
        )
        
        # Convert to GeoDataFrame
        geoms = list(results)
        if geoms:
            gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
            gdf.to_file(output_shp)
            print(f"    ✓ Water polygon saved: {output_shp}")
            print(f"    ✓ Total water features: {len(gdf)}")
        else:
            print(f"    ⚠ No water features found!")
            return None
    
    return output_shp

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main preprocessing workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - PREPROCESSING MODULE (Open Source)")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # Validate input files
    valid_files = validate_monthly_composites()
    
    if not valid_files:
        print("ERROR: No valid input files found. Exiting.")
        return
    
    # Process each monthly composite
    print("\n" + "="*60)
    print("STEP 2: Extracting Bands and Creating Water Masks")
    print("="*60)
    
    results = {}
    
    for filepath in valid_files:
        # Extract month number from filename
        filename = os.path.basename(filepath)
        month_str = filename.split("_")[-1].replace(".tif", "")
        
        print(f"\n{'─'*40}")
        print(f"Processing: {filename}")
        print(f"{'─'*40}")
        
        # Extract bands
        lst_path, ndwi_path = extract_bands(filepath, month_str)
        
        # Create water binary mask
        water_binary_path = extract_water_body(ndwi_path, month_str)
        
        # Convert to polygon (optional - only for July as reference)
        if month_str == "07":
            water_polygon = water_raster_to_polygon(water_binary_path, month_str)
            results[month_str] = {
                'LST': lst_path,
                'NDWI': ndwi_path,
                'Water_Binary': water_binary_path,
                'Water_Polygon': water_polygon
            }
        else:
            results[month_str] = {
                'LST': lst_path,
                'NDWI': ndwi_path,
                'Water_Binary': water_binary_path
            }
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Processed {len(results)} monthly composites")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return results


if __name__ == "__main__":
    main()

