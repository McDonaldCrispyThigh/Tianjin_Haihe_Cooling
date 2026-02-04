"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 01 Preprocessing
DESCRIPTION: 
    - Validate downloaded GEE monthly composites
    - Extract LST and NDWI bands from multi-band TIFs
    - Create water body binary mask from NDWI
AUTHOR: Congyuan Zheng
DATE: 2026-02
=============================================================================
"""

import arcpy
from arcpy.sa import *
import os

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
    """Configure ArcPy environment settings."""
    arcpy.env.workspace = RAW_TIF_DIR
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32650)  # UTM Zone 50N
    
    # Check out Spatial Analyst extension
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        print("✓ Spatial Analyst extension checked out successfully.")
    else:
        raise RuntimeError("Spatial Analyst extension is not available!")
    
    # Create output directories if they don't exist
    for directory in [OUTPUT_DIR, VECTOR_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

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
            # Validate raster properties
            raster = arcpy.Raster(filepath)
            band_count = raster.bandCount
            print(f"  ✓ Month {month_str}: Found ({band_count} bands)")
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
    
    # Extract LST (Band 1)
    lst_output = os.path.join(month_output, f"LST_{month_str}.tif")
    arcpy.management.MakeRasterLayer(input_tif, "temp_layer", band_index="1")
    arcpy.management.CopyRaster("temp_layer", lst_output)
    print(f"    ✓ LST extracted: {lst_output}")
    
    # Extract NDWI (Band 2)
    ndwi_output = os.path.join(month_output, f"NDWI_{month_str}.tif")
    arcpy.management.MakeRasterLayer(input_tif, "temp_layer2", band_index="2")
    arcpy.management.CopyRaster("temp_layer2", ndwi_output)
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
    
    # Apply conditional statement
    ndwi = Raster(ndwi_raster)
    water_binary = Con(ndwi > threshold, 1, 0)
    water_binary.save(water_binary_output)
    
    print(f"    ✓ Water binary saved: {water_binary_output}")
    return water_binary_output

def water_raster_to_polygon(water_binary_raster, month_str):
    """
    Convert binary water raster to polygon shapefile.
    """
    print(f"\n  Converting water raster to polygon...")
    
    output_shp = os.path.join(VECTOR_DIR, f"Water_Polygon_{month_str}.shp")
    
    # Raster to Polygon conversion
    arcpy.conversion.RasterToPolygon(
        in_raster=water_binary_raster,
        out_polygon_features=output_shp,
        simplify="SIMPLIFY",
        raster_field="Value"
    )
    
    # Select only water pixels (gridcode = 1)
    water_only_shp = os.path.join(VECTOR_DIR, f"Water_Only_{month_str}.shp")
    arcpy.analysis.Select(output_shp, water_only_shp, "gridcode = 1")
    
    print(f"    ✓ Water polygon saved: {water_only_shp}")
    return water_only_shp

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main preprocessing workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - PREPROCESSING MODULE")
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
    
    # Check in extension
    arcpy.CheckInExtension("Spatial")
    
    return results


if __name__ == "__main__":
    main()

