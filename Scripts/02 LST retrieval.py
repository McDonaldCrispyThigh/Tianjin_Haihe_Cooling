"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 02 LST Retrieval & Buffer Analysis
DESCRIPTION: 
    - Create multi-ring buffers around Haihe River
    - Calculate zonal statistics (mean LST per buffer zone)
    - Export results for gradient analysis
AUTHOR: Congyuan Zheng
DATE: 2026-02
=============================================================================
"""

import arcpy
from arcpy.sa import *
import os
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling"

# Input paths
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")
HAIHE_RIVER = os.path.join(VECTOR_DIR, "Haihe_River.shp")  # Main river boundary

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
BUFFER_OUTPUT = os.path.join(VECTOR_DIR, "Haihe_Buffers.shp")
STATS_OUTPUT = os.path.join(PROJECT_ROOT, "Data")

# Buffer configuration
BUFFER_DISTANCES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 
                    350, 400, 450, 500, 600, 700, 800, 900, 1000]  # meters
BUFFER_UNIT = "Meters"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Configure ArcPy environment."""
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32650)
    
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        print("✓ Spatial Analyst extension ready.")
    else:
        raise RuntimeError("Spatial Analyst extension not available!")

# ============================================================================
# MULTI-RING BUFFER CREATION
# ============================================================================

def create_multi_ring_buffer(input_feature, output_feature, distances):
    """
    Create multiple ring buffers around the Haihe River.
    
    Each ring represents a distance zone (e.g., 0-30m, 30-60m, etc.)
    """
    print("\n" + "="*60)
    print("STEP 1: Creating Multi-Ring Buffers")
    print("="*60)
    
    # Format distances as string for tool input
    distance_str = ";".join([str(d) for d in distances])
    
    print(f"  Input feature: {input_feature}")
    print(f"  Buffer distances: {distances[0]}m to {distances[-1]}m")
    print(f"  Number of rings: {len(distances)}")
    
    # Create multiple ring buffer
    arcpy.analysis.MultipleRingBuffer(
        Input_Features=input_feature,
        Output_Feature_class=output_feature,
        Distances=distances,
        Buffer_Unit=BUFFER_UNIT,
        Field_Name="distance",
        Dissolve_Option="ALL",
        Outside_Polygons_Only="OUTSIDE_ONLY"
    )
    
    print(f"  ✓ Buffers created: {output_feature}")
    
    # Count features
    count = int(arcpy.GetCount_management(output_feature)[0])
    print(f"  ✓ Total buffer zones: {count}")
    
    return output_feature

# ============================================================================
# ZONAL STATISTICS
# ============================================================================

def calculate_zonal_statistics(buffer_feature, lst_raster, month_str, output_table):
    """
    Calculate mean LST for each buffer zone.
    
    This creates the data needed for the "Distance vs Temperature" curve.
    """
    print(f"\n  Calculating zonal statistics for Month {month_str}...")
    
    # Run Zonal Statistics as Table
    arcpy.sa.ZonalStatisticsAsTable(
        in_zone_data=buffer_feature,
        zone_field="distance",
        in_value_raster=lst_raster,
        out_table=output_table,
        ignore_nodata="DATA",
        statistics_type="MEAN"
    )
    
    print(f"    ✓ Statistics table: {output_table}")
    return output_table

def export_stats_to_excel(gdb_table, output_excel, month_str):
    """
    Convert geodatabase table to Excel for visualization.
    """
    print(f"\n  Exporting statistics to Excel...")
    
    # Read table into pandas DataFrame
    fields = ["distance", "MEAN", "COUNT", "AREA"]
    available_fields = [f.name for f in arcpy.ListFields(gdb_table)]
    fields_to_use = [f for f in fields if f in available_fields]
    
    data = []
    with arcpy.da.SearchCursor(gdb_table, fields_to_use) as cursor:
        for row in cursor:
            data.append(row)
    
    df = pd.DataFrame(data, columns=fields_to_use)
    df['Month'] = month_str
    
    # Sort by distance
    df = df.sort_values('distance')
    
    # Export to Excel
    df.to_excel(output_excel, index=False)
    print(f"    ✓ Excel exported: {output_excel}")
    
    return df

# ============================================================================
# BATCH PROCESSING - ALL MONTHS
# ============================================================================

def process_all_months(buffer_feature):
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
            print(f"  ⚠ Month {month_str}: LST raster not found, skipping...")
            continue
        
        print(f"\n{'─'*40}")
        print(f"Processing Month {month_str}")
        print(f"{'─'*40}")
        
        # Output table in geodatabase
        gdb_path = os.path.join(PROJECT_ROOT, "Tianjin_Haihe_Cooling.gdb")
        stats_table = os.path.join(gdb_path, f"Stats_LST_Month_{month_str}")
        
        # Calculate zonal statistics
        calculate_zonal_statistics(buffer_feature, lst_raster, month_str, stats_table)
        
        # Export to Excel
        excel_output = os.path.join(STATS_OUTPUT, f"Gradient_Month_{month_str}.xlsx")
        df = export_stats_to_excel(stats_table, excel_output, month_str)
        
        all_results.append(df)
    
    # Combine all months into one master Excel file
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        master_excel = os.path.join(STATS_OUTPUT, "All_Months_Gradient.xlsx")
        combined_df.to_excel(master_excel, index=False)
        print(f"\n✓ Master gradient file: {master_excel}")
    
    return all_results

# ============================================================================
# COOLING THRESHOLD ANALYSIS
# ============================================================================

def analyze_cooling_threshold(excel_file, month_str):
    """
    Determine the Threshold Value of Efficiency (TVoE).
    
    The TVoE is the distance at which the cooling effect becomes negligible.
    We detect this by finding where the temperature gradient flattens.
    """
    print(f"\n  Analyzing cooling threshold for Month {month_str}...")
    
    df = pd.read_excel(excel_file)
    df = df.sort_values('distance')
    
    # Calculate temperature gradient (rate of change)
    df['temp_gradient'] = df['MEAN'].diff() / df['distance'].diff()
    
    # Find where gradient drops below threshold (e.g., 0.005°C/m)
    gradient_threshold = 0.005
    threshold_rows = df[abs(df['temp_gradient']) < gradient_threshold]
    
    if not threshold_rows.empty:
        tvoe = threshold_rows.iloc[0]['distance']
        print(f"    ✓ Cooling Threshold Distance (TVoE): {tvoe} meters")
    else:
        tvoe = None
        print(f"    ⚠ Could not determine TVoE (gradient never flattens)")
    
    # Calculate cooling intensity
    water_temp = df.iloc[0]['MEAN']  # Temperature at river edge
    urban_temp = df.iloc[-1]['MEAN']  # Temperature at 1000m
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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main LST retrieval and buffer analysis workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - LST RETRIEVAL & BUFFER ANALYSIS")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check if river boundary exists
    if not os.path.exists(HAIHE_RIVER):
        print(f"ERROR: Haihe River shapefile not found at {HAIHE_RIVER}")
        print("Please ensure you have the river boundary file.")
        return
    
    # Step 1: Create multi-ring buffers
    buffer_feature = create_multi_ring_buffer(
        HAIHE_RIVER, 
        BUFFER_OUTPUT, 
        BUFFER_DISTANCES
    )
    
    # Step 2: Process all months
    results = process_all_months(buffer_feature)
    
    # Step 3: Analyze cooling threshold for July (peak summer)
    print("\n" + "="*60)
    print("STEP 3: Cooling Threshold Analysis (July)")
    print("="*60)
    
    july_excel = os.path.join(STATS_OUTPUT, "Gradient_Month_07.xlsx")
    if os.path.exists(july_excel):
        threshold_results = analyze_cooling_threshold(july_excel, "07")
    
    # Final summary
    print("\n" + "="*60)
    print("LST RETRIEVAL COMPLETE")
    print("="*60)
    
    # Check in extension
    arcpy.CheckInExtension("Spatial")


if __name__ == "__main__":
    main()

