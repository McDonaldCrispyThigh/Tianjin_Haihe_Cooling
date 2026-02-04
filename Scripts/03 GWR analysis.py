"""
=============================================================================
PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
SCRIPT: 03 GWR Analysis (Geographically Weighted Regression)
DESCRIPTION: 
    - Create sample point grid for GWR analysis
    - Extract LST and distance values to points
    - Run GWR to analyze spatial heterogeneity of cooling effect
    - Visualize results
AUTHOR: Congyuan Zheng
DATE: 2026-02
=============================================================================
"""

import arcpy
from arcpy.sa import *
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"D:\Douments\UNIVERSITY\2025-2026_2\GEOG_4503\Tianjin_Haihe_Cooling"

# Input paths
RAW_TIF_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw_TIF")
VECTOR_DIR = os.path.join(PROJECT_ROOT, "Data", "Vector")
HAIHE_RIVER = os.path.join(VECTOR_DIR, "Haihe_River.shp")

# Output paths
GDB_PATH = os.path.join(PROJECT_ROOT, "Tianjin_Haihe_Cooling.gdb")
MAPS_DIR = os.path.join(PROJECT_ROOT, "Maps")

# GWR Configuration
CELL_SIZE = 150  # meters - grid cell size for sample points
STUDY_AREA_BUFFER = 1500  # meters - analysis extent around river

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Configure ArcPy environment."""
    arcpy.env.workspace = GDB_PATH
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32650)
    
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        print("✓ Spatial Analyst extension ready.")
    else:
        raise RuntimeError("Spatial Analyst extension not available!")

# ============================================================================
# STEP 1: CREATE SAMPLE POINT GRID
# ============================================================================

def create_study_area_extent(river_feature, buffer_distance):
    """
    Create a study area polygon by buffering the river.
    GWR will only run within this extent.
    """
    print("\n" + "="*60)
    print("STEP 1: Creating Study Area Extent")
    print("="*60)
    
    study_area = os.path.join(GDB_PATH, "Study_Area_Extent")
    
    arcpy.analysis.Buffer(
        in_features=river_feature,
        out_feature_class=study_area,
        buffer_distance_or_field=f"{buffer_distance} Meters",
        dissolve_option="ALL"
    )
    
    print(f"  ✓ Study area created: {buffer_distance}m buffer around river")
    return study_area

def create_fishnet_points(study_area, cell_size):
    """
    Create a regular grid of sample points within the study area.
    
    These points will be used to extract raster values and run GWR.
    GWR requires POINT features, not rasters.
    """
    print("\n" + "="*60)
    print("STEP 2: Creating Sample Point Grid")
    print("="*60)
    
    # Get extent of study area
    desc = arcpy.Describe(study_area)
    extent = desc.extent
    
    # Create fishnet
    fishnet_output = os.path.join(GDB_PATH, "Fishnet_Grid")
    fishnet_label = os.path.join(GDB_PATH, "Fishnet_Grid_label")  # Points at cell centers
    
    arcpy.management.CreateFishnet(
        out_feature_class=fishnet_output,
        origin_coord=f"{extent.XMin} {extent.YMin}",
        y_axis_coord=f"{extent.XMin} {extent.YMax}",
        cell_width=cell_size,
        cell_height=cell_size,
        number_rows=None,
        number_columns=None,
        corner_coord=f"{extent.XMax} {extent.YMax}",
        labels="LABELS",
        template=study_area,
        geometry_type="POLYGON"
    )
    
    # Clip points to study area
    sample_points = os.path.join(GDB_PATH, "Sample_Points")
    arcpy.analysis.Clip(fishnet_label, study_area, sample_points)
    
    # Count points
    point_count = int(arcpy.GetCount_management(sample_points)[0])
    print(f"  ✓ Sample points created: {point_count} points")
    print(f"  ✓ Grid resolution: {cell_size}m x {cell_size}m")
    
    return sample_points

# ============================================================================
# STEP 2: EXTRACT VALUES TO POINTS
# ============================================================================

def calculate_distance_to_river(sample_points, river_feature):
    """
    Calculate Euclidean distance from each sample point to the nearest river edge.
    This creates the key explanatory variable for GWR.
    """
    print("\n" + "="*60)
    print("STEP 3: Calculating Distance to River")
    print("="*60)
    
    # Use Near tool to calculate distance
    arcpy.analysis.Near(
        in_features=sample_points,
        near_features=river_feature,
        search_radius=f"{STUDY_AREA_BUFFER} Meters",
        location="NO_LOCATION",
        angle="NO_ANGLE"
    )
    
    # Rename field for clarity
    arcpy.management.AlterField(
        in_table=sample_points,
        field="NEAR_DIST",
        new_field_name="Dist_River",
        new_field_alias="Distance to River (m)"
    )
    
    print(f"  ✓ Distance field added: 'Dist_River'")
    return sample_points

def extract_lst_to_points(sample_points, lst_raster, month_str):
    """
    Extract LST values from raster to sample points.
    """
    print(f"\n  Extracting LST values for Month {month_str}...")
    
    # Extract values
    arcpy.sa.ExtractMultiValuesToPoints(
        in_point_features=sample_points,
        in_rasters=[[lst_raster, f"LST_{month_str}"]],
        bilinear_interpolate_values="BILINEAR"
    )
    
    print(f"    ✓ LST values extracted to field: 'LST_{month_str}'")

# ============================================================================
# STEP 3: RUN GWR ANALYSIS
# ============================================================================

def run_gwr_analysis(sample_points, dependent_var, explanatory_vars, month_str):
    """
    Run Geographically Weighted Regression (GWR).
    
    GWR allows the relationship between LST and distance to vary spatially,
    revealing WHERE the cooling effect is strong vs weak.
    
    Parameters:
    - dependent_var: LST (temperature)
    - explanatory_vars: Distance to River (and optionally NDVI, NDBI)
    - kernel: Adaptive Bi-square (adjusts bandwidth based on local point density)
    """
    print("\n" + "="*60)
    print(f"STEP 4: Running GWR Analysis (Month {month_str})")
    print("="*60)
    
    # Output paths
    gwr_output = os.path.join(GDB_PATH, f"GWR_Results_{month_str}")
    gwr_coeff_raster = os.path.join(GDB_PATH, f"GWR_Coeff_{month_str}")
    
    print(f"  Dependent Variable: {dependent_var}")
    print(f"  Explanatory Variables: {explanatory_vars}")
    print(f"  Kernel Type: ADAPTIVE")
    print(f"  Bandwidth Method: AICc")
    
    try:
        # Run Geographically Weighted Regression
        arcpy.stats.GWR(
            in_features=sample_points,
            dependent_variable=dependent_var,
            explanatory_variables=explanatory_vars,
            out_featureclass=gwr_output,
            kernel_type="ADAPTIVE",
            bandwidth_method="AICc",
            number_of_neighbors=None,
            distance=None,
            out_prediction_featureclass=None,
            prediction_explanatory_variables=None,
            scale=True,
            local_weighting_scheme="BISQUARE"
        )
        
        print(f"  ✓ GWR completed successfully")
        print(f"  ✓ Results saved to: {gwr_output}")
        
    except arcpy.ExecuteError:
        print(f"  ✗ GWR failed: {arcpy.GetMessages(2)}")
        return None
    
    # Extract key statistics from GWR output
    analyze_gwr_results(gwr_output, month_str)
    
    return gwr_output

def analyze_gwr_results(gwr_output, month_str):
    """
    Analyze and report GWR results.
    """
    print(f"\n  Analyzing GWR Results...")
    
    # List fields in GWR output
    fields = [f.name for f in arcpy.ListFields(gwr_output)]
    
    # Look for coefficient and R-squared fields
    coeff_field = [f for f in fields if "Coeff" in f and "Dist" in f]
    r2_field = [f for f in fields if "LocalR2" in f or "R2" in f]
    
    if coeff_field:
        # Calculate statistics on coefficient field
        coeff_name = coeff_field[0]
        
        # Get min, max, mean of coefficients
        with arcpy.da.SearchCursor(gwr_output, [coeff_name]) as cursor:
            coeffs = [row[0] for row in cursor if row[0] is not None]
        
        if coeffs:
            print(f"\n  GWR Coefficient Statistics ({coeff_name}):")
            print(f"    Min: {min(coeffs):.6f}")
            print(f"    Max: {max(coeffs):.6f}")
            print(f"    Mean: {sum(coeffs)/len(coeffs):.6f}")
            
            # Interpretation
            mean_coeff = sum(coeffs)/len(coeffs)
            if mean_coeff > 0:
                print(f"\n  ★ Interpretation: Positive coefficient indicates")
                print(f"    temperature INCREASES with distance from river")
                print(f"    → The river has a COOLING effect")

# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def create_gwr_map(gwr_output, month_str):
    """
    Create a visualization of GWR coefficients.
    
    Note: This creates the feature class for manual styling in ArcGIS Pro.
    For publication-quality maps, use ArcGIS Pro's Layout view.
    """
    print(f"\n  Preparing visualization layer for Month {month_str}...")
    
    # Convert GWR points to raster for visualization
    coeff_raster = os.path.join(GDB_PATH, f"GWR_Coeff_Raster_{month_str}")
    
    # Find coefficient field
    fields = [f.name for f in arcpy.ListFields(gwr_output)]
    coeff_field = [f for f in fields if "Coeff" in f and "Dist" in f]
    
    if coeff_field:
        arcpy.conversion.PointToRaster(
            in_features=gwr_output,
            value_field=coeff_field[0],
            out_rasterdataset=coeff_raster,
            cell_assignment="MEAN",
            cellsize=CELL_SIZE
        )
        print(f"    ✓ Coefficient raster created: {coeff_raster}")
    
    return coeff_raster

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main GWR analysis workflow."""
    print("\n" + "="*60)
    print("THE BLUE SPINE - GWR ANALYSIS MODULE")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check inputs
    if not os.path.exists(HAIHE_RIVER):
        print(f"ERROR: River shapefile not found: {HAIHE_RIVER}")
        return
    
    # Step 1: Create study area and sample points
    study_area = create_study_area_extent(HAIHE_RIVER, STUDY_AREA_BUFFER)
    sample_points = create_fishnet_points(study_area, CELL_SIZE)
    
    # Step 2: Calculate distance to river
    sample_points = calculate_distance_to_river(sample_points, HAIHE_RIVER)
    
    # Step 3: Extract LST for July (peak summer)
    july_lst = os.path.join(RAW_TIF_DIR, "Tianjin_Monthly_Median_07.tif")
    
    if os.path.exists(july_lst):
        extract_lst_to_points(sample_points, july_lst, "07")
        
        # Step 4: Run GWR
        gwr_output = run_gwr_analysis(
            sample_points=sample_points,
            dependent_var="LST_07",
            explanatory_vars=["Dist_River"],
            month_str="07"
        )
        
        # Step 5: Create visualization
        if gwr_output:
            create_gwr_map(gwr_output, "07")
    else:
        print(f"WARNING: July LST raster not found: {july_lst}")
    
    # Final summary
    print("\n" + "="*60)
    print("GWR ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Open ArcGIS Pro and add the GWR_Results layer")
    print("  2. Symbolize by 'Coeff_Dist_River' field")
    print("  3. Areas with HIGHER coefficients = stronger cooling effect")
    print("  4. Export map to Maps/ folder")
    
    # Check in extension
    arcpy.CheckInExtension("Spatial")


if __name__ == "__main__":
    main()

