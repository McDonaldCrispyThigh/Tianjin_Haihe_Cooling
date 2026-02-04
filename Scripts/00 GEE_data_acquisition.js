/**
 * =============================================================================
 * PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
 * SCRIPT: 00 GEE Data Acquisition
 * PLATFORM: Google Earth Engine (https://code.earthengine.google.com/)
 * DESCRIPTION: 
 *   Exports 12 monthly median composite images using Landsat 8/9 data (2020-2025).
 *   Each image contains LST (Land Surface Temperature) and NDWI bands.
 * AUTHOR: Congyuan Zheng
 * DATE: 2026-02
 * =============================================================================
 */

// =============================================================================
// 1. STUDY AREA DEFINITION
// =============================================================================

// Tianjin Central Districts (6 districts: Heping, Nankai, Hexi, Hedong, Hebei, Hongqiao)
var geometry = ee.Geometry.Polygon([
  [
    [116.95280761718944, 38.89871221966305],
    [117.88527221679881, 38.89871221966305],
    [117.88527221679881, 39.35042471792459],
    [116.95280761718944, 39.35042471792459],
    [116.95280761718944, 38.89871221966305]
  ]
], null, false);

Map.centerObject(geometry, 11);

// =============================================================================
// 2. LOAD LANDSAT 8/9 DATASETS
// =============================================================================

var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");
var landsat = l8.merge(l9);

// =============================================================================
// 3. PRE-PROCESSING FUNCTION
// =============================================================================

/**
 * Applies cloud masking, radiometric calibration, and calculates LST & NDWI.
 * @param {ee.Image} image - Input Landsat image
 * @return {ee.Image} - Processed image with LST_Celsius and NDWI bands
 */
function preprocess(image) {
  // QA_PIXEL cloud masking
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)   // Dilated Cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))       // Cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))       // Cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0));      // Cloud Shadow

  // Radiometric calibration
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B10').multiply(0.0034172).add(149.0);

  // Calculate indices (Float32 for compatibility)
  var lst = thermalBand.subtract(273.15).rename('LST_Celsius').float();
  var ndwi = opticalBands.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI').float();

  return image.addBands(lst)
              .addBands(ndwi)
              .updateMask(mask)
              .select(['LST_Celsius', 'NDWI']);
}

// =============================================================================
// 4. APPLY FILTERS AND PREPROCESSING
// =============================================================================

var filteredCol = landsat
  .filterBounds(geometry)
  .filterDate('2020-01-01', '2025-12-31')
  .filter(ee.Filter.lt('CLOUD_COVER', 30))
  .map(preprocess);

print('Total filtered images:', filteredCol.size());

// =============================================================================
// 5. CREATE MONTHLY MEDIAN COMPOSITES
// =============================================================================

var months = ee.List.sequence(1, 12);

var monthlyComposites = months.map(function(m) {
  var monthNum = ee.Number(m);
  
  // Aggregate all images from the same month across all years
  var composite = filteredCol
    .filter(ee.Filter.calendarRange(monthNum, monthNum, 'month'))
    .median()
    .clip(geometry)
    .set('month', monthNum);
    
  return composite;
});

// =============================================================================
// 6. VISUALIZATION (Sample: June)
// =============================================================================

var sampleMonth = ee.Image(monthlyComposites.get(5)); // June (0-indexed)

var lstViz = {
  bands: ['LST_Celsius'],
  min: 0,
  max: 40,
  palette: ['blue', 'cyan', 'green', 'yellow', 'red']
};

var ndwiViz = {
  bands: ['NDWI'],
  min: -0.5,
  max: 0.5,
  palette: ['brown', 'white', 'blue']
};

Map.addLayer(sampleMonth, lstViz, 'June Median LST');
Map.addLayer(sampleMonth, ndwiViz, 'June Median NDWI', false);

// =============================================================================
// 7. BATCH EXPORT TO GOOGLE DRIVE
// =============================================================================

for (var i = 0; i < 12; i++) {
  var monthImage = ee.Image(monthlyComposites.get(i));
  var monthName = (i + 1) < 10 ? '0' + (i + 1) : String(i + 1);
  var fileName = 'Tianjin_Monthly_Median_' + monthName;
  
  Export.image.toDrive({
    image: monthImage,
    description: fileName,
    folder: 'Tianjin_Monthly_Project',
    scale: 30,
    region: geometry,
    crs: 'EPSG:32650',  // UTM Zone 50N
    maxPixels: 1e9
  });
}

// =============================================================================
// OUTPUT SPECIFICATION
// =============================================================================
/*
 * Exported files (12 total):
 *   - Tianjin_Monthly_Median_01.tif ... Tianjin_Monthly_Median_12.tif
 * 
 * Band structure:
 *   - Band 1: LST_Celsius (Land Surface Temperature in °C)
 *   - Band 2: NDWI (Normalized Difference Water Index)
 * 
 * Spatial Reference: EPSG:32650 (WGS 84 / UTM zone 50N)
 * Resolution: 30m
 */
