/**
 * =============================================================================
 * PROJECT: The Blue Spine - Tianjin Haihe Cooling Analysis
 * SCRIPT: 00 GEE Data Acquisition (v2.0 - Multi-Variable)
 * PLATFORM: Google Earth Engine (https://code.earthengine.google.com/)
 * DESCRIPTION: 
 *   Exports 12 monthly median composite images using Landsat 8/9 data (2020-2025).
 *   Each image contains: LST, NDVI, NDBI, NDWI bands for multi-variable GWR.
 * AUTHOR: Congyuan Zheng
 * DATE: 2026-02
 * VERSION: 2.0 - Added NDVI and NDBI for multivariate analysis
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
// 3. PRE-PROCESSING FUNCTION (v2.0 - Multi-Variable)
// =============================================================================

/**
 * Applies cloud masking, radiometric calibration, and calculates all indices.
 * @param {ee.Image} image - Input Landsat image
 * @return {ee.Image} - Processed image with LST, NDVI, NDBI, NDWI bands
 * 
 * Landsat 8/9 Band Reference:
 *   SR_B2 = Blue, SR_B3 = Green, SR_B4 = Red
 *   SR_B5 = NIR, SR_B6 = SWIR1, SR_B7 = SWIR2
 *   ST_B10 = Thermal
 */
function preprocess(image) {
  // QA_PIXEL cloud masking
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)   // Dilated Cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))       // Cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))       // Cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0));      // Cloud Shadow

  // Radiometric calibration for optical bands
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  
  // Thermal band calibration
  var thermalBand = image.select('ST_B10').multiply(0.0034172).add(149.0);

  // =========================================================================
  // Calculate Indices (All as Float32 for compatibility)
  // =========================================================================
  
  // 1. LST (Land Surface Temperature in Celsius)
  var lst = thermalBand.subtract(273.15).rename('LST_Celsius').float();
  
  // 2. NDVI (Normalized Difference Vegetation Index)
  //    NDVI = (NIR - Red) / (NIR + Red) = (B5 - B4) / (B5 + B4)
  var ndvi = opticalBands.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI').float();
  
  // 3. NDBI (Normalized Difference Built-up Index)
  //    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) = (B6 - B5) / (B6 + B5)
  var ndbi = opticalBands.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI').float();
  
  // 4. NDWI (Normalized Difference Water Index)
  //    NDWI = (Green - NIR) / (Green + NIR) = (B3 - B5) / (B3 + B5)
  var ndwi = opticalBands.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI').float();

  return image.addBands(lst)
              .addBands(ndvi)
              .addBands(ndbi)
              .addBands(ndwi)
              .updateMask(mask)
              .select(['LST_Celsius', 'NDVI', 'NDBI', 'NDWI']);
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
// 6. VISUALIZATION (Sample: July)
// =============================================================================

var sampleMonth = ee.Image(monthlyComposites.get(6)); // July (0-indexed)

var lstViz = {
  bands: ['LST_Celsius'],
  min: 20, max: 45,
  palette: ['blue', 'cyan', 'green', 'yellow', 'red']
};

var ndviViz = {
  bands: ['NDVI'],
  min: -0.2, max: 0.8,
  palette: ['brown', 'yellow', 'green', 'darkgreen']
};

var ndbiViz = {
  bands: ['NDBI'],
  min: -0.3, max: 0.3,
  palette: ['green', 'white', 'red']
};

var ndwiViz = {
  bands: ['NDWI'],
  min: -0.5, max: 0.5,
  palette: ['brown', 'white', 'blue']
};

Map.addLayer(sampleMonth, lstViz, 'July LST', true);
Map.addLayer(sampleMonth, ndviViz, 'July NDVI', false);
Map.addLayer(sampleMonth, ndbiViz, 'July NDBI', false);
Map.addLayer(sampleMonth, ndwiViz, 'July NDWI', false);

// =============================================================================
// 7. BATCH EXPORT TO GOOGLE DRIVE (v2.0 - 4 bands)
// =============================================================================

// [WARNING] IMPORTANT: New files will have 4 bands instead of 2!
// Band order: LST_Celsius, NDVI, NDBI, NDWI

for (var i = 0; i < 12; i++) {
  var monthImage = ee.Image(monthlyComposites.get(i));
  var monthName = (i + 1) < 10 ? '0' + (i + 1) : String(i + 1);
  
  // Use v2 suffix to distinguish from old files
  var fileName = 'Tianjin_Monthly_v2_' + monthName;
  
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
// OUTPUT SPECIFICATION (v2.0)
// =============================================================================
/*
 * Exported files (12 total):
 *   - Tianjin_Monthly_v2_01.tif ... Tianjin_Monthly_v2_12.tif
 * 
 * Band structure (4 bands):
 *   - Band 1: LST_Celsius (Land Surface Temperature in °C)
 *   - Band 2: NDVI (Normalized Difference Vegetation Index)
 *   - Band 3: NDBI (Normalized Difference Built-up Index)
 *   - Band 4: NDWI (Normalized Difference Water Index)
 * 
 * Spatial Reference: EPSG:32650 (WGS 84 / UTM zone 50N)
 * Resolution: 30m
 * 
 * Variable meanings for GWR:
 *   - NDVI: Vegetation density (higher = more green space, cooler)
 *   - NDBI: Built-up density (higher = more impervious surface, hotter)
 *   - NDWI: Water presence (higher = water body)
 */
