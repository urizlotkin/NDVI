import ee
import geemap
import pandas as pd
from tqdm import tqdm



def get_coordinates():
    df = pd.read_csv('data/data points.csv')
    df = df[['OID_', 'point_long', 'point_lat', 'gridcode', 'Prime_unit']]
    coordinates = list(zip(df['point_long'], df['point_lat']))
    return coordinates


def applyScaleFactors(image):
    """
    Applies the Landsat Collection 2 Level-2 scale and offset
    to surface reflectance bands. Assumes the same scale/offset for
    all SR_Bx bands: scale=2.75e-05, offset=-0.2
    """
    srBands = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5', 'SR_B7']
    
    # Multiply by 2.75e-05 (scale), then add -0.2 (offset).
    scaled = image.select(srBands).multiply(2.75e-05).add(-0.2)
    
    # Overwrite original SR_Bx bands with scaled versions.
    # Any other bands (e.g., QA_PIXEL) remain unaffected.
    return image.addBands(scaled, overwrite=True)


def maskLandsatClouds(image):
    # 'QA_PIXEL' band contains cloud and shadow flags
    qa = image.select('QA_PIXEL')
    
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloud_shadow_bit_mask = 1 << 3  # 2^3
    clouds_bit_mask       = 1 << 5  # 2^5

    # Both flags set to 0 means the pixel is neither shadowed nor cloudy.
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
              .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
    
    # Return the masked image, and keep all other bands
    return image.updateMask(mask)


def addNVDI(image):
    """
    Adds an NDVI band to the input image, named 'NDVI'.
    
    - For Landsat 4, 5, 7, uses bands B4 (NIR) and B3 (Red).
    - For Landsat 8, 9,   uses bands B5 (NIR) and B4 (Red).
    """
    # Convert the image's 'Satellite' property to an ee.String
    sat = ee.String(image.get('Satellite'))

    def ndvi_4_5_7(img):
        # NDVI = (B4 - B3) / (B4 + B3)
        return img.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')

    def ndvi_8_9(img):
        # NDVI = (B5 - B4) / (B5 + B4)
        return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # This condition will be True if sat == 'Landsat 4' OR 'Landsat 5' OR 'Landsat 7'
    condition_4_5_7 = (
        sat.compareTo('Landsat 4').eq(0)
        .Or(sat.compareTo('Landsat 5').eq(0))
        .Or(sat.compareTo('Landsat 7').eq(0))
    )

    # Use ee.Algorithms.If(...) to pick the formula
    ndvi_band = ee.Algorithms.If(
        condition_4_5_7,      # condition
        ndvi_4_5_7(image),    # true case
        ndvi_8_9(image)       # false case
    )

    # ee.Algorithms.If returns a generic ComputedObject; cast it to ee.Image
    ndvi_image = ee.Image(ndvi_band)

    # Return the original image with the NDVI band added
    return image.addBands(ndvi_image)


def image_to_feature(img, roi, point_x, point_y):
    """ Converts an image to a Feature with properties:
        Date, POINT_X, POINT_Y, Satellite, NDVI
    """
    # Compute mean NDVI over the ROI
    stats = img.select('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30,
        bestEffort=True
    )    
    # If there's no NDVI data, skip creating a feature
    # Note: We can't do a direct if-check in server-side code,
    # but we can store the result anyway. We'll filter later if needed.
    
    # Create properties server-side
    date_str = img.date().format('YYYY-MM-dd')
    satellite = img.get('Satellite')
    ndvi_value = stats.get('NDVI')  # This is an ee.Number or null
    
    # Return a Feature with no geometry (or .centroid if you want)
    return ee.Feature(None, {
        'Date': date_str,
        'POINT_X': point_x,
        'POINT_Y': point_y,
        'Satellite': satellite,
        'NDVI': ndvi_value
    })


def collect_landsat_data(time_start, time_end, coordinates, path_to_save):
    data = []  # This will store the final results

    for coord in tqdm(coordinates):
        try:
            roi = ee.Geometry.Point(coord).buffer(200).bounds()

            # Build merged landsat_collection ...
            landsat_4 = (ee.ImageCollection("LANDSAT/LT04/C02/T1_L2")
                        .filterBounds(roi)
                        .filterDate(time_start, time_end)
                        .filterMetadata('CLOUD_COVER_LAND', 'less_than', 10)
                        .map(lambda image: image.set('Satellite', 'Landsat 4')))

            landsat_5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                        .filterBounds(roi)
                        .filterDate(time_start, time_end)
                        .filterMetadata('CLOUD_COVER_LAND', 'less_than', 10)
                        .map(lambda image: image.set('Satellite', 'Landsat 5')))

            landsat_7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                        .filterBounds(roi)
                        .filterDate(time_start, time_end)
                        .filterMetadata('CLOUD_COVER_LAND', 'less_than', 10)
                        .map(lambda image: image.set('Satellite', 'Landsat 7')))

            landsat_8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                        .filterBounds(roi)
                        .filterDate(time_start, time_end)
                        .filterMetadata('CLOUD_COVER_LAND', 'less_than', 10)
                        .map(lambda image: image.set('Satellite', 'Landsat 8')))

            landsat_9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                        .filterBounds(roi)
                        .filterDate(time_start, time_end)
                        .filterMetadata('CLOUD_COVER_LAND', 'less_than', 10)
                        .map(lambda image: image.set('Satellite', 'Landsat 9')))

            landsat_collection = (landsat_4
                                .merge(landsat_5)
                                .merge(landsat_7)
                                .merge(landsat_8)
                                .merge(landsat_9)
                                # .map(maskLandsatClouds)       # <--- NEW: Mask clouds
                                .map(applyScaleFactors)       # <--- Apply scale & offset
                                .map(addNVDI))               # <--- Compute NDVI

            # Map a function that creates a Feature (Date, Satellite, NDVI, etc.) for each image
            feature_collection = landsat_collection.map(
                lambda img: image_to_feature(img, roi, coord[0], coord[1])
            )
            size = feature_collection.size().getInfo()
            if size == 0:
                # Skip this coordinate because there are no images
                continue
            # Convert that FeatureCollection to a list of Features
            features = feature_collection.toList(feature_collection.size())
            # Bring it to Python in one go (still might be large, but far fewer calls)
            features_info = features.getInfo()
            data_point_added = 0
            # Loop through the features in Python
            for f in tqdm(features_info):
                prop = f['properties']
                ndvi_value = prop.get('NDVI')
                if ndvi_value is None:
                    continue  # skip no-data
                data_point_added += 1
                data.append({
                    'Date': prop.get('Date'),
                    'POINT_X': prop.get('POINT_X'),
                    'POINT_Y': prop.get('POINT_Y'),
                    'Satellite': prop.get('Satellite'),
                    'NDVI': ndvi_value
                })
            
            print(f"Added {data_point_added} data points for coordinate {coord}")
            print(f"Total data points: {len(data)}")
        except Exception as e:
            print(f"Error for coordinate {coord}: {e}")
            continue
    # Finally save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(path_to_save, index=False)
    print(f"Saved to {path_to_save}")
    
    
def collect_sentinel_data(time_start, time_end, coordinates, path_to_save):
    data = []  # This will store the final results

    # Define a helper to add NDVI to Sentinel-2 images
    def addNDVI_sentinel(image):
        # NDVI = (B8 - B4) / (B8 + B4)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # Set a 'Satellite' property (like 'Sentinel-2')
        image = image.set('Satellite', 'Sentinel-2')
        return image.addBands(ndvi)

    for coord in tqdm(coordinates):
        try:
            # Create a 500 m buffer around the point
            roi = ee.Geometry.Point(coord).buffer(200).bounds()

            # Build the Sentinel-2 ImageCollection
            sentinel_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate(time_start, time_end)
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10)
                .map(addNDVI_sentinel)
            )

            # Map a function that creates a Feature (Date, Satellite, NDVI, etc.) for each image
            feature_collection = sentinel_collection.map(
                lambda img: image_to_feature(img, roi, coord[0], coord[1])
            )

            # Check collection size before calling .toList(...)
            size = feature_collection.size().getInfo()
            if size == 0:
                # No Sentinel-2 images found for this coordinate/time range
                continue

            # Convert the FeatureCollection to a list in one request
            features = feature_collection.toList(size)
            features_info = features.getInfo()

            data_point_added = 0
            # Loop through the features in Python
            for f in features_info:
                prop = f['properties']
                ndvi_value = prop.get('NDVI')
                if ndvi_value is None:
                    # skip no-data
                    continue

                data_point_added += 1
                data.append({
                    'Date': prop.get('Date'),
                    'POINT_X': prop.get('POINT_X'),
                    'POINT_Y': prop.get('POINT_Y'),
                    'Satellite': prop.get('Satellite'),
                    'NDVI': ndvi_value
                })

            print(f"Added {data_point_added} data points for coordinate {coord}")
            print(f"Total data points: {len(data)}")
        except Exception as e:
            print(f"Error for coordinate {coord}: {e}")
            continue
    # Finally, save to CSV
    df = pd.DataFrame(data)
    df.to_csv(path_to_save, index=False)
    print(f"Saved to {path_to_save}")
    
    
def addNDVI_modis(image):
    """
    MODIS/006/MOD13Q1 has an 'NDVI' band that is typically scaled by 0.0001.
    We select it, rescale, rename, and add a 'Satellite' property = 'MODIS'.
    """
    # Select the raw NDVI band
    ndvi_raw = image.select('NDVI')
    
    # Rescale NDVI if needed (check the product doc to confirm scale factor = 0.0001)
    # ndvi_scaled = ndvi_raw.multiply(0.0001).rename('NDVI')

    # Set the Satellite property
    image = image.set('Satellite', 'MODIS')

    # Return image with the scaled NDVI band added
    return image.addBands(ndvi_raw)


def addModisQuality(image, roi):
    """
    For MODIS/006/MOD13Q1, compute the fraction of pixels in the ROI
    where SummaryQA == 0 (often means good quality / not cloudy).
    Adds a property 'good_fraction'.
    """
    # Create a binary mask: 1 where SummaryQA == 0, else 0
    good_mask = image.select('SummaryQA').eq(0)
    
    # Compute mean of that mask in the ROI -> fraction of good pixels
    stats = good_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=250,      # MOD13Q1 is 250m resolution
        bestEffort=True
    )
    good_fraction = ee.Number(stats.get('SummaryQA'))
    
    # Attach 'good_fraction' as a property
    return image.set('good_fraction', good_fraction)


def collect_modis_data(time_start, time_end, coordinates, path_to_save):
    """
    Collects NDVI data from MODIS MOD13Q1 (Terra) for the given time range and coordinates.
    Saves the results to 'modis_data.csv' with columns:
      Date, POINT_X, POINT_Y, Satellite, NDVI
    """
    data = []  # Accumulate all results

    for coord in tqdm(coordinates):
        try:
        # Define a 500 m buffer around the point
            roi = ee.Geometry.Point(coord).buffer(200).bounds()

            # Build the MODIS collection (MOD13Q1 = Terra 16-day 250m NDVI)
            modis_collection = (
                ee.ImageCollection("MODIS/061/MOD13A1")
                .filterDate(time_start, time_end)
                .filterBounds(roi)
                # Compute fraction of good (non-cloudy) pixels
                .map(lambda img: addModisQuality(img, roi))
                # Keep only images where >= 90% of pixels are good
                # .filter(ee.Filter.lte('CLOUD_COVER', 10))
                .map(addNDVI_modis)  # adds 'NDVI' band and sets 'Satellite' to 'MODIS'
            )
            # modis_collection = modis_collection.filter(ee.Filter.lte('CLOUD_COVER', 10))
            # Convert each image into a Feature with date, NDVI, etc.
            feature_collection = modis_collection.map(
                lambda img: image_to_feature(img, roi, coord[0], coord[1])
            )

            # Check if the collection has any images
            size = feature_collection.size().getInfo()
            if size == 0:
                # No MODIS images found for this coordinate/time range
                continue

            # Convert the FeatureCollection to a list in one request
            features = feature_collection.toList(size)
            features_info = features.getInfo()

            # Loop through the features in Python
            data_point_added = 0
            for f in features_info:
                prop = f['properties']
                ndvi_value = prop.get('NDVI')
                if ndvi_value is None:
                    # skip no-data
                    continue

                data_point_added += 1
                data.append({
                    'Date': prop.get('Date'),
                    'POINT_X': prop.get('POINT_X'),
                    'POINT_Y': prop.get('POINT_Y'),
                    'Satellite': prop.get('Satellite'),
                    'NDVI': ndvi_value
                })

            print(f"Added {data_point_added} data points for coordinate {coord}")
            print(f"Total data points: {len(data)}")
        except Exception as e:
            print(f"Error for coordinate {coord}: {e}")
            continue
    # Finally, save to a CSV
    df = pd.DataFrame(data)
    df.to_csv(path_to_save, index=False)
    print(f"Saved to {path_to_save}")
    
    


if __name__ == '__main__':
    coordinates = get_coordinates()
    # Define the time range
    time_start = '1982-01-01'
    # time_end = '2000-01-01'
    # time_start = '2022-12-03'
    time_end = '2025-01-01'
    # Initialize the Earth Engine API  
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='ee-uri')  
    # coordinates = coordinates[7533:]
    collect_landsat_data(time_start, time_end, coordinates, path_to_save='data/rest_landsat_data.csv')
    # collect_sentinel_data(time_start, time_end, coordinates, path_to_save='data/sentinel_data.csv')
    # collect_modis_data(time_start, time_end, coordinates, path_to_save='data/modis_data_2022-2025.csv')



