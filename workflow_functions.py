"""
Cyprus Terrace Detection: Workflow Functions Module

This module provides the core workflow functions for processing satellite imagery
and DEM data for terrace detection across Cyprus. It includes functions for:

- Geospatial data clipping and resampling
- DEM processing and terrain analysis  
- Texture feature extraction using GLCM
- Edge detection processing
- Zonal statistics calculation
- Complete polygon processing pipeline

Key Functions:
- process_workflow: Main processing pipeline
- buffer_make: Create processing buffers around polygons
- clip_and_resample_*: Various clipping and resampling functions
- calculate_and_save_texture_features: GLCM texture analysis
- populate_segments: Zonal statistics computation

Author: Andrei Kedich
Date: 2025
"""

# Core geospatial and data processing libraries
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd

# Raster processing libraries
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import richdem as rd
import geowombat as gw
from pysheds.grid import Grid
from rasterstats import zonal_stats

# Utility libraries
from tqdm import tqdm
from shapely.geometry import Polygon, mapping

# Custom modules
import fast_glcm
from edges_cyprus import edge_detection


def create_processing_buffer(polygon_file_path, buffer_distance_meters=80):
    """
    Create a rectangular buffer around polygon geometries for raster processing.
    
    Args:
        polygon_file_path (str): Path to polygon file (GPKG, SHP, etc.)
        buffer_distance_meters (int): Buffer distance in meters (default: 80)
        
    Returns:
        tuple: (buffer_utm, buffer_wgs84) - Buffer geometries in UTM and WGS84 projections
               Returns (None, None) if error occurs
    """
    try:
        # Load polygon data
        polygon_data = gpd.read_file(polygon_file_path)

        # Ensure UTM projection (EPSG:32636 for Cyprus region)
        if polygon_data.crs != "EPSG:32636":
            polygon_utm = polygon_data.to_crs(epsg=32636)
        else:
            polygon_utm = polygon_data

        # Get bounding box coordinates
        min_x, min_y, max_x, max_y = polygon_utm.total_bounds

        # Create rectangular buffer around bounding box
        buffer_polygon = Polygon([
            (min_x - buffer_distance_meters, min_y - buffer_distance_meters),
            (max_x + buffer_distance_meters, min_y - buffer_distance_meters),
            (max_x + buffer_distance_meters, max_y + buffer_distance_meters),
            (min_x - buffer_distance_meters, max_y + buffer_distance_meters)
        ])

        # Create buffer GeoDataFrames
        buffer_utm = gpd.GeoDataFrame([1], geometry=[buffer_polygon], crs="EPSG:32636")
        buffer_wgs84 = buffer_utm.to_crs("EPSG:4326")

        print(f"Processing buffer created successfully ({buffer_distance_meters}m)")
        return buffer_utm, buffer_wgs84

    except Exception as e:
        print(f"Error creating processing buffer: {e}")
        return None, None

# Function to clip and save image raster
def clip_and_save_img(raster_path, buffer_gdf, clipped_raster_path):
    try:
        # Open the raster
        with rasterio.open(raster_path) as src:
            # Define buffer geometry
            buffer_geometry = [buffer_gdf.geometry[0].__geo_interface__]

            # Clip the raster using the mask function
            out_image, out_transform = mask(src, buffer_geometry, crop=True)
            out_meta = src.meta.copy()  # Copy the metadata

            # Update the metadata with new dimensions, transformation, and CRS
            out_meta.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform,
                'dtype': 'uint8',
                'nodata': 0,
                'compress': 'deflate',
            })

            # Save the clipped raster
            with rasterio.open(clipped_raster_path, 'w', **out_meta) as dst:
                dst.write(out_image)  # Convert to int8 before writing

        print(f"Raster clipped and saved: {clipped_raster_path}")
        return True

    except Exception as e:
        print(f"Error clipping and saving raster: {e}")
        return False
    
#new function when Sentinel imagery was introduced

def clip_and_resample_raster(raster_path, reference_image_path, buffer_gdf, output_raster_path):
    try:
        print(f"Processing raster: {raster_path}")

        # Open the input raster
        with rasterio.open(raster_path) as src:
            # Convert the buffer geometry to the same CRS as the raster
            buffer_gdf = buffer_gdf.to_crs(src.crs)
            # Use the first (and only) geometry for clipping
            buffer_geometry = [buffer_gdf.geometry.iloc[0].__geo_interface__]
            
            # Clip the raster
            out_image, out_transform = mask(src, buffer_geometry, crop=True)
            
            # Open the reference image to derive resampling parameters
            with rasterio.open(reference_image_path) as ref:
                # Calculate target transformation, width, and height
                transform, width, height = calculate_default_transform(
                    src.crs, ref.crs, ref.width, ref.height, *ref.bounds,
                    dst_width=ref.width, dst_height=ref.height
                )
                
                # Update metadata for the resampled raster
                resampled_meta = src.meta.copy()
                resampled_meta.update({
                    "crs": ref.crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "dtype": 'uint16',  # Output type set to uint16
                    "nodata": 0,
                    "compress": "DEFLATE",
                    "count": 1  # Ensure single band
                })
                
                # Resample and write the final output
                with rasterio.open(output_raster_path, "w", **resampled_meta) as dest:
                    reproject(
                        source=out_image[0].astype("uint16"),
                        destination=rasterio.band(dest, 1),
                        src_transform=out_transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=ref.crs,
                        resampling=Resampling.nearest,
                        dst_nodata=0
                    )

        print(f"Raster clipped and resampled successfully: {output_raster_path}")
        return output_raster_path

    except Exception as e:
        print(f"Error in clip_and_resample_raster: {e}")
        return None


def clip_and_resample_dem(dem_raster_path, reference_image_path, buffer_gdf, clipped_dem_path, resampled_dem_path):
    try:
        # Step 1: Clip the DEM using the buffer GeoDataFrame
        with rasterio.open(dem_raster_path) as src:
            # Perform the clipping
            clipped, transform = mask(src, buffer_gdf.geometry, crop=True, nodata=-9999)
            
            # Update metadata for the clipped DEM
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "nodata": -9999
            })

        # Save the clipped DEM
        with rasterio.open(clipped_dem_path, 'w', **clipped_meta) as dest:
            dest.write(clipped)
        print(f"Clipped DEM saved to: {clipped_dem_path}")

        # Step 2: Resample the clipped DEM to match the reference image
        with rasterio.open(reference_image_path) as ref_img:
            ref_transform = ref_img.transform
            ref_crs = ref_img.crs
            ref_height = ref_img.height
            ref_width = ref_img.width
            ref_resolution = ref_img.res[0]  # Assumes square pixels

            # Open the clipped DEM for resampling
            with rasterio.open(clipped_dem_path) as clipped_src:
                # Reproject and resample
                resampled_meta = clipped_src.meta.copy()
                resampled_meta.update({
                    "crs": ref_crs,
                    "transform": ref_transform,
                    "height": ref_height,
                    "width": ref_width,
                    "nodata": -9999
                })

                # Create resampled DEM
                with rasterio.open(resampled_dem_path, 'w', **resampled_meta) as dest:
                    for band in range(1, clipped_src.count + 1):
                        reproject(
                            source=rasterio.band(clipped_src, band),
                            destination=rasterio.band(dest, band),
                            src_transform=clipped_src.transform,
                            src_crs=clipped_src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=Resampling.bilinear
                        )

        print(f"DEM clipped and resampled, saved to: {resampled_dem_path}")
        return resampled_dem_path

    except Exception as e:
        print(f"Error clipping and resampling DEM: {e}")
        return None


# Function to update and compress raster
def update_and_compress_raster(input_path, output_name):
    try:
        with rasterio.open(input_path) as src:
            data = src.read(1)
            meta = src.meta
        meta.update({
            'compress': 'deflate',
            'dtype': rasterio.float32
        })

        clipped_raster_folder = os.path.dirname(input_path)
        output_path = os.path.join(clipped_raster_folder, f'{output_name}.tif')
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data, 1)
        os.remove(input_path)  # Erase the old file
        print(f"Raster updated and compressed: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error updating and compressing raster: {e}")
        return None


def clip_and_resample_landcover(landcover_raster_path, reference_image_path, buffer_gdf, resampled_cover_path):
    try:
        print(f"Clipping and resampling landcover raster: {landcover_raster_path}")

        # Open the landcover raster
        with rasterio.open(landcover_raster_path) as src:
            # Convert the geometries to the same CRS as the raster
            buffer_gdf = buffer_gdf.to_crs(src.crs)
            
            # Mask the raster using the geometries (clip)
            out_image, out_transform = mask(src, buffer_gdf.geometry, crop=True, nodata=255)
            
            # Open the reference image to get the target transform, width, and height
            with rasterio.open(reference_image_path) as ref:
                # Calculate the target transform, width, and height for resampling
                transform, width, height = calculate_default_transform(
                    src.crs, ref.crs, ref.width, ref.height, *ref.bounds, dst_width=ref.width, dst_height=ref.height
                )

                # Update metadata for the resampled raster
                resampled_meta = src.meta.copy()
                resampled_meta.update({
                    "crs": ref.crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "compress": "DEFLATE",
                    "dtype": 'uint8'
                })

                # Resample and write the final output directly
                with rasterio.open(resampled_cover_path, "w", **resampled_meta) as dest:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=out_image[i-1],
                            destination=rasterio.band(dest, i),
                            src_transform=out_transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=ref.crs,
                            resampling=Resampling.nearest,
                            dst_nodata=255
                        )

        print(f"Landcover raster clipped and resampled, saved: {resampled_cover_path}")
        return resampled_cover_path

    except Exception as e:
        print(f"Error clipping and resampling landcover raster: {e}")
        return None

    
def process_dem_flowdir(clipped_dem_path, clipped_raster_folder, grid_num, resampled_dem_path):
    try:
        # Load DEM using pysheds Grid
        grid = Grid.from_raster(clipped_dem_path, data_name='dem')
        dem = grid.read_raster(clipped_dem_path, data_name='dem')

        # Fill depressions and resolve flats
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)

        # Save inflated DEM
        infldem_path = os.path.join(clipped_raster_folder, 'inflated_dem_' + grid_num + ".tif")
        grid.to_raster(inflated_dem, infldem_path, dtype='int16', nodata=-1, overwrite=True, compress='DEFLATE')

        # Compute flow direction
        dirmap = (90, 45, 360, 315, 270, 225, 180, 135)
        fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

        # Save flow direction raster
        flowdir_path = os.path.join(clipped_raster_folder, 'flowdir_' + grid_num + ".tif")
        grid.to_raster(fdir, flowdir_path, dtype='int16', nodata=-1, overwrite=True, compress='DEFLATE')

        # Resample flow direction raster using rasterio
        flowdir_path_res = os.path.join(clipped_raster_folder, 'flowdir_res_' + grid_num + ".tif")
        
        with rasterio.open(flowdir_path) as src:
            with rasterio.open(resampled_dem_path) as ref:
                transform = ref.transform
                width = ref.width
                height = ref.height

            data = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
            profile = src.profile
            profile.update(
                transform=transform,
                width=width,
                height=height,
                compress='DEFLATE',
                nodata=-1
            )

            with rasterio.open(flowdir_path_res, 'w', **profile) as dst:
                dst.write(data, 1)

        print(f"Flow direction raster processed and resampled, saved: {flowdir_path_res}")
        return flowdir_path_res

    except Exception as e:
        print(f"Error processing DEM for flow direction: {e}")
        return None

def run_edge_detection(image_path, canny_params, flowdir_image):
    print("Running edge detection...")

    path_list = []
    for i, (canny_low, canny_up) in enumerate(tqdm(canny_params, desc='Edge detection progress')):
        try:
            canny_path, filtered_path = edge_detection(image_path, canny_low, canny_up, flowdir_image, num_directions=3, connectivity=8)
            path_list.extend([canny_path, filtered_path])
        except Exception as e:
            print(f"Error in edge detection iteration {i+1}: {e}")
            continue
    
    print("Edge detection completed.")
    return path_list

def calculate_and_save_texture_features(red_image_path, clipped_raster_folder, grid_num):
        try:
            with rasterio.open(red_image_path) as src:
                image = src.read(1)  # Read the red band (assuming it's the first band)
                meta_img = src.meta

            image_sq = np.squeeze(image)
            angles = [0, 90, 180, 270]

            features_to_calc = ['mean', 'std', 'cont', 'homo', 'energy', 'cluster_shade', 'entropy']
            all_features_selection = []
            for feature in features_to_calc:
                result = fast_glcm.har_stats(image=image_sq, angles=angles, gray_levels=16, kernel_size=31, features_to_calculate=feature)
                all_features_selection.append(result)

            meta_img.update({
                'compress': 'deflate',
                'dtype': rasterio.float32
            })

            saved_paths = []

            for i, layer in enumerate(tqdm(all_features_selection, desc='Calculating and saving texture features')):
                output_path = features_to_calc[i] + "_text_" + grid_num + ".tif"
                path = os.path.join(clipped_raster_folder, output_path)
                saved_paths.append(path)
                with rasterio.open(path, 'w', **meta_img) as dst:
                    dst.write(layer, 1)
                
            print("Texture features calculated and saved.")
            return saved_paths

        except Exception as e:
            print(f"Error in calculating or saving texture features: {e}")
            return []

def stat_polygons(polygon_file, raster_file, stats_to_calc, band_num, band_name):
    try:
        # Read the raster file to get the projection
        with rasterio.open(raster_file) as src:
            raster_crs = src.crs
        
        # Reproject the polygons to the raster's CRS
        polygon_file = polygon_file.to_crs(raster_crs)
        
        # Calculate zonal statistics
        calc_stat = zonal_stats(polygon_file, raster_file, stats=stats_to_calc)
        
        # Add the statistics as new columns to the polygons GeoDataFrame
        for stat in stats_to_calc:
            polygon_file[f'{stat}_{band_name}'] = [s[stat] for s in calc_stat]
        
        return polygon_file
    
    except Exception as e:
        print(f"Error processing band '{band_name}': {e}")
        return polygon_file  # Return the original file without modifications in case of an error

def populate_segments(polygon_layer, raster_paths_list):
    try:
        # Make a folder
        folder_name = os.path.splitext(os.path.basename(polygon_layer))[0]
        folder_name_parts = folder_name.split("_")
        folder_name = "_".join(folder_name_parts[:2])  # Select first and second elements
        grid_num = folder_name_parts[1]
        
        # Create grid_data folder within working_directory if it doesn't exist
        working_directory = "E:/Cyprus_paper_data"
        grid_data_folder = os.path.join(working_directory, 'grid_all_polysSen')
        os.makedirs(grid_data_folder, exist_ok=True)
        print(f"Grid data folder created/existed at: {grid_data_folder}")
        
        band_names = [
            'red', 'green', 'blue', 'grayscale', 'elevation', 'slope', 'profcurv', 'plancurv', 'landcover',
            'mean_text', 'std_text', 'cont_text', 'homo_text', 'energy_text', 'clsh_text', 'entropy_text',
            'canny_200_400', "edges_200_400", 'canny_150_300', "edges_150_300", 'canny_100_250', "edges_100_250"
        ]
        
        # Sets of statistics to calculate
        mean = ['mean']    
        mean_std = ['mean', 'std']
        mean_std_range = ['mean', 'std', 'range']
        mean_std_min_max = ['mean', 'std', 'min', 'max']
        mean_range = ['mean', 'range']
        majority = ['majority']
        rng = ['range']
        range_std = ['range', 'std']
        percentile = ['percentile_10', 'percentile_90', 'mean', 'std', 'range']
        
        what_calculate = [mean, mean, mean, mean, range_std, percentile, range_std, range_std, majority,
                          mean_std, mean_std, mean_std, mean_std, mean_std, mean_std, mean_std,
                          mean, mean, mean, mean, mean, mean]
        
        # num_cores = multiprocessing.cpu_count() // 4
        
        polygons = gpd.read_file(polygon_layer)

        def process_band(polygons, band_loc, band_num):
            try:
                polygons_copy = polygons.copy()
                return stat_polygons(polygons_copy, band_loc, what_calculate[band_num - 1], band_num, band_names[band_num - 1])
            except Exception as e:
                print(f"Error processing band number {band_num}: {e}")
                return polygons.copy()  # Return the original polygons in case of an error

        start_time = time.time()

        # Iterate over each band
        results = []
        for band_num, band_loc in enumerate(raster_paths_list, start=1):
            result = process_band(polygons, band_loc, band_num)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time / 60)
        seconds = int(total_time % 60)
        seconds = "{:02d}".format(seconds)
        print(f"Total time taken: {minutes} minutes {seconds} seconds")
        
        geometry_column = results[0]['geometry']
        other_columns = pd.concat([gdf.drop(columns=['geometry']) for gdf in results], axis=1)
        merged_gdf = pd.concat([geometry_column, other_columns], axis=1)
        final_gdf = merged_gdf.loc[:, ~merged_gdf.columns.duplicated()]
        final_gdf = gpd.GeoDataFrame(final_gdf)
        final_gdf = final_gdf.dropna(subset=['majority_landcover'])  # remove rows with NaN values
        final_gdf = final_gdf[final_gdf['majority_landcover'] != -15]  # remove nodata values
        
        output_path = os.path.join(grid_data_folder, f'polygons_{grid_num}_stats.gpkg')
        final_gdf.to_file(output_path, driver='GPKG')
        print(f"New polygons GeoDataFrame with the added statistics saved successfully at: {output_path}")
    
    except Exception as e:
        print(f"Error in populate_segments: {e}")

def process_workflow(polygon_path, google_img, sentinel, landcover_path, dem_path):
    # try:
    # Step 2: Create a buffer around the polygon

    buffer_gdf, buffer_gdf_wgs84 = create_processing_buffer(polygon_path)

    # Step 3: Create folder and path names
    folder_name = os.path.splitext(os.path.basename(polygon_path))[0]  # here we select the first file
    folder_name_parts = folder_name.split("_")
    folder_name = "_".join(folder_name_parts[:2])  # Select first and second elements
    grid_num = folder_name_parts[1]

    # Create grid_data folder within working_directory if it doesn't exist
    working_directory = "E:/Cyprus_paper_data"
    grid_data_folder = os.path.join(working_directory, 'grid_data_allSen')
    os.makedirs(grid_data_folder, exist_ok=True)

    # Create clipped_raster_folder within grid_data_folder
    clipped_raster_folder = os.path.join(grid_data_folder, folder_name)
    os.makedirs(clipped_raster_folder, exist_ok=True)

    # Step 4: Clip the Google image to the buffer (only red and gray bands)
    clipped_google_paths = {}
    for color, tif_path1 in google_img.items():
        raster_name = os.path.splitext(os.path.basename(tif_path1))[0]
        output_name = f"{raster_name}_{grid_num}.tif"
        clipped_google_paths[color] = os.path.join(clipped_raster_folder, output_name)
        print(clipped_google_paths[color])

    for color, tif_path1 in google_img.items():
        success = clip_and_save_img(tif_path1, buffer_gdf, clipped_google_paths[color])
        print(success)
        if not success:
            raise Exception(f"Error processing Google {color} band.")

    print(clipped_google_paths["gray"], clipped_google_paths["red"])
    reference_image_path = clipped_google_paths['gray']  # Use gray as the reference for resampling

    # Step 5: Clip and resample Sentinel imagery (all bands)
    clipped_sentinel_paths = {}
    for color, tif_path2 in sentinel.items():
        raster_name = os.path.splitext(os.path.basename(tif_path2))[0]
        output_name = f"{raster_name}_{grid_num}.tif"
        clipped_sentinel_paths[color] = os.path.join(clipped_raster_folder, output_name)

    for color, tif_path2 in sentinel.items():
        success = clip_and_resample_raster(tif_path2, reference_image_path, buffer_gdf, clipped_sentinel_paths[color])
        if not success:
            raise Exception(f"Error processing Sentinel {color} band.")

    print(clipped_sentinel_paths["gray"], clipped_sentinel_paths["red"],
        clipped_sentinel_paths["green"], clipped_sentinel_paths["blue"])

    # Step 5: Clip and resample DEM
    dem_name = "dem_clipped_" + grid_num
    dem_name_res = "dem_resampled_" + grid_num

    clipped_dem_path = os.path.join(clipped_raster_folder, dem_name + '.tif')
    resampled_dem_path = os.path.join(clipped_raster_folder, dem_name_res + '.tif')

    resampled_dem_path = clip_and_resample_dem(dem_path, reference_image_path, buffer_gdf, clipped_dem_path,
                                                resampled_dem_path)
    if not resampled_dem_path:
        raise Exception("Error processing DEM.")

    print(resampled_dem_path)

    # Step 6: Slope, profile curvature, and planform curvature
    # Define paths to the output raster files
    raster_attributes = ['slope_degrees', 'profile_curvature', 'planform_curvature']
    attribute_files = {attr: os.path.join(clipped_raster_folder, f'{attr}_{grid_num}.tif') for attr in
                        raster_attributes}

    # Load the DEM data
    dem_data = rd.LoadGDAL(resampled_dem_path)

    # Generate terrain attributes and save them
    for attr, path in attribute_files.items():
        terrain_attr = rd.TerrainAttribute(dem_data, attrib=attr)
        rd.SaveGDAL(path, terrain_attr)

    # Update and compress each attribute raster
    updated_files = {}
    for attr, path in attribute_files.items():
        updated_files[attr] = update_and_compress_raster(path, attr + '_upd')

    slope_dem_path_new = updated_files['slope_degrees']
    prof_curv_path_new = updated_files['profile_curvature']
    plan_curv_path_new = updated_files['planform_curvature']

    print(slope_dem_path_new, prof_curv_path_new, plan_curv_path_new)

    # Step 7: Flow direction raster
    flowdir_path_res = process_dem_flowdir(clipped_dem_path, clipped_raster_folder, grid_num, resampled_dem_path)
    if not flowdir_path_res:
        raise Exception("Error processing flow direction.")

    # Step 8: Landcover
    clipped_cover_path = os.path.join(clipped_raster_folder, f"landcover_clipped_{grid_num}.tif")
    resampled_cover_path = os.path.join(clipped_raster_folder, f"landcover_resampled_{grid_num}.tif")

    landcover_path = landcover_path  # original landcover path (large file)

    resampled_cover_path = clip_and_resample_landcover(landcover_path, reference_image_path, buffer_gdf_wgs84, resampled_cover_path)

    print(resampled_cover_path)

    # Step 9: Texture features
    red_image_path = clipped_google_paths['red']
    text_paths = calculate_and_save_texture_features(red_image_path, clipped_raster_folder, grid_num)

    for path in text_paths:
        print(path)

    # Parameters for Canny edge detection
    canny_params = [
        (200, 400),
        (150, 300),
        (100, 250)
    ]

    # Run edge detection function
    paths_edges = run_edge_detection(reference_image_path, canny_params, flowdir_path_res)

    all_paths = [
        clipped_sentinel_paths.get("red", ""),
        clipped_sentinel_paths.get("green", ""),
        clipped_sentinel_paths.get("blue", ""),
        clipped_sentinel_paths.get("gray", ""),
        resampled_dem_path or "",
        slope_dem_path_new or "",
        prof_curv_path_new or "",
        plan_curv_path_new or "",
        resampled_cover_path or ""
    ]

    # Remove empty strings from the list if any path was None
    all_paths = [path for path in all_paths if path]

    # Extend with lists
    if text_paths:
        all_paths.extend(text_paths)
    if paths_edges:
        all_paths.extend(paths_edges)

    # Step 10: Populate segments
    populate_segments(polygon_path, all_paths)
            
    print("=" * 60)
    print("POLYGON PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Processed polygon: {os.path.basename(polygon_path)}")
    print(f"Output directory: {grid_data_folder}")
    print("Generated features:")
    print("  - Spectral bands (Google & Sentinel)")
    print("  - Terrain analysis (DEM, slope, curvatures)")
    print("  - Texture features (GLCM)")
    print("  - Edge detection results")
    print("  - Land cover classification")
    print("=" * 60)