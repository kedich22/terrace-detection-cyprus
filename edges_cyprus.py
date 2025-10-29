"""
Cyprus Terrace Detection: Edge Detection Module

This module provides functions for detecting terrace edges in satellite imagery using
Canny edge detection, flow direction analysis, and contour filtering techniques.

Key Functions:
- canny_edge: Basic Canny edge detection on grayscale imagery
- detect_terraces: Directional terrace detection using flow direction
- back_detection: Enhanced edge detection using flood fill techniques  
- filter_contours: Contour filtering based on sinuosity and length criteria
- edge_detection: Complete edge detection pipeline

Author: Andrei Kedich
Date: 2025
"""

import os
import statistics
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from skimage.measure import label, regionprops
from tqdm import tqdm


def resample_raster(raster_to_resample_path, reference_raster_path, output_path):
    """
    Resample a raster to match the dimensions and transform of a reference raster.
    
    Args:
        raster_to_resample_path (str): Path to the raster that needs resampling
        reference_raster_path (str): Path to the reference raster for target dimensions
        output_path (str): Path where the resampled raster will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with rasterio.open(reference_raster_path) as ref_src:
            target_height = ref_src.height
            target_width = ref_src.width
            target_transform = ref_src.transform

            with rasterio.open(raster_to_resample_path) as src:
                # Resample data to match reference dimensions
                resampled_data = src.read(
                    out_shape=(src.count, target_height, target_width),
                    resampling=Resampling.nearest,
                )

                # Clean up nodata values: replace -2 with -1
                resampled_data[resampled_data == -2] = -1

                # Update profile for output
                output_profile = src.profile.copy()
                output_profile.update({
                    "transform": target_transform,
                    "width": target_width,
                    "height": target_height,
                    "compress": "deflate",
                    "dtype": "int16",
                })

                # Save resampled raster
                with rasterio.open(output_path, "w", **output_profile) as dst:
                    dst.write(resampled_data)

        print(f"Raster resampled successfully: {output_path}")
        return True

    except Exception as e:
        print(f"Error during raster resampling: {e}")
        return False

def canny_edge_detection(image_path, low_threshold, high_threshold, output_path=None):
    """
    Apply Canny edge detection to a grayscale raster image.
    
    Args:
        image_path (str): Path to input grayscale image
        low_threshold (int): Lower threshold for edge detection
        high_threshold (int): Upper threshold for edge detection  
        output_path (str, optional): Path to save edge detection result
        
    Returns:
        numpy.ndarray: Binary edge image (0s and 255s)
    """
    try:
        # Load grayscale image
        with rasterio.open(image_path) as src:
            grayscale_image = src.read(1)
            image_metadata = src.meta.copy()

        # Apply Canny edge detection algorithm
        edge_result = cv2.Canny(grayscale_image, low_threshold, high_threshold)

        # Save result if output path specified
        if output_path:
            # Update metadata for edge output
            image_metadata.update({
                'count': 1, 
                'dtype': 'uint8', 
                'compress': 'deflate', 
                'nodata': 30
            })

            with rasterio.open(output_path, "w", **image_metadata) as dst:
                dst.write(edge_result, 1)

            print(f"Canny edge detection completed: {output_path}")

        return edge_result

    except Exception as e:
        print(f"Error in Canny edge detection: {e}")
        return None

def adapt_flow_direction(flow_direction_array, connectivity=4):
    """
    Adapt flow direction values for terrace detection by converting to specific directional codes.
    
    Args:
        flow_direction_array (numpy.ndarray): Flow direction raster array
        connectivity (int): Either 4 or 8 for connectivity type
        
    Returns:
        numpy.ndarray: Adapted flow direction array suitable for contour detection
    """
    # Create a copy to avoid modifying the original array
    adapted_directions = flow_direction_array.copy()
    
    if connectivity == 8:
        # 8-connectivity direction mapping
        direction_mapping = {
            360: 270, 45: 315, 90: 360, 135: 45,
            180: 90, 225: 135, 270: 180, 315: 225
        }
    elif connectivity == 4:
        # 4-connectivity direction mapping (combines opposite directions)
        # Group opposite directions together
        adapted_directions[(adapted_directions == 360) | (adapted_directions == 180)] = 90
        adapted_directions[(adapted_directions == 45) | (adapted_directions == 225)] = 135
        adapted_directions[(adapted_directions == 90) | (adapted_directions == 270)] = 360
        adapted_directions[(adapted_directions == 135) | (adapted_directions == 315)] = 45
        
        return adapted_directions
    else:
        raise ValueError("Connectivity must be either 4 or 8")
    
    # Apply 8-connectivity mapping
    for original_dir, new_dir in direction_mapping.items():
        adapted_directions[adapted_directions == original_dir] = new_dir
    
    return adapted_directions

def detect_terraces(edge_image_path, flow_direction_path, output_path, directions1or3=3):
    try:
        with rasterio.open(edge_image_path) as src:
            edge_image = src.read(1)
            meta = src.meta.copy()

        with rasterio.open(flow_direction_path) as src_contour:
            contour_direction = src_contour.read(1)

        # Ensure the edge image is binary
        edge_image = edge_image >= 1

        # Adapt flow direction
        contour_direction = adapt_flow_direction(contour_direction, 4)

        rownum, colnum = edge_image.shape
        valid_directions = {360, 45, 90, 135, 180, 225, 270, 315}
        unique_coordinates_all = set()

        def process_directions(edge_img, dir_offsets):
            terraces = []
            for row in tqdm(range(rownum)):
                for col in range(colnum):
                    current_row, current_col = row, col
                    current_ter = []
                    while edge_img[current_row, current_col] == 1:
                        coords = (current_row, current_col)
                        current_ter.append(coords)
                        current_cell_dir = contour_direction[current_row, current_col]

                        if current_cell_dir not in valid_directions:
                            break

                        offset = dir_offsets.get(current_cell_dir, None)
                        if offset:
                            new_row, new_col = current_row + offset[0], current_col + offset[1]
                            if 0 <= new_row < rownum and 0 <= new_col < colnum:
                                current_row, current_col = new_row, new_col
                            else:
                                break
                        else:
                            break

                    if len(current_ter) > 1:
                        for ter in current_ter:
                            edge_img[ter[0], ter[1]] = 0
                            unique_coordinates_all.add(ter)
                        terraces.append(current_ter)
            return terraces

        primary_dir_offsets = {
            360: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)
        }
        plus45_dir_offsets = {
            360: (1, 1), 45: (1, 0), 90: (1, -1), 135: (0, 1)
        }
        minus45_dir_offsets = {
            360: (1, -1), 45: (0, 1), 90: (1, 1), 135: (1, 0)
        }

        if directions1or3 == 3:
            print("Processing primary directions...")
            process_directions(edge_image.copy(), primary_dir_offsets)
            print("Processing +45 directions...")
            process_directions(edge_image.copy(), plus45_dir_offsets)
            print("Processing -45 directions...")
            process_directions(edge_image.copy(), minus45_dir_offsets)
        elif directions1or3 == 1:
            print("Processing primary directions only...")
            process_directions(edge_image.copy(), primary_dir_offsets)
        else:
            print("Invalid directions1or3 value; must be 1 or 3.")
            return

        terraces_raster = np.zeros_like(edge_image)
        merged_coordinates = list(unique_coordinates_all)

        for coord in merged_coordinates:
            terraces_raster[coord[0], coord[1]] = 1

        labeled_terraces = label(terraces_raster, connectivity=2)
        regions = regionprops(labeled_terraces)
        merged_terraces = [region.coords.tolist() for region in regions]

        lengths = [len(terrace) for terrace in merged_terraces]
        average_length = statistics.mean(lengths)
        filtered_terraces = [terrace for terrace in merged_terraces if len(terrace) > average_length]

        filtered_terraces_raster = np.zeros_like(edge_image)
        for terrace in filtered_terraces:
            for coord in terrace:
                filtered_terraces_raster[coord[0], coord[1]] = 1

        meta.update(count=1, dtype="uint8", compress="deflate", nodata=30)

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(filtered_terraces_raster, 1)

        print(f"Terraces detection completed. Output saved to: {output_path}")
        return filtered_terraces_raster

    except Exception as e:
        print(f"An error occurred in detect_terraces: {e}")

def back_detection(canny_edges_path, terrace_detect_path, save_path, connectivity=8):
    try:
        # Open the Canny edges image
        with rasterio.open(canny_edges_path) as src:
            canny_image = src.read(1)
            profile = src.profile

        # Open the terrace detection result image
        with rasterio.open(terrace_detect_path) as src_contour:
            directional_result_image = src_contour.read(1)

        # Set terrace detections to 255 for overlaying
        directional_result_image[directional_result_image == 1] = 255

        # Negate the Canny edges image
        negated_canny_image = cv2.bitwise_not(canny_image)

        # Resize the directional result image to match the size of the negated Canny image
        directional_result_resized = cv2.resize(
            directional_result_image,
            (negated_canny_image.shape[1], negated_canny_image.shape[0]),
        )

        # Create a 3-channel image for overlaying
        overlay = cv2.cvtColor(negated_canny_image, cv2.COLOR_GRAY2BGR)

        # Create a mask from the directional result
        _, mask = cv2.threshold(directional_result_resized, 0, 255, cv2.THRESH_BINARY)

        # Overlay the directional result on the negated Canny image
        overlay[mask > 0] = [0, 50, 0]

        # Create a binary mask
        value1 = [0, 50, 0]
        value2 = [0, 0, 0]

        mask = np.all((overlay == value1) | (overlay == value2), axis=-1)
        mask = (mask * 255).astype(np.uint8)

        # Negate the mask for flood fill
        mask = cv2.bitwise_not(mask)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

        # Flood Fill settings
        height, width, _ = overlay.shape
        new_color = [0, 255, 0]
        lo_diff = [0, 50, 0]

        # Perform flood fill on the overlay
        with tqdm(total=height * width, desc="Flood Fill Progress") as pbar:
            for x in range(height):
                for y in range(width):
                    if (overlay[x, y] == [0, 50, 0]).all():
                        cv2.floodFill(
                            overlay,
                            mask,
                            (y, x),
                            newVal=new_color,
                            loDiff=lo_diff,
                            flags=connectivity,
                        )
                    pbar.update(1)

        # Convert the overlay to grayscale and adjust values
        gray_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        gray_image[gray_image == 255] = 0
        gray_image[gray_image == 150] = 255

        # Save the resulting image
        profile.update(count=1, dtype="uint8", compress="deflate", nodata=30)
        with rasterio.open(save_path, "w", **profile) as dst:
            dst.write(gray_image, 1)

        print(f"Back detection completed. Output saved to: {save_path}")
        return gray_image

    except Exception as e:
        print(f"An error occurred in back_detection: {e}")

def calculate_sinuosity(contour):
    rect = cv2.minAreaRect(contour)  # bounding rectangle
    box = cv2.boxPoints(rect)  # rectangle 4 points

    # Extract the first and third points (indices 0 and 2) from the box
    point1 = box[0]
    point2 = box[2]

    # Calculate the diagonal length (straight-line distance) using the Euclidean distance formula
    straight_line_distance = np.sqrt(
        (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
    )

    actual_path_length = cv2.arcLength(contour, closed=False)  # actual length

    sinuosity = actual_path_length / straight_line_distance

    return sinuosity

def filter_contours(terrace_path, output_path):
    try:
        # Open the terrace detection result image
        with rasterio.open(terrace_path) as src:
            edge_image = src.read(1)
            profile = src.profile

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate sinuosity and length for all contours
        sinuosity_list = []
        length_list = []

        for contour in contours:
            contour_length = cv2.arcLength(contour, closed=False)
            length_list.append(contour_length)

            sinuosity = calculate_sinuosity(contour)
            sinuosity_list.append(sinuosity)

        # Convert lists to numpy arrays for statistical calculations
        length_array = np.array(length_list)
        sinuosity_array = np.array(sinuosity_list)

        # Calculate length and sinuosity thresholds
        min_length = np.min(length_array) * 2
        max_length = np.mean(length_array) + 2 * np.std(length_array)
        max_sin = np.mean(sinuosity_array) + 2 * np.std(sinuosity_array)

        # Filter contours based on length and sinuosity thresholds
        filtered_contours = []
        filtered_sinuosity_list = []
        filtered_length_list = []

        for contour in tqdm(contours, desc="Filtering Contours"):
            contour_length = cv2.arcLength(contour, closed=False)
            sinuosity = calculate_sinuosity(contour)
            
            if sinuosity < max_sin and min_length < contour_length < max_length:
                filtered_contours.append(contour)
                filtered_sinuosity_list.append(sinuosity)
                filtered_length_list.append(contour_length)

        # Create a mask to draw the filtered contours
        filtered_contour_mask = np.zeros_like(edge_image)
        cv2.drawContours(filtered_contour_mask, filtered_contours, -1, 255, thickness=1)

        # Save the filtered contours mask
        profile.update(count=1, dtype="uint8", compress="deflate", nodata=30)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(filtered_contour_mask, 1)

        print(f"Contour processing completed. Output saved to: {output_path}")
        return filtered_contour_mask

    except Exception as e:
        print(f"An error occurred in process_contours: {e}")

#-----------------------------------------------------------------------------------------
## call of the functions
#-----------------------------------------------------------------------------------------

def edge_detection(image_path, canny_low_threshold, canny_high_threshold, flow_direction_raster, 
                  output_dir=None, num_directions=3, connectivity=8):
    """
    Complete terrace edge detection pipeline combining Canny edge detection,
    directional analysis, and contour filtering.
    
    Args:
        image_path (str): Path to input grayscale image
        canny_low_threshold (int): Lower threshold for Canny edge detection
        canny_high_threshold (int): Upper threshold for Canny edge detection
        flow_direction_raster (str): Path to flow direction raster
        output_dir (str, optional): Output directory (defaults to image directory)
        num_directions (int): Number of directional passes (1 or 3)
        connectivity (int): Connectivity for back detection (4 or 8)
        
    Returns:
        tuple: (canny_edges_path, filtered_edges_path) - Paths to output files
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    # Step 1: Canny edge detection
    canny_output_path = os.path.join(output_dir, f"canny_{canny_low_threshold}_{canny_high_threshold}.tif")
    print(f"Step 1/5: Running Canny edge detection ({canny_low_threshold}-{canny_high_threshold})...")
    canny_edge_detection(image_path, canny_low_threshold, canny_high_threshold, canny_output_path)

    # Step 2: Resample flow direction to match Canny output
    resampled_flowdir_path = os.path.join(output_dir, "flowdir_resampled_temp.tif")
    print("Step 2/5: Resampling flow direction raster...")
    resample_raster(flow_direction_raster, canny_output_path, resampled_flowdir_path)

    # Step 3: Directional terrace detection
    directional_output_path = os.path.join(output_dir, f"directional_temp_{canny_low_threshold}_{canny_high_threshold}.tif")
    print("Step 3/5: Detecting terraces using directional analysis...")
    detect_terraces(canny_output_path, resampled_flowdir_path, directional_output_path, directions1or3=num_directions)

    # Step 4: Back detection using flood fill
    back_detection_path = os.path.join(output_dir, f"back_detection_temp_{canny_low_threshold}_{canny_high_threshold}.tif")
    print("Step 4/5: Applying back detection with flood fill...")
    back_detection(canny_output_path, directional_output_path, back_detection_path, connectivity=connectivity)

    # Step 5: Filter contours by sinuosity and length
    final_output_path = os.path.join(output_dir, f"edges_{canny_low_threshold}_{canny_high_threshold}.tif")
    print("Step 5/5: Filtering contours by geometry criteria...")
    filter_contours(back_detection_path, final_output_path)

    # Clean up temporary files
    temp_files = [resampled_flowdir_path, directional_output_path, back_detection_path]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"Edge detection pipeline completed successfully!")
    print(f"  Canny edges: {canny_output_path}")
    print(f"  Filtered edges: {final_output_path}")

    return canny_output_path, final_output_path

# Example usage:
# if __name__ == "__main__":
#     image_path = "./data/grayscale_image.tif"
#     flow_dir_path = "./data/flow_direction.tif"
#     canny_edges, filtered_edges = edge_detection(
#         image_path, 
#         canny_low_threshold=200, 
#         canny_high_threshold=400,
#         flow_direction_raster=flow_dir_path,
#         num_directions=3,
#         connectivity=8
#     )

