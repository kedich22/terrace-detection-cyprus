"""
Cyprus Terrace Detection: Fast GLCM Texture Analysis Module

This module provides efficient computation of Gray-Level Co-occurrence Matrix (GLCM)
texture features for satellite imagery analysis. It includes optimized functions for
calculating Haralick texture statistics commonly used in remote sensing applications.

Key Functions:
- fast_glcm_omni: Compute GLCM for multiple angles and distances
- har_stats: Calculate Haralick texture statistics from GLCM

Features:
- Multi-directional texture analysis (0°, 45°, 90°, 135°, etc.)
- Configurable kernel sizes and gray levels
- Comprehensive Haralick texture measures
- Optimized for large raster processing

Author: Andrei Kedich
Date: 2025
"""

import numpy as np
import cv2


def fast_glcm_omni(image, value_min=0, value_max=255, gray_levels=8, kernel_size=5, 
                   pixel_distance=1.0, angles=[0.0]):
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM) for texture analysis.
    
    This function efficiently calculates GLCM texture features across multiple
    angles and distances for each pixel in the input image using a sliding window approach.
    
    Args:
        image (numpy.ndarray): Input grayscale image, shape=(height, width), dtype=uint8
        value_min (int): Minimum pixel value for digitization (default: 0)
        value_max (int): Maximum pixel value for digitization (default: 255)
        gray_levels (int): Number of gray levels for GLCM quantization (default: 8)
        kernel_size (int): Size of sliding window for local GLCM calculation (default: 5)
        pixel_distance (float): Distance between pixel pairs in pixels (default: 1.0)
        angles (list): List of angles in degrees for directional analysis (default: [0.0])
        
    Returns:
        numpy.ndarray: Averaged GLCM for each pixel, shape=(gray_levels, gray_levels, height, width)
        
    Example:
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> glcm = fast_glcm_omni(image, gray_levels=16, kernel_size=7, angles=[0, 45, 90, 135])
    """

    # Extract image dimensions and parameters
    height, width = image.shape
    ks = kernel_size

    # Digitize image values to gray levels
    bins = np.linspace(value_min, value_max + 1, gray_levels + 1)
    digitized_image = np.digitize(image, bins) - 1

    # Initialize average GLCM array
    average_glcm = np.zeros((gray_levels, gray_levels, height, width), dtype=np.float32)

    # Process each angle
    for angle in angles:
        # Calculate pixel displacement for current angle
        dx = pixel_distance * np.cos(np.deg2rad(angle))
        dy = pixel_distance * np.sin(np.deg2rad(-angle))
        
        # Create transformation matrix for pixel shifting
        transformation_matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
        
        # Create shifted version of digitized image
        shifted_image = cv2.warpAffine(
            digitized_image, 
            transformation_matrix, 
            (width, height), 
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Generate GLCM for current angle
        current_glcm = np.zeros((gray_levels, gray_levels, height, width), dtype=np.uint8)
        
        for i in range(gray_levels):
            for j in range(gray_levels):
                # Create mask for pixel pair co-occurrences
                co_occurrence_mask = ((digitized_image == i) & (shifted_image == j))
                current_glcm[i, j, co_occurrence_mask] = 1

        # Apply convolution kernel to accumulate local co-occurrences
        convolution_kernel = np.ones((ks, ks), dtype=np.uint8)
        for i in range(gray_levels):
            for j in range(gray_levels):
                current_glcm[i, j] = cv2.filter2D(current_glcm[i, j], -1, convolution_kernel)

        # Convert to float and add to average
        current_glcm = current_glcm.astype(np.float32)
        average_glcm += current_glcm

    # Average across all angles
    average_glcm /= len(angles)
    
    return average_glcm




def har_stats(image, value_min=0, value_max=255, gray_levels=16, kernel_size=31, 
              pixel_distance=1.0, angles=[0, 90, 180, 270], features_to_calculate="all"):

    """
    Calculate Haralick texture statistics from GLCM.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        value_min (int): Minimum pixel value (default: 0)
        value_max (int): Maximum pixel value (default: 255)
        gray_levels (int): Number of gray levels for GLCM (default: 16)
        kernel_size (int): Size of local window (default: 31)
        pixel_distance (float): Distance between pixel pairs (default: 1.0)
        angles (list): List of angles for directional analysis (default: [0, 90, 180, 270])
        features_to_calculate (str or list): Features to calculate - "all" or specific list
        
    Returns:
        list: List of texture feature arrays, one per requested feature
    """
    # Compute GLCM using the optimized function
    glcm = fast_glcm_omni(image, value_min, value_max, gray_levels, kernel_size, pixel_distance, angles)
    height, width = image.shape
    
    # Normalize GLCM - add small constant to avoid division by zero
    glcm_sum = np.sum(glcm, axis=(0, 1)) + 1.0 / (kernel_size ** 2)
    normalized_glcm = glcm / glcm_sum

    # Maximum probability feature
    max_probability = np.max(normalized_glcm, axis=(0, 1))
    
    # Initialize basic statistics
    mean_texture = np.zeros((height, width), dtype=np.float32)
    variance_texture = np.zeros((height, width), dtype=np.float32)
    std_texture = np.zeros((height, width), dtype=np.float32)
    
    # Compute mean texture
    if features_to_calculate == "all" or "mean" in features_to_calculate:
        for i in range(gray_levels):
            for j in range(gray_levels):
                mean_texture += normalized_glcm[i, j] * i / (gray_levels ** 2)

    # Compute standard deviation texture
    if features_to_calculate == "all" or "std" in features_to_calculate:
        for i in range(gray_levels):
            for j in range(gray_levels):
                variance_texture += normalized_glcm[i, j] * (i - mean_texture) ** 2
        std_texture = np.sqrt(variance_texture)  # Convert variance to standard deviation

    # Initialize Haralick texture feature arrays
    texture_features = {
        "homogeneity": np.zeros((height, width), dtype=np.float32),
        "dissimilarity": np.zeros((height, width), dtype=np.float32),
        "contrast": np.zeros((height, width), dtype=np.float32),
        "angular_second_moment": np.zeros((height, width), dtype=np.float32),
        "max_probability": np.zeros((height, width), dtype=np.float32),
        "entropy": np.zeros((height, width), dtype=np.float32),
        "cluster_prominence": np.zeros((height, width), dtype=np.float32),
        "cluster_shade": np.zeros((height, width), dtype=np.float32),
        "correlation": np.zeros((height, width), dtype=np.float32),
        "difference_entropy": np.zeros((height, width), dtype=np.float32),
        "difference_variance": np.zeros((height, width), dtype=np.float32),
        "inverse_difference": np.zeros((height, width), dtype=np.float32),
        "sum_average": np.zeros((height, width), dtype=np.float32),
        "sum_entropy": np.zeros((height, width), dtype=np.float32),
        "sum_squares": np.zeros((height, width), dtype=np.float32),
        "sum_variance": np.zeros((height, width), dtype=np.float32)
    }

    # Calculate core Haralick texture features
    for i in range(gray_levels):
        for j in range(gray_levels):
            # Calculate commonly used texture features
            if features_to_calculate == "all" or "homo" in features_to_calculate:
                texture_features["homogeneity"] += normalized_glcm[i, j] / (1.0 + (i - j) ** 2)
            
            if features_to_calculate == "all" or "cont" in features_to_calculate:
                texture_features["contrast"] += normalized_glcm[i, j] * (i - j) ** 2
            
            if features_to_calculate == "all" or "energy" in features_to_calculate:
                texture_features["angular_second_moment"] += normalized_glcm[i, j] ** 2
            
            if features_to_calculate == "all" or "entropy" in features_to_calculate:
                # Add small value to avoid log(0)
                texture_features["entropy"] -= normalized_glcm[i, j] * np.log(normalized_glcm[i, j] + 1e-10)
            
            if features_to_calculate == "all" or "cluster_shade" in features_to_calculate:
                texture_features["cluster_shade"] += normalized_glcm[i, j] * (i + j - 2 * mean_texture) ** 4

    # Calculate energy from angular second moment
    if features_to_calculate == "all" or "energy" in features_to_calculate:
        energy = np.sqrt(texture_features["angular_second_moment"])
    else:
        energy = texture_features["angular_second_moment"]

    # Prepare results based on requested features
    result_features = []
    
    # Define the standard order of features for consistency
    feature_order = ["mean", "std", "cont", "homo", "energy", "cluster_shade", "entropy"]
    
    for feature_name in feature_order:
        if features_to_calculate == "all" or feature_name in features_to_calculate:
            if feature_name == "mean":
                result_features.append(mean_texture)
            elif feature_name == "std":
                result_features.append(std_texture)
            elif feature_name == "cont":
                result_features.append(texture_features["contrast"])
            elif feature_name == "homo":
                result_features.append(texture_features["homogeneity"])
            elif feature_name == "energy":
                result_features.append(energy)
            elif feature_name == "cluster_shade":
                result_features.append(texture_features["cluster_shade"])
            elif feature_name == "entropy":
                result_features.append(texture_features["entropy"])

    return result_features