import cv2
import torch
import numpy as np
import math
import os
from scipy.spatial.transform import Rotation as R
from pyproj import Geod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_midas_model(model_type="MiDaS_small"):
    """
    Load MiDaS depth estimation model with model selection.
    
    Args:
        model_type: Type of MiDaS model to load. Options:
                   - "MiDaS_small": Faster but less accurate
                   - "MiDaS": Original model, more accurate but slower
                   - "DPT_Large": Best quality, slowest
                   - "DPT_Hybrid": Balanced quality and speed
    
    Returns:
        model: Loaded MiDaS model
        transform: Appropriate transform for the model
    """
    # Check for valid model type
    valid_models = ["MiDaS_small", "MiDaS", "DPT_Large", "DPT_Hybrid"]
    if model_type not in valid_models:
        print(f"Warning: Invalid model_type '{model_type}'. Using 'MiDaS_small' instead.")
        model_type = "MiDaS_small"
    
    print(f"Loading {model_type} model...")
    
    # Load model
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()
    
    # Select appropriate transform
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
    elif model_type == "MiDaS_small":
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    else:  # MiDaS
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).default_transform
    
    return midas, transform, device

# Usage example:
# model, transform, device = load_midas_model("DPT_Hybrid")

def estimate_depth_enhanced_calibration(image_path, model, transform, device=None, 
                                        auto_calibrate=True, camera_height=1.4, 
                                        pitch_degrees=12.0, gopro_model="HERO8"):
    """
    Estimate depth with improved scaling using multiple reference points.
    
    Args:
        image_path: Path to the input image
        model: MiDaS model
        transform: MiDaS transform
        device: Computation device (cpu or cuda)
        auto_calibrate: Whether to auto-calibrate the depth map
        camera_height: Height of the camera from the ground in meters
        pitch_degrees: Downward pitch of the camera in degrees
        gopro_model: GoPro camera model for FOV estimation
        
    Returns:
        img: Original image
        depth_map_meters: Depth map in meters
        reference_pixels: Reference points used for calibration
        reference_distances: Reference distances in meters
    """
    # Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device if device else next(model.parameters()).device)

    # Generate depth prediction
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Handle negative depth values
    prediction = np.maximum(prediction, 0.1)  # Ensure positive depth with minimum threshold

    if auto_calibrate:
        # Get camera FOV information
        camera_fov = get_fov_from_camera_params(gopro_model)
        vertical_fov = camera_fov["vertical"]

        # Get image dimensions
        image_height = img.shape[0]
        image_width = img.shape[1]

        # Define multiple reference points at different positions on the ground
        # We'll sample a grid across the bottom half of the image
        num_horizontal = 5  # Number of horizontal sample points
        num_vertical = 3    # Number of vertical sample points
        reference_pixels = []
        reference_distances = []
        
        # Use only lower half of image (more likely to be ground)
        min_v = int(image_height * 0.5)  
        
        # Sample points in a grid pattern
        for v_idx in range(num_vertical):
            v = min_v + (image_height - min_v) * (v_idx + 0.5) / num_vertical
            v = int(v)
            
            for h_idx in range(num_horizontal):
                u = int(image_width * (h_idx + 0.5) / num_horizontal)
                
                # Calculate the reference distance using ground plane geometry
                ref_distance = calculate_ground_distance(
                    v, image_height, camera_height, pitch_degrees, vertical_fov
                )
                
                # Only use points with valid distances
                if ref_distance > 0 and ref_distance < 100:  # Reasonable range check
                    reference_pixels.append((u, v))
                    reference_distances.append(ref_distance)
        
        if not reference_pixels:
            print("Warning: Could not find valid reference points. Using default calibration.")
            # Default to a single reference point if no valid points found
            v = int(image_height * 0.75)
            u = int(image_width / 2)
            ref_distance = calculate_ground_distance(
                v, image_height, camera_height, pitch_degrees, vertical_fov
            )
            reference_pixels = [(u, v)]
            reference_distances = [ref_distance]
        
        # Calculate scaling factors for each reference point
        scaling_factors = []
        for (u, v), ref_distance in zip(reference_pixels, reference_distances):
            depth_at_ref = prediction[v, u]
            if depth_at_ref > 0.1:
                scaling_factors.append(ref_distance / depth_at_ref)
        
        # Remove outliers (RANSAC-like approach)
        if len(scaling_factors) > 3:
            # Sort and take the middle 60% of values to remove outliers
            scaling_factors.sort()
            start_idx = int(len(scaling_factors) * 0.2)
            end_idx = int(len(scaling_factors) * 0.8)
            robust_scaling_factors = scaling_factors[start_idx:end_idx]
            depth_scale = np.median(robust_scaling_factors)
        else:
            # If we have too few points, use the median
            depth_scale = np.median(scaling_factors) if scaling_factors else 1.0
        
        # Apply scaling - convert to meters
        depth_map_meters = prediction * depth_scale

        # Apply a bilateral filter to reduce noise while preserving edges
        depth_map_meters = cv2.bilateralFilter(depth_map_meters.astype(np.float32),
                                              d=7, sigmaColor=0.1, sigmaSpace=5.0)

        return img, depth_map_meters, reference_pixels, reference_distances
    else:
        # Return the raw depth map without calibration
        return img, prediction, None, None
    
def estimate_depth(image_path, model, transform, auto_calibrate=True, camera_height=1.4, pitch_degrees=12.0, gopro_model="HERO8"):
    """
    Estimate depth with improved scaling to absolute metrics using ground plane calibration.
    
    Args:
        image_path: Path to the input image
        model: MiDaS model
        transform: MiDaS transform
        auto_calibrate: Whether to auto-calibrate the depth map using ground plane geometry
        camera_height: Height of the camera from the ground in meters
        pitch_degrees: Downward pitch of the camera in degrees
        gopro_model: GoPro camera model for FOV estimation
        
    Returns:
        img: Original image
        depth_map_meters: Depth map in meters
        reference_pixel: Reference point used for calibration (if auto_calibrate=True)
        reference_distance: Reference distance in meters (if auto_calibrate=True)
    """
    # Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(next(model.parameters()).device)

    # Generate depth prediction
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Handle negative depth values
    prediction = np.maximum(prediction, 0.1)  # Ensure positive depth with minimum threshold

    if auto_calibrate:
        # Get camera FOV information
        camera_fov = get_fov_from_camera_params(gopro_model)
        vertical_fov = camera_fov["vertical"]

        # Determine reference point for ground plane (visible road start)
        image_height = img.shape[0]
        image_width = img.shape[1]

        # Use a point approximately 3/4 down the image for the visible road
        ground_ref_v = int(image_height * 0.75)
        ground_ref_u = int(image_width / 2)  # Center horizontally

        # Calculate the reference distance using ground plane geometry
        reference_distance = calculate_ground_distance(
            ground_ref_v, image_height, camera_height, pitch_degrees, vertical_fov
        )

        # Get depth at the reference point
        relative_depth_at_reference = prediction[ground_ref_v, ground_ref_u]

        # Calculate scaling factor
        if relative_depth_at_reference > 0.1:
            depth_scale = reference_distance / relative_depth_at_reference
        else:
            # Try to find a better reference point nearby
            window_size = 25
            y_min = max(0, ground_ref_v - window_size)
            y_max = min(image_height, ground_ref_v + window_size + 1)
            x_min = max(0, ground_ref_u - window_size)
            x_max = min(image_width, ground_ref_u + window_size + 1)

            window = prediction[y_min:y_max, x_min:x_max]
            max_depth_idx = np.unravel_index(window.argmax(), window.shape)

            # Convert to image coordinates
            better_v = y_min + max_depth_idx[0]
            better_u = x_min + max_depth_idx[1]
            better_depth = prediction[better_v, better_u]

            # Calculate new reference distance
            new_reference_distance = calculate_ground_distance(
                better_v, image_height, camera_height, pitch_degrees, vertical_fov
            )

            depth_scale = new_reference_distance / better_depth
            
            # Update reference point
            ground_ref_u, ground_ref_v = better_u, better_v
            reference_distance = new_reference_distance

        # Apply scaling - convert to meters
        depth_map_meters = prediction * depth_scale

        # Apply a bilateral filter to reduce noise while preserving edges
        depth_map_meters = cv2.bilateralFilter(depth_map_meters.astype(np.float32),
                                              d=7, sigmaColor=0.1, sigmaSpace=5.0)

        reference_pixel = (ground_ref_u, ground_ref_v)
        return img, depth_map_meters, reference_pixel, reference_distance
    else:
        # Return the raw depth map without calibration
        return img, prediction, None, None

def get_depth_at_pixel(depth_map, x, y):
    """Get depth value at specific pixel coordinates."""
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        return depth_map[y, x]
    else:
        raise ValueError(f"Pixel coordinates ({x}, {y}) are outside the image bounds: {depth_map.shape[1]}x{depth_map.shape[0]}")

def estimate_distance_from_depth(depth_value, scale=1.0):
    """Convert depth value to distance in meters."""
    return depth_value * scale

def calculate_depth_confidence(x, y, depth_map, window_size=5):
    """
    Calculate confidence in the depth estimate based on depth variation in the local neighborhood.
    Lower variation indicates higher confidence.

    Args:
        x, y: Pixel coordinates
        depth_map: Depth map
        window_size: Size of the neighborhood window to analyze

    Returns:
        confidence: 0-1 value representing confidence (1 = highest)
    """
    # Create a small window around the point
    y_min = max(0, y-window_size)
    y_max = min(depth_map.shape[0], y+window_size+1)
    x_min = max(0, x-window_size)
    x_max = min(depth_map.shape[1], x+window_size+1)

    local_region = depth_map[y_min:y_max, x_min:x_max]

    # Calculate coefficient of variation (std/mean) as a measure of consistency
    mean_depth = np.mean(local_region)
    if mean_depth > 0:
        std_depth = np.std(local_region)
        coeff_variation = std_depth / mean_depth

        # Convert to confidence (0-1 scale)
        # Lower variation = higher confidence
        confidence = max(0, min(1, 1 - coeff_variation))
    else:
        confidence = 0

    return confidence

def calculate_ground_distance(v, image_height, camera_height, pitch_deg, v_fov_deg):
    """
    Calculate the distance to a point on the ground plane using perspective geometry.

    Args:
        v: Vertical pixel coordinate (from top of image)
        image_height: Height of the image in pixels
        camera_height: Height of the camera from the ground in meters
        pitch_deg: Downward pitch of the camera in degrees
        v_fov_deg: Vertical field of view in degrees

    Returns:
        distance: Distance to the ground point in meters
    """
    # Calculate the angle for each pixel
    deg_per_pixel = v_fov_deg / image_height

    # Get center of image
    center_v = image_height / 2

    # Calculate angle from optical axis (negative for points below center)
    pixel_angle = (center_v - v) * deg_per_pixel

    # Total angle from horizontal
    total_angle_rad = math.radians(pitch_deg - pixel_angle)

    # Calculate distance using trigonometry (adjacent = opposite / tan(angle))
    if total_angle_rad > 0:  # Make sure we're looking downward
        distance = camera_height / math.tan(total_angle_rad)
        return distance
    else:
        return float('inf')  # Point is above horizon

def get_fov_from_camera_params(gopro_model):
    """Get the diagonal, horizontal, and vertical FOV for a camera model."""
    # Default FOV values for different GoPro models
    camera_fov = {
        "HERO8": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8},
        "HERO9": {"diagonal": 84, "horizontal": 73.6, "vertical": 53.4},
        "HERO10": {"diagonal": 84, "horizontal": 73.6, "vertical": 53.4},
        "HERO7": {"diagonal": 78, "horizontal": 66.9, "vertical": 45.8},
        "DEFAULT": {"diagonal": 80, "horizontal": 69.5, "vertical": 49.8}
    }

    return camera_fov.get(gopro_model.upper(), camera_fov["DEFAULT"])

def get_camera_parameters(model="HERO8", resolution=None):
    """
    Get generic camera parameters based on model and resolution.
    Values are approximate and should be replaced with actual calibration when possible.

    Args:
        model: Camera model (e.g., "HERO8")
        resolution: Tuple (width, height) of image resolution
        
    Returns:
        fx, fy, cx, cy (camera intrinsics)
    """
    if resolution is None:
        # Use a default resolution if none provided
        width, height = 2666, 2000
    else:
        width, height = resolution
        
    # Generic camera parameters
    camera_params = {
        "HERO8": {
            "focal_length_mm": 2.8,
            "sensor_width_mm": 6.17,
            "sensor_height_mm": 4.55,
            "fov_degrees": get_fov_from_camera_params("HERO8")["diagonal"],
        },
        "HERO9": {
            "focal_length_mm": 2.92,
            "sensor_width_mm": 6.20,
            "sensor_height_mm": 4.65,
            "fov_degrees": get_fov_from_camera_params("HERO9")["diagonal"],
        },
        "HERO10": {
            "focal_length_mm": 2.92,
            "sensor_width_mm": 6.20,
            "sensor_height_mm": 4.65,
            "fov_degrees": get_fov_from_camera_params("HERO10")["diagonal"],
        },
        # Default for unknown models
        "DEFAULT": {
            "focal_length_mm": 2.8,
            "sensor_width_mm": 6.0,
            "sensor_height_mm": 4.5,
            "fov_degrees": get_fov_from_camera_params("DEFAULT")["diagonal"],
        }
    }

    # Get parameters for the specified model or use DEFAULT
    params = camera_params.get(model.upper(), camera_params["DEFAULT"])

    # Calculate focal length in pixels using field of view
    fx = width / (2 * math.tan(math.radians(params["fov_degrees"] / 2)))
    fy = height / (2 * math.tan(math.radians(params["fov_degrees"] * height / width / 2)))

    # Principal point (usually at the center of the image)
    cx = width / 2
    cy = height / 2

    return fx, fy, cx, cy

def pixel_to_camera_coords(u, v, Z, fx, fy, cx, cy):
    """Convert pixel coordinates to camera coordinates."""
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

def apply_orientation(point, yaw, pitch, roll):
    """Apply camera orientation to the point in camera coordinates."""
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    return r.apply(point)

def post_process_depth_map(depth_map, image, method='advanced'):
    """
    Apply advanced post-processing to the depth map.
    
    Args:
        depth_map: Input depth map
        image: Original RGB image
        method: Post-processing method to use
               - 'basic': Simple bilateral filtering
               - 'guided': Guided filtering using RGB image
               - 'advanced': Combination of multiple techniques
               
    Returns:
        processed_depth: Processed depth map
    """
    # Convert to float32 for processing
    depth_map = depth_map.astype(np.float32)
    
    if method == 'basic':
        # Simple bilateral filtering
        processed_depth = cv2.bilateralFilter(depth_map, d=7, sigmaColor=0.1, sigmaSpace=5.0)
        
    elif method == 'guided':
        # Convert image to grayscale for guided filter
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Apply guided filter
        processed_depth = cv2.ximgproc.guidedFilter(
            guide, depth_map, radius=8, eps=0.1**2
        )
        
    elif method == 'advanced':
        # Step 1: Initial bilateral filtering to reduce noise
        filtered_depth = cv2.bilateralFilter(depth_map, d=7, sigmaColor=0.1, sigmaSpace=5.0)
        
        # Step 2: Edge-aware smoothing with guided filter
        gray_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        guided_depth = cv2.ximgproc.guidedFilter(
            gray_guide, filtered_depth, radius=8, eps=0.1**2
        )
        
        # Step 3: Detect edges in the image
        edges = cv2.Canny(image, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        
        # Step 4: Detect depth discontinuities
        depth_edges = cv2.Laplacian(filtered_depth, cv2.CV_32F)
        depth_edges = np.abs(depth_edges) > 0.1 * np.max(np.abs(depth_edges))
        depth_edges = depth_edges.astype(np.uint8) * 255
        
        # Step 5: Combine edge information
        combined_edges = cv2.bitwise_or(edges_dilated, depth_edges)
        edge_mask = combined_edges > 0
        
        # Step 6: Blend based on edge information
        processed_depth = np.copy(guided_depth)
        # Keep original depth near edges for better boundary preservation
        processed_depth[edge_mask] = filtered_depth[edge_mask]
        
        # Step 7: Fill holes using inpainting on areas with very low confidence
        invalid_mask = (depth_map < 0.1).astype(np.uint8) * 255
        if np.any(invalid_mask):
            processed_depth = cv2.inpaint(
                processed_depth, invalid_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
            )
            
        # Step 8: Final smoothing with small bilateral filter
        processed_depth = cv2.bilateralFilter(
            processed_depth, d=5, sigmaColor=0.05, sigmaSpace=2.0
        )
    else:
        raise ValueError(f"Unknown post-processing method: {method}")
    
    return processed_depth

def visualize_depth_comparison(original_depth, processed_depth, image_path=None):
    """
    Visualize original and processed depth maps side by side.
    
    Args:
        original_depth: Original depth map
        processed_depth: Processed depth map
        image_path: Path to save the visualization (None for display only)
    """
    # Normalize for visualization
    orig_viz = cv2.normalize(original_depth, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    proc_viz = cv2.normalize(processed_depth, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Apply color maps
    orig_colored = cv2.applyColorMap(orig_viz, cv2.COLORMAP_INFERNO)
    proc_colored = cv2.applyColorMap(proc_viz, cv2.COLORMAP_INFERNO)
    
    # Create difference map
    diff = np.abs(processed_depth - original_depth)
    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    diff_colored = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
    
    # Create comparison visualization
    h, w = original_depth.shape
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    comparison[:, 0:w, :] = orig_colored
    comparison[:, w:2*w, :] = proc_colored
    comparison[:, 2*w:3*w, :] = diff_colored
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Processed', (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Difference', (2*w+10, 30), font, 1, (255, 255, 255), 2)
    
    if image_path:
        cv2.imwrite(image_path, comparison)
        return image_path
    else:
        return comparison

def calculate_improved_confidence(x, y, depth_map, rgb_image=None, window_size=7):
    """
    Calculate confidence in the depth estimate using multiple metrics.
    
    Args:
        x, y: Pixel coordinates
        depth_map: Depth map
        rgb_image: Original RGB image (optional, for additional metrics)
        window_size: Size of the neighborhood window to analyze
    
    Returns:
        confidence: 0-1 value representing confidence (1 = highest)
        confidence_components: Dictionary of individual confidence metrics
    """
    # Ensure coordinates are valid
    h, w = depth_map.shape
    if not (0 <= y < h and 0 <= x < w):
        return 0.0, {'valid': 0.0}
    
    # Create a window around the point
    y_min = max(0, y - window_size)
    y_max = min(h, y + window_size + 1)
    x_min = max(0, x - window_size)
    x_max = min(w, x + window_size + 1)
    
    # Extract local regions
    local_depth = depth_map[y_min:y_max, x_min:x_max]
    
    confidence_components = {}
    
    # 1. Calculate depth consistency (lower variation = higher confidence)
    mean_depth = np.mean(local_depth)
    if mean_depth > 0:
        std_depth = np.std(local_depth)
        coeff_variation = std_depth / mean_depth
        # Convert to confidence (0-1 scale)
        depth_consistency = max(0, min(1, 1 - coeff_variation))
    else:
        depth_consistency = a0
    confidence_components['depth_consistency'] = depth_consistency
    
    # 2. Edge confidence - lower confidence near depth discontinuities
    if local_depth.size > 4:  # Ensure we have enough pixels
        # Calculate gradient magnitude
        gx = cv2.Sobel(local_depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(local_depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Normalize gradient magnitude to 0-1 range
        max_grad = np.max(grad_mag)
        if max_grad > 0:
            norm_grad = grad_mag / max_grad
        else:
            norm_grad = np.zeros_like(grad_mag)
        
        # Center pixel gradient value
        center_y = min(y - y_min, local_depth.shape[0] - 1)
        center_x = min(x - x_min, local_depth.shape[1] - 1)
        center_grad = norm_grad[center_y, center_x]
        
        # Convert to confidence (lower gradient = higher confidence)
        edge_confidence = max(0, min(1, 1 - center_grad))
    else:
        edge_confidence = 0.5  # Default if window is too small
    confidence_components['edge_confidence'] = edge_confidence
    
    # 3. Depth value confidence - higher for more typical values
    # Typical outdoor depth values are between 1-50m
    center_depth = depth_map[y, x]
    if 0.5 <= center_depth <= 50:
        # Higher confidence for depths in the typical range
        # Use logistic function centered at 5m with width parameter of 10
        depth_value_conf = 1.0 - 1.0 / (1.0 + np.exp(-(np.abs(center_depth - 5.0) / 10.0)))
    else:
        # Low confidence for extreme values
        depth_value_conf = 0.2
    confidence_components['depth_value'] = depth_value_conf
    
    # 4. Image texture confidence (if RGB image is provided)
    if rgb_image is not None:
        # Extract local RGB region
        local_rgb = rgb_image[y_min:y_max, x_min:x_max]
        
        # Convert to grayscale and calculate texture variance
        if len(local_rgb.shape) == 3:  # Ensure it's a color image
            local_gray = cv2.cvtColor(local_rgb, cv2.COLOR_BGR2GRAY)
            texture_variance = np.var(local_gray)
            
            # Normalize to 0-1 (higher variance = higher confidence up to a point)
            texture_conf = min(1.0, texture_variance / 400.0)  # 400 is a normalization factor
        else:
            texture_conf = 0.5  # Default if grayscale
        confidence_components['texture'] = texture_conf
    
    # 5. Combine metrics with weighted average
    weights = {
        'depth_consistency': 0.4,
        'edge_confidence': 0.3,
        'depth_value': 0.2,
        'texture': 0.1
    }
    
    total_confidence = 0.0
    total_weight = 0.0
    
    for metric, value in confidence_components.items():
        if metric in weights:
            total_confidence += value * weights[metric]
            total_weight += weights[metric]
    
    if total_weight > 0:
        final_confidence = total_confidence / total_weight
    else:
        final_confidence = 0.5  # Default confidence
    
    return final_confidence, confidence_components

def visualize_confidence_map(depth_map, rgb_image=None, output_path=None):
    """
    Generate a confidence map visualization for the entire depth map.
    
    Args:
        depth_map: Depth map
        rgb_image: Original RGB image (optional)
        output_path: Path to save visualization
        
    Returns:
        confidence_map: 2D confidence map
        visualization: Colored visualization
    """
    h, w = depth_map.shape
    confidence_map = np.zeros((h, w), dtype=np.float32)
    
    # For speed, calculate confidence at a lower resolution
    stride = 8
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            conf, _ = calculate_improved_confidence(x, y, depth_map, rgb_image)
            # Fill the block with the same confidence value
            y_end = min(y + stride, h)
            x_end = min(x + stride, w)
            confidence_map[y:y_end, x:x_end] = conf
    
    # Create visualization
    conf_viz = (confidence_map * 255).astype(np.uint8)
    conf_colored = cv2.applyColorMap(conf_viz, cv2.COLORMAP_JET)
    
    if output_path:
        cv2.imwrite(output_path, conf_colored)
    
    return confidence_map, conf_colored

def project_gps(lat, lon, bearing, distance_m, altitude_change=0):
    """
    Project a point from a starting GPS position along a bearing for a specified distance.
    
    Args:
        lat, lon: Starting coordinates in decimal degrees
        bearing: Direction in degrees (0=North, 90=East, etc.)
        distance_m: Distance in meters
        altitude_change: Change in altitude (vertical) in meters
        
    Returns:
        lat2, lon2: Projected coordinates in decimal degrees
        alt2: New altitude
    """
    geod = Geod(ellps='WGS84')
    lon2, lat2, _ = geod.fwd(lon, lat, bearing, distance_m)
    return lat2, lon2

def local_to_gps(offset, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to GPS.
    
    Args:
        offset: Numpy array [east, up, north] in meters
        lat0, lon0, alt0: Reference GPS coordinates
        
    Returns:
        lat1, lon1, alt1: Resulting GPS coordinates
    """
    geod = Geod(ellps="WGS84")
    east, up, north = offset[0], offset[1], offset[2]
    horizontal_dist = np.hypot(east, north)
    azimuth = np.degrees(np.arctan2(east, north)) % 360
    lon1, lat1, _ = geod.fwd(lon0, lat0, azimuth, horizontal_dist)
    return lat1, lon1, alt0 + up

def annotate_image(img, point, label, color=(0, 255, 0), radius=10, font_scale=0.6, thickness=2):
    """
    Annotate an image with a point and label.
    
    Args:
        img: Input image
        point: (x, y) coordinates to mark
        label: Text label to add
        color: BGR color tuple
        radius: Circle radius
        font_scale: Size of text
        thickness: Line thickness
        
    Returns:
        Annotated image
    """
    annotated = img.copy()
    cv2.circle(annotated, point, radius, color, -1)
    cv2.putText(annotated, label, (point[0]+10, point[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return annotated

def save_depth_heatmap(depth_map, output_path, reference_pixel=None, reference_distance=None):
    """
    Save a heatmap visualization of the depth map.
    
    Args:
        depth_map: Depth map array
        output_path: Path to save the visualization
        reference_pixel: Optional reference pixel (x,y) to mark
        reference_distance: Optional reference distance to display
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth (meters)')
    
    if reference_pixel is not None:
        plt.scatter(reference_pixel[0], reference_pixel[1], c='white', s=50)
        if reference_distance is not None:
            plt.annotate(f"{reference_distance:.1f}m", 
                         (reference_pixel[0]+10, reference_pixel[1]-10),
                         color='white', fontsize=8)
    
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_3d_position(camera_coords, object_coords, reference_coords=None, output_path=None):
    """
    Visualize the 3D positioning between camera and object.
    
    Args:
        camera_coords: Camera position (typically origin)
        object_coords: Object position in 3D space
        reference_coords: Optional reference point coordinates
        output_path: Path to save the visualization
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera at origin
    ax.scatter([camera_coords[0]], [camera_coords[1]], [camera_coords[2]], 
               color='red', s=100, label='Camera')

    # Plot object
    ax.scatter([object_coords[0]], [object_coords[1]], [object_coords[2]],
               color='blue', s=100, label='Object')

    # Draw line from camera to object
    ax.plot([camera_coords[0], object_coords[0]], 
            [camera_coords[1], object_coords[1]], 
            [camera_coords[2], object_coords[2]], 'b--')

    # Plot reference point if provided
    if reference_coords is not None:
        ax.scatter([reference_coords[0]], [reference_coords[1]], [reference_coords[2]],
                   color='green', s=100, label='Reference')
        # Draw line from camera to reference
        ax.plot([camera_coords[0], reference_coords[0]], 
                [camera_coords[1], reference_coords[1]], 
                [camera_coords[2], reference_coords[2]], 'g--')

    # Set labels
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('3D Positioning')
    ax.legend()

    # Equal aspect ratio 
    max_range = max(1.0, np.max(np.abs([
        object_coords[0], object_coords[1], object_coords[2],
        0, 0, 0  # Camera is at origin
    ])))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def estimate_object_gps(
    image_path, x, y,
    camera_lat, camera_lon, camera_alt,
    camera_model="HERO8",
    yaw=0.0, pitch=12.0, roll=0.0,
    camera_height=1.4,
    model=None, transform=None,
    visualize=False
):
    """
    Estimate the GPS coordinates of an object in the image.
    
    Args:
        image_path: Path to the input image
        x, y: Pixel coordinates of the object
        camera_lat, camera_lon, camera_alt: Camera GPS coordinates
        camera_model: Camera model for intrinsics estimation
        yaw, pitch, roll: Camera orientation in degrees
        camera_height: Height of the camera from ground in meters
        model: Optional pre-loaded MiDaS model
        transform: Optional pre-loaded MiDaS transform
        visualize: Whether to create and return visualizations
        
    Returns:
        lat, lon, alt: Estimated GPS coordinates of the object
        confidence: Confidence score for the estimation (0-1)
        distance: Estimated distance in meters
        visualizations: Dict of visualization paths if visualize=True
    """
    # Load MiDaS model if not provided
    if model is None or transform is None:
        model, transform = load_midas_model()
    
    # Estimate depth with auto-calibration
    img, depth_map, reference_pixel, reference_distance = estimate_depth(
        image_path, model, transform,
        auto_calibrate=True,
        camera_height=camera_height,
        pitch_degrees=pitch,
        gopro_model=camera_model
    )
    
    # Get image dimensions
    image_height, image_width = img.shape[:2]
    resolution = (image_width, image_height)
    
    # Get camera intrinsics
    fx, fy, cx, cy = get_camera_parameters(camera_model, resolution)
    
    # Get camera FOV information
    camera_fov = get_fov_from_camera_params(camera_model)
    vertical_fov = camera_fov["vertical"]
    
    # Check if pixel coordinates are valid
    if not (0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]):
        raise ValueError(f"Pixel coordinates ({x}, {y}) are outside the image bounds")

    # Get depth at the specified pixel
    Z = depth_map[y, x]
    
    # Check if the depth is too small, try to find a better point nearby
    if Z < 0.5:  # If depth is very small, likely inaccurate
        # Find the point with highest depth in a small window
        window_size = 25
        y_min = max(0, y - window_size)
        y_max = min(depth_map.shape[0], y + window_size + 1)
        x_min = max(0, x - window_size)
        x_max = min(depth_map.shape[1], x + window_size + 1)

        window = depth_map[y_min:y_max, x_min:x_max]
        max_depth_idx = np.unravel_index(window.argmax(), window.shape)

        # Convert to image coordinates
        better_y = y_min + max_depth_idx[0]
        better_x = x_min + max_depth_idx[1]
        better_Z = depth_map[better_y, better_x]

        if better_Z > Z:
            x, y, Z = better_x, better_y, better_Z
    
    # Calculate confidence in the depth estimate
    confidence = calculate_depth_confidence(x, y, depth_map)
    
    # If we still have a very low depth, use a fallback based on the auto-calibration
    if Z < 0.5:
        # Estimate distance based on position in image using ground plane geometry
        fallback_distance = calculate_ground_distance(
            y, image_height, camera_height, pitch, vertical_fov
        )
        
        # Only use this if the point is likely on the ground plane
        if 0 < fallback_distance < 100:  # Reasonable range check
            Z = fallback_distance
        else:
            # Use reference distance as approximate value
            Z = reference_distance
    
    # Convert pixel to camera coordinates
    point_cam = pixel_to_camera_coords(x, y, Z, fx, fy, cx, cy)
    
    # Apply camera orientation
    point_world = apply_orientation(point_cam, yaw, -pitch, roll)  # negate pitch if Y is down
    
    # Convert to ENU coordinates (East-North-Up)
    point_world_ENU = np.array([point_world[0], -point_world[1], point_world[2]])  # flip Y to up
    
    # Calculate straight-line distance
    distance = np.linalg.norm(point_world_ENU)
    
    # Convert to GPS
    lat, lon, alt = local_to_gps(point_world_ENU, camera_lat, camera_lon, camera_alt)
    
    # Create visualizations if requested
    visualizations = {}
    if visualize:
        # Annotate the original image
        annotated_img = annotate_image(img, (x, y), f"{Z:.1f}m")
        annotated_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
        cv2.imwrite(annotated_path, annotated_img)
        visualizations['annotated_image'] = annotated_path
        
        # Create depth heatmap
        heatmap_path = os.path.splitext(image_path)[0] + "_depth.jpg"
        save_depth_heatmap(depth_map, heatmap_path, (x, y), Z)
        visualizations['depth_map'] = heatmap_path
        
        # Create 3D visualization
        vis_3d_path = os.path.splitext(image_path)[0] + "_3d.jpg"
        camera_origin = np.array([0, 0, 0])
        visualize_3d_position(camera_origin, point_world_ENU, None, vis_3d_path)
        visualizations['3d_position'] = vis_3d_path
    
    return lat, lon, alt, confidence, distance, visualizations if visualize else None

def estimate_depth_hybrid(image_path, model, transform, device=None,
                         camera_height=1.4, pitch_degrees=12.0, 
                         gopro_model="HERO8", blend_factor=0.5):
    """
    Estimate depth using a hybrid approach that combines MiDaS depth estimation
    with ground plane geometry.
    
    Args:
        image_path: Path to the input image
        model: MiDaS model
        transform: MiDaS transform
        device: Computation device
        camera_height: Height of the camera from the ground in meters
        pitch_degrees: Downward pitch of the camera in degrees
        gopro_model: GoPro camera model for FOV estimation
        blend_factor: How much to blend between methods (0=pure geometry, 1=pure depth)
        
    Returns:
        img: Original image
        hybrid_depth_map: Combined depth map in meters
        confidence_map: Pixel-wise confidence map
    """
    # Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    image_height, image_width = img.shape[:2]
    
    # Get camera FOV information
    camera_fov = get_fov_from_camera_params(gopro_model)
    vertical_fov = camera_fov["vertical"]
    
    # 1. Generate MiDaS depth map
    input_batch = transform(img_rgb).to(device if device else next(model.parameters()).device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    # Ensure positive values
    midas_depth = np.maximum(prediction, 0.1)
    
    # Apply auto-calibration to get metric depths
    # Simplified from your original code
    min_v = int(image_height * 0.5)
    max_v = int(image_height * 0.9)
    center_u = int(image_width / 2)
    
    # Sample point at 3/4 down the image for reference
    ref_v = int(image_height * 0.75)
    ref_distance = calculate_ground_distance(
        ref_v, image_height, camera_height, pitch_degrees, vertical_fov
    )
    ref_depth = midas_depth[ref_v, center_u]
    scale_factor = ref_distance / ref_depth
    
    midas_depth_meters = midas_depth * scale_factor
    
    # 2. Generate geometry-based depth map
    geometry_depth = np.zeros_like(midas_depth)
    
    # For each pixel, calculate the ground distance
    for v in range(image_height):
        # Calculate distance using ground plane geometry
        geom_distance = calculate_ground_distance(
            v, image_height, camera_height, pitch_degrees, vertical_fov
        )
        
        # Fill the row with this distance
        geometry_depth[v, :] = geom_distance
    
    # Set points above horizon to a large value
    geometry_depth[geometry_depth < 0] = 100.0
    geometry_depth[geometry_depth == float('inf')] = 100.0
    
    # 3. Generate confidence maps for both approaches
    # MiDaS confidence based on edge consistency
    midas_confidence = np.ones_like(midas_depth_meters)
    
    # Edge detection on depth map
    depth_edges = cv2.Laplacian(midas_depth_meters, cv2.CV_32F)
    depth_edges = np.abs(depth_edges)
    
    # Lower confidence near edges
    edge_threshold = 0.1 * np.max(depth_edges)
    edge_mask = depth_edges > edge_threshold
    midas_confidence[edge_mask] *= 0.5
    
    # Lower confidence for extreme values
    extreme_mask = (midas_depth_meters < 0.5) | (midas_depth_meters > 50.0)
    midas_confidence[extreme_mask] *= 0.3
    
    # Geometry confidence based on vertical position
    # Higher confidence for lower parts of the image (more likely to be ground)
    geometry_confidence = np.zeros_like(geometry_depth)
    for v in range(image_height):
        # Linearly decreasing confidence from bottom to horizon
        relative_position = 1.0 - (v / image_height)
        row_confidence = max(0.0, min(1.0, 1.0 - relative_position))
        geometry_confidence[v, :] = row_confidence
    
    # 4. Create surface normal map to help identify ground plane
    # Sobel derivatives
    gx = cv2.Sobel(midas_depth_meters, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(midas_depth_meters, cv2.CV_32F, 0, 1, ksize=3)
    
    # Normal vectors (z component is 1 for simplicity)
    normal_x = -gx
    normal_y = -gy
    normal_z = np.ones_like(gx)
    
    # Normalize
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    # Ground plane has normal pointing upward (normal_y is close to 1)
    ground_likelihood = normal_y
    
    # 5. Blend the depth maps based on confidence and ground likelihood
    hybrid_depth_map = np.zeros_like(midas_depth_meters)
    combined_confidence = np.zeros_like(midas_depth_meters)
    
    for v in range(image_height):
        for u in range(image_width):
            # Base confidence from both methods
            m_conf = midas_confidence[v, u]
            g_conf = geometry_confidence[v, u]
            
            # Adjust confidence based on ground likelihood
            ground_factor = max(0, ground_likelihood[v, u])
            g_conf *= (0.2 + 0.8 * ground_factor)  # Boost geometry confidence for ground points
            
            # Calculate adaptive blend factor
            adaptive_blend = blend_factor * m_conf / (m_conf + g_conf + 1e-6)
            
            # Apply blending
            midas_value = midas_depth_meters[v, u]
            geom_value = geometry_depth[v, u]
            
            # Special case for likely sky/far background
            if v < image_height * 0.4 and midas_value > 20:
                # Far background - trust MiDaS more
                adaptive_blend = 0.9
            
            # Weighted blend
            hybrid_depth_map[v, u] = (
                adaptive_blend * midas_value + 
                (1 - adaptive_blend) * geom_value
            )
            
            # Combined confidence is the max of individual confidences
            combined_confidence[v, u] = max(m_conf * adaptive_blend, 
                                           g_conf * (1 - adaptive_blend))
    
    # 6. Post-process the hybrid depth map
    # Simple bilateral filter
    hybrid_depth_map = cv2.bilateralFilter(
        hybrid_depth_map.astype(np.float32), d=7, sigmaColor=0.1, sigmaSpace=5.0
    )
    
    return img, hybrid_depth_map, combined_confidence

def calculate_object_distance_hybrid(depth_map, confidence_map, pixel_coords, 
                                    ground_depth, min_confidence=0.5):
    """
    Calculate object distance using the hybrid depth map with confidence.
    
    Args:
        depth_map: Hybrid depth map
        confidence_map: Confidence map
        pixel_coords: (x, y) pixel coordinates
        ground_depth: Ground plane depth for fallback
        min_confidence: Minimum confidence threshold
        
    Returns:
        distance: Estimated distance in meters
        confidence: Confidence in the estimate (0-1)
    """
    x, y = pixel_coords
    
    # Get depth at target pixel
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        depth = depth_map[y, x]
        conf = confidence_map[y, x]
    else:
        return ground_depth, 0.0  # Out of bounds
    
    # If confidence is too low, use neighboring pixels or fallback
    if conf < min_confidence:
        # Sample a small window around the point
        window_size = 5
        y_min = max(0, y - window_size)
        y_max = min(depth_map.shape[0], y + window_size + 1)
        x_min = max(0, x - window_size)
        x_max = min(depth_map.shape[1], x + window_size + 1)
        
        window_depths = depth_map[y_min:y_max, x_min:x_max]
        window_confs = confidence_map[y_min:y_max, x_min:x_max]
        
        # Find the point with highest confidence in the window
        max_conf_idx = np.unravel_index(window_confs.argmax(), window_confs.shape)
        best_depth = window_depths[max_conf_idx]
        best_conf = window_confs[max_conf_idx]
        
        if best_conf > min_confidence:
            return best_depth, best_conf
        else:
            # Blend with ground depth as fallback
            blend_ratio = conf / min_confidence
            blended_depth = blend_ratio * depth + (1 - blend_ratio) * ground_depth
            return blended_depth, conf
    
    return depth, conf


