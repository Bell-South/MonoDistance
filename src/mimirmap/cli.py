import argparse
import os
from mimirmap.old_core import (
    load_midas_model,
    estimate_depth,
    get_depth_at_pixel,
    calculate_depth_confidence,
    estimate_distance_from_depth,
    calculate_ground_distance,
    project_gps,
    annotate_image,
    get_fov_from_camera_params,
    save_depth_heatmap,
    estimate_object_gps
)
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Estimate object distance and GPS from monocular image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("x", type=int, help="X coordinate of the object in image")
    parser.add_argument("y", type=int, help="Y coordinate of the object in image")
    parser.add_argument("latitude", type=float, help="Camera latitude")
    parser.add_argument("longitude", type=float, help="Camera longitude")
    
    # Optional arguments with defaults
    parser.add_argument("--output-dir", type=str, default="./output", 
                        help="Directory to save output files")
    parser.add_argument("--altitude", type=float, default=0.0,
                        help="Camera altitude in meters")
    parser.add_argument("--camera-height", type=float, default=1.4,
                        help="Camera height above ground in meters")
    parser.add_argument("--pitch", type=float, default=12.0,
                        help="Camera pitch in degrees (downward tilt)")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Camera yaw in degrees (0=North, 90=East)")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Camera roll in degrees")
    parser.add_argument("--camera-model", type=str, default="HERO8",
                        help="Camera model (HERO7, HERO8, HERO9, HERO10)")
    parser.add_argument("--method", type=str, choices=["auto", "geometry", "depth"], 
                        default="auto", help="Method to estimate distance")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for depth estimation (only used with --method=depth)")
    parser.add_argument("--save-depth", action="store_true",
                        help="Save depth map visualization")
    parser.add_argument("--save-3d", action="store_true",
                        help="Save 3D position visualization")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")

    args = parser.parse_args()

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Processing image: {args.image_path}")
        print(f"Target coordinates: ({args.x}, {args.y})")
        print(f"Camera GPS: {args.latitude}, {args.longitude}, {args.altitude}m")
        print(f"Camera orientation: Yaw={args.yaw}°, Pitch={args.pitch}°, Roll={args.roll}°")
        print(f"Camera model: {args.camera_model}")
        print(f"Camera height above ground: {args.camera_height}m")
    
    # Calculate output paths
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    annotated_path = os.path.join(args.output_dir, f"{base_filename}_annotated.jpg")
    depth_path = os.path.join(args.output_dir, f"{base_filename}_depth.jpg") if args.save_depth else None
    position_3d_path = os.path.join(args.output_dir, f"{base_filename}_3d.jpg") if args.save_3d else None
    
    if args.method == "auto":
        # Use the comprehensive estimation function
        if args.verbose:
            print("Using auto-calibrated method with depth estimation and geometry")
        
        # Load MiDaS model
        model, transform = load_midas_model()
        
        # Estimate object GPS
        lat, lon, alt, confidence, distance, visualizations = estimate_object_gps(
            args.image_path, args.x, args.y,
            args.latitude, args.longitude, args.altitude,
            args.camera_model,
            args.yaw, args.pitch, args.roll,
            args.camera_height,
            model, transform,
            visualize=True
        )
        
        # Copy visualizations to the output directory if needed
        if visualizations:
            for vis_type, vis_path in visualizations.items():
                if vis_path and os.path.exists(vis_path):
                    # If output paths are different from the auto-generated ones
                    if vis_type == 'annotated_image' and annotated_path != vis_path:
                        cv2.imwrite(annotated_path, cv2.imread(vis_path))
                    elif vis_type == 'depth_map' and depth_path and depth_path != vis_path:
                        cv2.imwrite(depth_path, cv2.imread(vis_path))
                    elif vis_type == '3d_position' and position_3d_path and position_3d_path != vis_path:
                        cv2.imwrite(position_3d_path, cv2.imread(vis_path))
        
    elif args.method == "geometry":
        # Load image to get dimensions
        img = cv2.imread(args.image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {args.image_path}")
        
        # Get camera FOV
        fov = get_fov_from_camera_params(args.camera_model)
        vertical_fov = fov["vertical"]
        
        # Calculate distance using ground plane geometry
        distance = calculate_ground_distance(
            args.y, img.shape[0], args.camera_height, args.pitch, vertical_fov
        )
        
        # Project GPS coordinates
        lat, lon = project_gps(args.latitude, args.longitude, args.yaw, distance)
        alt = args.altitude  # No altitude change in geometry method
        
        # Calculate confidence (fixed for geometry method)
        confidence = 0.7  # Simple geometry is reasonably confident but not perfect
        
        # Create annotated image
        annotated = annotate_image(img, (args.x, args.y), f"{distance:.1f}m")
        cv2.imwrite(annotated_path, annotated)
        
    else:  # args.method == "depth"
        # Load MiDaS model
        model, transform = load_midas_model()
        
        # Estimate depth
        img, depth_map = estimate_depth(args.image_path, model, transform, auto_calibrate=False)[:2]
        
        # Get depth at target pixel
        depth_value = get_depth_at_pixel(depth_map, args.x, args.y)
        
        # Apply scaling
        distance = estimate_distance_from_depth(depth_value, args.scale)
        
        # Calculate confidence
        confidence = calculate_depth_confidence(args.x, args.y, depth_map)
        
        # Project GPS coordinates
        lat, lon = project_gps(args.latitude, args.longitude, args.yaw, distance)
        alt = args.altitude  # No altitude change in simple depth method
        
        # Create annotated image
        annotated = annotate_image(img, (args.x, args.y), f"{distance:.1f}m")
        cv2.imwrite(annotated_path, annotated)
        
        # Save depth visualization if requested
        if args.save_depth and depth_path:
            save_depth_heatmap(depth_map, depth_path, (args.x, args.y), distance)
    
    # Print results
    print(f"\nResults for point ({args.x}, {args.y}):")
    print(f"Estimated distance: {distance:.2f} meters")
    print(f"Estimated GPS: {lat:.6f}, {lon:.6f}, {alt:.2f}m")
    print(f"Confidence: {confidence:.2f} (scale 0-1)")
    
    # Print output file locations
    print(f"\nSaved annotated image to: {annotated_path}")
    if args.save_depth and depth_path:
        print(f"Saved depth visualization to: {depth_path}")
    if args.save_3d and position_3d_path and os.path.exists(position_3d_path):
        print(f"Saved 3D position visualization to: {position_3d_path}")
    
    # Print Google Maps link
    maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    print(f"\nView on Google Maps: {maps_link}")


if __name__ == "__main__":
    main()