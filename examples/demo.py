#!/usr/bin/env python3
"""
MimirMap Example Script

This script demonstrates how to use the MimirMap library to estimate GPS coordinates
of objects in monocular images using MiDaS depth estimation and geometric calculations.
"""

import os
import argparse
import cv2
import matplotlib.pyplot as plt
from mimirmap.core import (
    load_midas_model,
    estimate_depth,
    calculate_depth_confidence,
    estimate_object_gps,
    save_depth_heatmap,
    calculate_ground_distance,
    get_fov_from_camera_params
)

def demo_interactive_mode(image_path, camera_lat, camera_lon, camera_alt, 
                          camera_model, camera_height, pitch, yaw, roll):
    """
    Demonstrate interactive mode where the user can click on the image to get GPS coordinates.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Define window name
    window_name = "MimirMap - Click to get GPS coordinates (Press ESC to exit)"
    
    # Load MiDaS model (do this once)
    print("Loading MiDaS model (this may take a moment)...")
    model, transform = load_midas_model()
    
    # Estimate depth (do this once)
    print("Estimating depth map...")
    _, depth_map, reference_pixel, reference_distance = estimate_depth(
        image_path, model, transform,
        auto_calibrate=True,
        camera_height=camera_height,
        pitch_degrees=pitch,
        gopro_model=camera_model
    )
    
    # Function to handle mouse clicks
    clicks = []
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            
            # Get depth at clicked point
            depth_value = depth_map[y, x]
            confidence = calculate_depth_confidence(x, y, depth_map)
            
            # Calculate distance using alternative method for comparison
            h, w = img.shape[:2]
            fov = get_fov_from_camera_params(camera_model)
            geom_dist = calculate_ground_distance(y, h, camera_height, pitch, fov["vertical"])
            
            # Estimate GPS
            lat, lon, alt, conf, dist, _ = estimate_object_gps(
                image_path, x, y,
                camera_lat, camera_lon, camera_alt,
                camera_model, yaw, pitch, roll, camera_height,
                model, transform,
                visualize=False
            )
            
            # Annotate the clicked point
            img_copy = img.copy()
            
            # Draw all previous clicks
            for i, (cx, cy) in enumerate(clicks):
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(img_copy, f"#{i+1}", (cx+5, cy-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display information
            print(f"\nPoint #{len(clicks)} at ({x}, {y}):")
            print(f"  Depth: {depth_value:.2f}m (confidence: {confidence:.2f})")
            print(f"  Ground geometry distance: {geom_dist:.2f}m")
            print(f"  Estimated distance: {dist:.2f}m")
            print(f"  Estimated GPS: {lat:.6f}, {lon:.6f}, {alt:.2f}m")
            print(f"  Google Maps: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
            
            # Update display
            cv2.imshow(window_name, img_copy)
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)
    
    # Show depth map in a separate window
    depth_img = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
    cv2.imshow("Depth Map", depth_colored)
    
    # Show the image and wait for clicks
    cv2.imshow(window_name, img)
    print("Click on the image to estimate GPS coordinates. Press ESC to exit.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Clean up
    cv2.destroyAllWindows()

def demo_batch_mode(image_path, points, camera_lat, camera_lon, camera_alt, 
                    camera_model, camera_height, pitch, yaw, roll, output_dir):
    """
    Demonstrate batch mode where GPS coordinates are estimated for a list of points.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MiDaS model
    print("Loading MiDaS model...")
    model, transform = load_midas_model()
    
    # Process each point
    results = []
    for i, (x, y) in enumerate(points):
        print(f"\nProcessing point #{i+1} at ({x}, {y})...")
        
        # Estimate GPS with visualizations
        lat, lon, alt, confidence, distance, visualizations = estimate_object_gps(
            image_path, x, y,
            camera_lat, camera_lon, camera_alt,
            camera_model, yaw, pitch, roll, camera_height,
            model, transform,
            visualize=True
        )
        
        # Store results
        results.append({
            'point_id': i+1,
            'coordinates': (x, y),
            'gps': (lat, lon, alt),
            'distance': distance,
            'confidence': confidence,
            'visualizations': visualizations
        })
        
        # Copy visualizations to output directory with numbered filenames
        for vis_type, vis_path in visualizations.items():
            if vis_path and os.path.exists(vis_path):
                ext = os.path.splitext(vis_path)[1]
                new_path = os.path.join(output_dir, f"point{i+1}_{vis_type}{ext}")
                cv2.imwrite(new_path, cv2.imread(vis_path))
        
        # Print results
        print(f"  Estimated distance: {distance:.2f}m")
        print(f"  Estimated GPS: {lat:.6f}, {lon:.6f}, {alt:.2f}m")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Google Maps: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
    
    # Create a summary visualization
    img = cv2.imread(image_path)
    for result in results:
        x, y = result['coordinates']
        point_id = result['point_id']
        distance = result['distance']
        
        # Draw circle and add label
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, f"#{point_id}: {distance:.1f}m", (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save summary image
    summary_path = os.path.join(output_dir, "summary.jpg")
    cv2.imwrite(summary_path, img)
    print(f"\nSaved summary image to: {summary_path}")
    
    # Create depth map visualization
    _, depth_map = estimate_depth(image_path, model, transform)[:2]
    depth_path = os.path.join(output_dir, "depth_map.jpg")
    save_depth_heatmap(depth_map, depth_path)
    print(f"Saved depth map to: {depth_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="MimirMap Example Script")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--lat", type=float, default=-34.636059, help="Camera latitude")
    parser.add_argument("--lon", type=float, default=-58.706453, help="Camera longitude")
    parser.add_argument("--alt", type=float, default=50.0, help="Camera altitude (meters)")
    parser.add_argument("--camera-model", type=str, default="HERO8", help="Camera model")
    parser.add_argument("--camera-height", type=float, default=1.4, help="Camera height (meters)")
    parser.add_argument("--pitch", type=float, default=12.0, help="Camera pitch (degrees)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Camera yaw (degrees)")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll (degrees)")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], 
                       default="interactive", help="Demo mode")
    parser.add_argument("--points", type=str, default="", 
                       help="Points for batch mode, format: 'x1,y1;x2,y2;...'")
    parser.add_argument("--output-dir", type=str, default="./output", 
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    print("\n===== MimirMap Demo =====")
    print(f"Image: {args.image_path}")
    print(f"Camera position: {args.lat}, {args.lon}, {args.alt}m")
    print(f"Camera model: {args.camera_model}")
    print(f"Camera height: {args.camera_height}m")
    print(f"Camera orientation: Yaw={args.yaw}°, Pitch={args.pitch}°, Roll={args.roll}°")
    print(f"Mode: {args.mode}")
    
    if args.mode == "interactive":
        demo_interactive_mode(
            args.image_path, args.lat, args.lon, args.alt,
            args.camera_model, args.camera_height, args.pitch, args.yaw, args.roll
        )
    else:  # batch mode
        if not args.points:
            print("Error: Batch mode requires --points argument")
            return
        
        # Parse points string (format: x1,y1;x2,y2;...)
        try:
            point_list = []
            for point_str in args.points.split(';'):
                if point_str:
                    x, y = map(int, point_str.split(','))
                    point_list.append((x, y))
            
            if not point_list:
                raise ValueError("No valid points found")
        except Exception as e:
            print(f"Error parsing points: {e}")
            print("Format should be: x1,y1;x2,y2;...")
            return
        
        print(f"Processing {len(point_list)} points: {point_list}")
        demo_batch_mode(
            args.image_path, point_list,
            args.lat, args.lon, args.alt,
            args.camera_model, args.camera_height, args.pitch, args.yaw, args.roll,
            args.output_dir
        )
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()