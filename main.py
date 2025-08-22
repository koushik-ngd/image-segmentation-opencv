#!/usr/bin/env python3
"""
Image Segmentation Project using OpenCV
Main application file with command-line interface
"""

import os
import sys
import argparse
from image_segmentation import ImageSegmentation

def main():
    parser = argparse.ArgumentParser(description='Image Segmentation using OpenCV')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--method', '-m', choices=['threshold', 'adaptive', 'watershed', 'kmeans', 'felzenszwalb', 'slic', 'grabcut'], 
                       default='threshold', help='Segmentation method to use')
    parser.add_argument('--output', '-o', help='Output path for segmented image')
    parser.add_argument('--display', '-d', action='store_true', help='Display results')
    parser.add_argument('--params', '-p', nargs='*', help='Additional parameters for segmentation')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)
    
    # Initialize segmentation
    seg = ImageSegmentation()
    
    try:
        # Load image
        print(f"Loading image: {args.image_path}")
        seg.load_image(args.image_path)
        
        # Apply segmentation based on method
        print(f"Applying {args.method} segmentation...")
        
        if args.method == 'threshold':
            threshold = 127
            if args.params:
                threshold = int(args.params[0])
            result = seg.threshold_segmentation(threshold_value=threshold)
            
        elif args.method == 'adaptive':
            block_size = 11
            c = 2
            if args.params:
                block_size = int(args.params[0])
                if len(args.params) > 1:
                    c = int(args.params[1])
            result = seg.adaptive_threshold_segmentation(block_size=block_size, c=c)
            
        elif args.method == 'watershed':
            markers = 10
            if args.params:
                markers = int(args.params[0])
            result = seg.watershed_segmentation(markers_count=markers)
            
        elif args.method == 'kmeans':
            k = 3
            if args.params:
                k = int(args.params[0])
            result = seg.kmeans_segmentation(k=k)
            
        elif args.method == 'felzenszwalb':
            scale = 100
            sigma = 0.5
            min_size = 50
            if args.params:
                scale = int(args.params[0])
                if len(args.params) > 1:
                    sigma = float(args.params[1])
                if len(args.params) > 2:
                    min_size = int(args.params[2])
            result = seg.felzenszwalb_segmentation(scale=scale, sigma=sigma, min_size=min_size)
            
        elif args.method == 'slic':
            n_segments = 100
            compactness = 10
            if args.params:
                n_segments = int(args.params[0])
                if len(args.params) > 1:
                    compactness = int(args.params[1])
            result = seg.slic_segmentation(n_segments=n_segments, compactness=compactness)
            
        elif args.method == 'grabcut':
            result = seg.grabcut_segmentation()
        
        print("Segmentation completed successfully!")
        
        # Get statistics
        stats = seg.get_segmentation_stats()
        print(f"Segmentation statistics:")
        print(f"  Original image shape: {stats['original_shape']}")
        print(f"  Segmented image shape: {stats['segmented_shape']}")
        print(f"  Unique values: {stats['unique_values']}")
        
        # Save result if output path specified
        if args.output:
            seg.save_segmented_image(args.output)
        else:
            # Generate default output path
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            output_path = f"{base_name}_{args.method}_segmented.jpg"
            seg.save_segmented_image(output_path)
        
        # Display results if requested
        if args.display:
            seg.display_results()
            
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
