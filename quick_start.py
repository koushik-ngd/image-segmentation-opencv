#!/usr/bin/env python3
"""
Quick Start Script for Image Segmentation Project
This script creates a sample image and demonstrates basic segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_segmentation import ImageSegmentation

def create_demo_image():
    """Create a demo image with different colored regions"""
    print("Creating demo image...")
    
    # Create a 400x400 image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add different colored regions
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)      # Blue
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)               # Green
    cv2.rectangle(img, (100, 250), (200, 350), (0, 0, 255), -1)    # Red
    cv2.ellipse(img, (300, 300), (60, 40), 45, 0, 360, (255, 255, 0), -1)  # Yellow
    
    # Add some text
    cv2.putText(img, "Demo", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, (400, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def quick_demo():
    """Run a quick demonstration of segmentation"""
    print("=" * 50)
    print("Image Segmentation Quick Demo")
    print("=" * 50)
    
    # Create demo image
    demo_img = create_demo_image()
    
    # Save demo image
    cv2.imwrite("demo_image.jpg", demo_img)
    print("Demo image saved as 'demo_image.jpg'")
    
    # Initialize segmentation
    seg = ImageSegmentation()
    
    # Load the demo image
    seg.load_image("demo_image.jpg")
    print("Demo image loaded successfully")
    
    # Test different segmentation methods
    methods = [
        ("Threshold", lambda: seg.threshold_segmentation(threshold_value=127)),
        ("K-Means (K=3)", lambda: seg.kmeans_segmentation(k=3)),
        ("SLIC", lambda: seg.slic_segmentation(n_segments=100, compactness=10))
    ]
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Apply each method
    for i, (method_name, method_func) in enumerate(methods):
        try:
            print(f"Testing {method_name}...")
            result = method_func()
            
            # Display result
            row = i // 3 + 1
            col = i % 3 + 1
            
            if len(result.shape) == 3:
                axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                axes[row, col].imshow(result, cmap='gray')
            
            axes[row, col].set_title(method_name)
            axes[row, col].axis('off')
            
            print(f"✓ {method_name} completed successfully")
            
        except Exception as e:
            print(f"✗ {method_name} failed: {str(e)}")
    
    # Hide unused subplots
    for i in range(2):
        for j in range(4):
            if i == 0 and j > 0:
                axes[i, j].set_visible(False)
            elif i == 1 and j > 3:
                axes[i, j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=150, bbox_inches='tight')
    print("Results saved as 'quick_demo_results.png'")
    
    # Save the plot without showing (headless mode)
    plt.close()  # Close the figure to free memory
    
    # Get statistics
    try:
        stats = seg.get_segmentation_stats()
        print("\nSegmentation Statistics:")
        print(f"  Original image shape: {stats['original_shape']}")
        print(f"  Segmented image shape: {stats['segmented_shape']}")
        print(f"  Unique values: {stats['unique_values']}")
    except Exception as e:
        print(f"Could not get statistics: {str(e)}")
    
    print("\nQuick demo completed!")
    print("Generated files:")
    print("  - demo_image.jpg")
    print("  - quick_demo_results.png")

if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"Error during quick demo: {str(e)}")
        print("Make sure all dependencies are installed correctly.")
