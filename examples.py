#!/usr/bin/env python3
"""
Examples and demonstrations for the Image Segmentation project
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_segmentation import ImageSegmentation
import os

def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple test image with different regions
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add different colored regions
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (100, 250), (200, 350), (0, 0, 255), -1)  # Red
    cv2.ellipse(img, (300, 300), (60, 40), 45, 0, 360, (255, 255, 0), -1)  # Yellow
    
    # Add some noise
    noise = np.random.randint(0, 50, (400, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def demonstrate_threshold_segmentation():
    """Demonstrate threshold segmentation"""
    print("=== Threshold Segmentation Demo ===")
    
    # Create sample image
    sample_img = create_sample_image()
    cv2.imwrite("sample_image.jpg", sample_img)
    
    # Initialize segmentation
    seg = ImageSegmentation()
    
    # Load the sample image
    seg.load_image("sample_image.jpg")
    
    # Apply different threshold values
    thresholds = [50, 100, 150, 200]
    
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    for i, thresh in enumerate(thresholds):
        result = seg.threshold_segmentation(threshold_value=thresh)
        plt.subplot(1, 5, i + 2)
        plt.imshow(result, cmap='gray')
        plt.title(f'Threshold = {thresh}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('threshold_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print("Threshold segmentation demo completed. Results saved to 'threshold_demo.png'")

def demonstrate_kmeans_segmentation():
    """Demonstrate K-means segmentation"""
    print("=== K-Means Segmentation Demo ===")
    
    # Create sample image
    sample_img = create_sample_image()
    cv2.imwrite("sample_image.jpg", sample_img)
    
    # Initialize segmentation
    seg = ImageSegmentation()
    seg.load_image("sample_image.jpg")
    
    # Apply different K values
    k_values = [2, 3, 4, 5, 6]
    
    plt.figure(figsize=(20, 4))
    
    # Original image
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    for i, k in enumerate(k_values):
        result = seg.kmeans_segmentation(k=k)
        plt.subplot(1, 6, i + 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'K = {k}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('kmeans_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print("K-means segmentation demo completed. Results saved to 'kmeans_demo.png'")

def demonstrate_watershed_segmentation():
    """Demonstrate watershed segmentation"""
    print("=== Watershed Segmentation Demo ===")
    
    # Create sample image
    sample_img = create_sample_image()
    cv2.imwrite("sample_image.jpg", sample_img)
    
    # Initialize segmentation
    seg = ImageSegmentation()
    seg.load_image("sample_image.jpg")
    
    # Apply watershed
    result = seg.watershed_segmentation()
    
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmented image
    plt.subplot(1, 3, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Watershed Segmentation')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    overlay = cv2.addWeighted(sample_img, 0.7, cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), 0.3, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('watershed_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print("Watershed segmentation demo completed. Results saved to 'watershed_demo.png'")

def demonstrate_slic_segmentation():
    """Demonstrate SLIC segmentation"""
    print("=== SLIC Segmentation Demo ===")
    
    # Create sample image
    sample_img = create_sample_image()
    cv2.imwrite("sample_image.jpg", sample_img)
    
    # Initialize segmentation
    seg = ImageSegmentation()
    seg.load_image("sample_image.jpg")
    
    # Apply different SLIC parameters
    n_segments_list = [50, 100, 200]
    compactness_list = [5, 10, 20]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Original image
    for i in range(3):
        axes[i, 0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
    
    for i, n_seg in enumerate(n_segments_list):
        for j, comp in enumerate(compactness_list):
            result = seg.slic_segmentation(n_segments=n_seg, compactness=comp)
            axes[i, j + 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[i, j + 1].set_title(f'n_segments={n_seg}\ncompactness={comp}')
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('slic_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print("SLIC segmentation demo completed. Results saved to 'slic_demo.png'")

def run_all_demos():
    """Run all demonstration functions"""
    print("Starting Image Segmentation Demonstrations...")
    print("=" * 50)
    
    try:
        demonstrate_threshold_segmentation()
        print()
        
        demonstrate_kmeans_segmentation()
        print()
        
        demonstrate_watershed_segmentation()
        print()
        
        demonstrate_slic_segmentation()
        print()
        
        print("All demonstrations completed successfully!")
        print("Generated files:")
        print("- sample_image.jpg")
        print("- threshold_demo.png")
        print("- kmeans_demo.png")
        print("- watershed_demo.png")
        print("- slic_demo.png")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
    
    finally:
        # Clean up temporary files
        if os.path.exists("sample_image.jpg"):
            os.remove("sample_image.jpg")

if __name__ == "__main__":
    run_all_demos()
