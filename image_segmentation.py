import cv2
import numpy as np
from skimage import segmentation, color
from scipy import ndimage
import matplotlib.pyplot as plt

class ImageSegmentation:
    def __init__(self):
        self.original_image = None
        self.segmented_image = None
        
    def load_image(self, image_path):
        """Load image from path"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.original_image
    
    def threshold_segmentation(self, threshold_value=127, max_value=255):
        """Simple threshold-based segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, segmented = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
        
        self.segmented_image = segmented
        return segmented
    
    def adaptive_threshold_segmentation(self, block_size=11, c=2):
        """Adaptive threshold segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        segmented = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, block_size, c)
        
        self.segmented_image = segmented
        return segmented
    
    def watershed_segmentation(self, markers_count=10):
        """Watershed segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create markers
        markers = np.zeros(gray.shape, dtype=np.int32)
        for i, contour in enumerate(contours):
            cv2.drawContours(markers, [contour], -1, i + 1, -1)
        
        # Apply watershed
        segmented = cv2.watershed(self.original_image, markers)
        
        # Convert to visualization format
        segmented_viz = np.zeros_like(gray)
        segmented_viz[segmented > 0] = 255
        
        self.segmented_image = segmented_viz
        return segmented_viz
    
    def kmeans_segmentation(self, k=3):
        """K-means clustering segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Reshape image for clustering
        pixel_values = self.original_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(self.original_image.shape)
        
        self.segmented_image = segmented
        return segmented
    
    def felzenszwalb_segmentation(self, scale=100, sigma=0.5, min_size=50):
        """Felzenszwalb segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Apply Felzenszwalb segmentation
        segments = segmentation.felzenszwalb(rgb_image, scale=scale, sigma=sigma, min_size=min_size)
        
        # Create segmented image
        segmented = color.label2rgb(segments, rgb_image, kind='avg')
        
        # Convert back to BGR
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
        
        self.segmented_image = segmented_bgr
        return segmented_bgr
    
    def slic_segmentation(self, n_segments=100, compactness=10):
        """SLIC (Simple Linear Iterative Clustering) segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Apply SLIC segmentation
        segments = segmentation.slic(rgb_image, n_segments=n_segments, compactness=compactness)
        
        # Create segmented image
        segmented = color.label2rgb(segments, rgb_image, kind='avg')
        
        # Convert back to BGR
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
        
        self.segmented_image = segmented_bgr
        return segmented_bgr
    
    def grabcut_segmentation(self, rect=None):
        """GrabCut segmentation"""
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Create a copy of the image
        img = self.original_image.copy()
        
        # If no rectangle provided, use the center portion
        if rect is None:
            height, width = img.shape[:2]
            rect = (width//4, height//4, width//2, height//2)
        
        # Create mask
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # Create temporary arrays
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask for probable and definite foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        segmented = img * mask2[:, :, np.newaxis]
        
        self.segmented_image = segmented
        return segmented
    
    def save_segmented_image(self, output_path):
        """Save the segmented image"""
        if self.segmented_image is None:
            raise ValueError("No segmented image available. Please run segmentation first.")
        
        cv2.imwrite(output_path, self.segmented_image)
        print(f"Segmented image saved to: {output_path}")
    
    def display_results(self, figsize=(15, 5)):
        """Display original and segmented images"""
        if self.original_image is None or self.segmented_image is None:
            raise ValueError("Both original and segmented images are required.")
        
        plt.figure(figsize=figsize)
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Segmented image
        plt.subplot(1, 2, 2)
        if len(self.segmented_image.shape) == 3:
            plt.imshow(cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(self.segmented_image, cmap='gray')
        plt.title('Segmented Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_segmentation_stats(self):
        """Get statistics about the segmentation"""
        if self.segmented_image is None:
            raise ValueError("No segmented image available.")
        
        stats = {
            'original_shape': self.original_image.shape if self.original_image is not None else None,
            'segmented_shape': self.segmented_image.shape,
            'unique_values': len(np.unique(self.segmented_image)),
            'min_value': np.min(self.segmented_image),
            'max_value': np.max(self.segmented_image)
        }
        
        return stats
