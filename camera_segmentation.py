#!/usr/bin/env python3
"""
Camera-based Image Segmentation using OpenCV
Real-time segmentation from webcam feed
"""

import cv2
import numpy as np
from image_segmentation import ImageSegmentation
import argparse

class CameraSegmentation:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.seg = ImageSegmentation()
        self.current_method = 'threshold'
        self.params = {
            'threshold': 127,
            'kmeans_k': 3,
            'slic_segments': 100,
            'slic_compactness': 10
        }
        
    def start_camera(self):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        print(f"Camera {self.camera_id} started successfully")
        print("Controls:")
        print("  't' - Toggle threshold segmentation")
        print("  'k' - Toggle K-means segmentation")
        print("  's' - Toggle SLIC segmentation")
        print("  'w' - Toggle watershed segmentation")
        print("  'a' - Toggle adaptive threshold")
        print("  'f' - Toggle Felzenszwalb segmentation")
        print("  'g' - Toggle GrabCut segmentation")
        print("  '1-9' - Adjust threshold value (0-255)")
        print("  'q' - Quit")
        print("  'SPACE' - Save current frame")
        
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def apply_segmentation(self, frame):
        """Apply the current segmentation method to the frame"""
        try:
            # Create a temporary file for the frame
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Load the frame
            self.seg.load_image(temp_path)
            
            # Apply segmentation based on current method
            if self.current_method == 'threshold':
                result = self.seg.threshold_segmentation(threshold_value=self.params['threshold'])
            elif self.current_method == 'kmeans':
                result = self.seg.kmeans_segmentation(k=self.params['kmeans_k'])
            elif self.current_method == 'slic':
                result = self.seg.slic_segmentation(
                    n_segments=self.params['slic_segments'], 
                    compactness=self.params['slic_compactness']
                )
            elif self.current_method == 'watershed':
                result = self.seg.watershed_segmentation()
            elif self.current_method == 'adaptive':
                result = self.seg.adaptive_threshold_segmentation()
            elif self.current_method == 'felzenszwalb':
                result = self.seg.felzenszwalb_segmentation()
            elif self.current_method == 'grabcut':
                result = self.seg.grabcut_segmentation()
            else:
                result = frame
            
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return result
            
        except Exception as e:
            print(f"Segmentation error: {str(e)}")
            return frame
    
    def create_overlay(self, original, segmented):
        """Create an overlay of original and segmented images"""
        if len(segmented.shape) == 3:
            # Color image - create side-by-side
            h, w = original.shape[:2]
            overlay = np.zeros((h, w*2, 3), dtype=np.uint8)
            overlay[:, :w] = original
            overlay[:, w:] = segmented
        else:
            # Grayscale image - create overlay
            segmented_color = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(original, 0.7, segmented_color, 0.3, 0)
            
        return overlay
    
    def run(self):
        """Main camera loop"""
        try:
            self.start_camera()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Apply segmentation
                segmented = self.apply_segmentation(frame)
                
                # Create overlay
                overlay = self.create_overlay(frame, segmented)
                
                # Add text overlay
                cv2.putText(overlay, f"Method: {self.current_method}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.current_method == 'threshold':
                    cv2.putText(overlay, f"Threshold: {self.params['threshold']}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif self.current_method == 'kmeans':
                    cv2.putText(overlay, f"K: {self.params['kmeans_k']}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif self.current_method == 'slic':
                    cv2.putText(overlay, f"Segments: {self.params['slic_segments']}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the result
                cv2.imshow('Camera Segmentation', overlay)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.current_method = 'threshold'
                    print("Switched to Threshold segmentation")
                elif key == ord('k'):
                    self.current_method = 'kmeans'
                    print("Switched to K-means segmentation")
                elif key == ord('s'):
                    self.current_method = 'slic'
                    print("Switched to SLIC segmentation")
                elif key == ord('w'):
                    self.current_method = 'watershed'
                    print("Switched to Watershed segmentation")
                elif key == ord('a'):
                    self.current_method = 'adaptive'
                    print("Switched to Adaptive threshold")
                elif key == ord('f'):
                    self.current_method = 'felzenszwalb'
                    print("Switched to Felzenszwalb segmentation")
                elif key == ord('g'):
                    self.current_method = 'grabcut'
                    print("Switched to GrabCut segmentation")
                elif key == ord(' '):  # SPACE key
                    # Save current frame and segmentation
                    cv2.imwrite(f"camera_original_{self.current_method}.jpg", frame)
                    cv2.imwrite(f"camera_segmented_{self.current_method}.jpg", segmented)
                    print(f"Saved frame and segmentation as camera_original_{self.current_method}.jpg and camera_segmented_{self.current_method}.jpg")
                elif key in [ord(str(i)) for i in range(1, 10)]:
                    # Adjust threshold (1-9 keys)
                    if self.current_method == 'threshold':
                        self.params['threshold'] = int(key - ord('0')) * 28  # 1-9 -> 28-252
                        print(f"Threshold set to {self.params['threshold']}")
                    elif self.current_method == 'kmeans':
                        self.params['kmeans_k'] = int(key - ord('0')) + 1  # 1-9 -> 2-10
                        print(f"K-means K set to {self.params['kmeans_k']}")
                    elif self.current_method == 'slic':
                        self.params['slic_segments'] = int(key - ord('0')) * 50  # 1-9 -> 50-450
                        print(f"SLIC segments set to {self.params['slic_segments']}")
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.stop_camera()

def main():
    parser = argparse.ArgumentParser(description='Camera-based Image Segmentation')
    parser.add_argument('--camera', '-c', type=int, default=0, 
                       help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    try:
        camera_seg = CameraSegmentation(camera_id=args.camera)
        camera_seg.run()
    except Exception as e:
        print(f"Failed to start camera: {str(e)}")
        print("Make sure your camera is connected and not being used by another application.")

if __name__ == "__main__":
    main()
