import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from image_segmentation import ImageSegmentation

def main():
    st.set_page_config(
        page_title="Image Segmentation with OpenCV & Camera",
        page_icon="📷",
        layout="wide"
    )
    
    st.title("📷 Image Segmentation with OpenCV & Camera")
    st.markdown("Upload an image or capture from camera and apply various segmentation techniques")
    
    # Sidebar for controls
    st.sidebar.header("Input Method")
    input_method = st.sidebar.selectbox(
        "Choose input method",
        ["File Upload", "Camera Capture"]
    )
    
    if input_method == "File Upload":
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)
            
    else:  # Camera Capture
        st.sidebar.markdown("### Camera Settings")
        camera_source = st.sidebar.selectbox(
            "Camera Source",
            ["Default Camera (0)", "External Camera (1)", "IP Camera"]
        )
        
        if st.sidebar.button("📸 Start Camera"):
            st.markdown("### Camera Feed")
            st.info("Camera functionality is available in the desktop version. Use `python camera_segmentation.py` for real-time camera segmentation.")
            
            # Show camera controls
            st.markdown("""
            **Camera Controls (when running desktop version):**
            - **t** - Threshold segmentation
            - **k** - K-means segmentation  
            - **s** - SLIC segmentation
            - **w** - Watershed segmentation
            - **a** - Adaptive threshold
            - **f** - Felzenszwalb segmentation
            - **g** - GrabCut segmentation
            - **1-9** - Adjust parameters
            - **SPACE** - Save current frame
            - **q** - Quit
            """)
            
            # Demo with sample image
            st.markdown("### Demo with Sample Image")
            demo_img = create_demo_image()
            st.image(demo_img, caption="Sample Demo Image", use_column_width=True)
            
            if st.button("Apply Demo Segmentation"):
                process_demo_image(demo_img)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This application demonstrates various image segmentation techniques:
    
    - **Threshold**: Simple binary thresholding
    - **Adaptive Threshold**: Adaptive thresholding for varying lighting
    - **Watershed**: Watershed algorithm for object separation
    - **K-Means**: Color-based clustering
    - **Felzenszwalb**: Graph-based segmentation
    - **SLIC**: Superpixel segmentation
    - **GrabCut**: Interactive foreground extraction
    """)
    
    # Quick start section
    if input_method == "Camera Capture":
        st.markdown("### Quick Start with Camera")
        st.code("""
# Run camera segmentation (desktop)
python camera_segmentation.py

# With specific camera
python camera_segmentation.py --camera 1

# Run web interface
streamlit run streamlit_camera_app.py
        """)

def create_demo_image():
    """Create a demo image for camera demo"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add different colored regions
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)      # Blue
    cv2.circle(img, (250, 100), 50, (0, 255, 0), -1)               # Green
    cv2.rectangle(img, (100, 200), (200, 250), (0, 0, 255), -1)    # Red
    
    # Add text
    cv2.putText(img, "Camera Demo", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add noise
    noise = np.random.randint(0, 30, (300, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_demo_image(demo_img):
    """Process the demo image with segmentation"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demo Image")
        st.image(demo_img, use_column_width=True)
    
    with col2:
        st.subheader("Segmentation Results")
        
        # Apply different methods
        methods = ["Threshold", "K-Means", "SLIC"]
        
        for method in methods:
            if st.button(f"Apply {method}"):
                with st.spinner(f"Processing {method}..."):
                    try:
                        # Save demo image temporarily
                        temp_path = "temp_demo.jpg"
                        cv2.imwrite(temp_path, cv2.cvtColor(demo_img, cv2.COLOR_RGB2BGR))
                        
                        # Initialize segmentation
                        seg = ImageSegmentation()
                        seg.load_image(temp_path)
                        
                        # Apply segmentation
                        if method == "Threshold":
                            result = seg.threshold_segmentation(threshold_value=127)
                        elif method == "K-Means":
                            result = seg.kmeans_segmentation(k=3)
                        elif method == "SLIC":
                            result = seg.slic_segmentation(n_segments=100, compactness=10)
                        
                        # Display result
                        if len(result.shape) == 3:
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        else:
                            # Convert grayscale to RGB for Streamlit
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
                        
                        st.image(result_rgb, caption=f"{method} Result", use_column_width=True)
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        st.error(f"Error during {method} segmentation: {str(e)}")

def process_uploaded_file(uploaded_file):
    """Process uploaded file with segmentation"""
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # Convert to PIL Image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Convert to OpenCV format
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif len(image_array.shape) == 3:  # RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Segmentation method selection
    method = st.sidebar.selectbox(
        "Select Segmentation Method",
        ["Threshold", "Adaptive Threshold", "Watershed", "K-Means", "Felzenszwalb", "SLIC", "GrabCut"]
    )
    
    # Method-specific parameters
    params = {}
    
    if method == "Threshold":
        params['threshold'] = st.sidebar.slider("Threshold Value", 0, 255, 127)
        
    elif method == "Adaptive Threshold":
        params['block_size'] = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
        params['c'] = st.sidebar.slider("C Value", -10, 10, 2)
        
    elif method == "Watershed":
        params['markers'] = st.sidebar.slider("Number of Markers", 5, 50, 10)
        
    elif method == "K-Means":
        params['k'] = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
        
    elif method == "Felzenszwalb":
        params['scale'] = st.sidebar.slider("Scale", 10, 500, 100)
        params['sigma'] = st.sidebar.slider("Sigma", 0.1, 2.0, 0.5, step=0.1)
        params['min_size'] = st.sidebar.slider("Min Size", 10, 200, 50)
        
    elif method == "SLIC":
        params['n_segments'] = st.sidebar.slider("Number of Segments", 10, 500, 100)
        params['compactness'] = st.sidebar.slider("Compactness", 1, 50, 10)
    
    # Apply segmentation button
    if st.sidebar.button("Apply Segmentation"):
        with st.spinner("Processing image..."):
            try:
                # Initialize segmentation
                seg = ImageSegmentation()
                
                # Save uploaded image temporarily
                temp_path = "temp_upload.jpg"
                cv2.imwrite(temp_path, image_array)
                
                # Load image
                seg.load_image(temp_path)
                
                # Apply segmentation
                if method == "Threshold":
                    result = seg.threshold_segmentation(threshold_value=params['threshold'])
                elif method == "Adaptive Threshold":
                    result = seg.adaptive_threshold_segmentation(
                        block_size=params['block_size'], c=params['c']
                    )
                elif method == "Watershed":
                    result = seg.watershed_segmentation(markers_count=params['markers'])
                elif method == "K-Means":
                    result = seg.kmeans_segmentation(k=params['k'])
                elif method == "Felzenszwalb":
                    result = seg.felzenszwalb_segmentation(
                        scale=params['scale'], sigma=params['sigma'], min_size=params['min_size']
                    )
                elif method == "SLIC":
                    result = seg.slic_segmentation(
                        n_segments=params['n_segments'], compactness=params['compactness']
                    )
                elif method == "GrabCut":
                    result = seg.grabcut_segmentation()
                
                # Display result
                with col2:
                    st.subheader(f"Segmented Image ({method})")
                    
                    if len(result.shape) == 3:
                        # Color image
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_column_width=True)
                    else:
                        # Grayscale image - convert to RGB for Streamlit
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
                        st.image(result_rgb, use_column_width=True)
                    
                    # Statistics
                    stats = seg.get_segmentation_stats()
                    st.info(f"**Statistics:**\n"
                           f"- Original shape: {stats['original_shape']}\n"
                           f"- Segmented shape: {stats['segmented_shape']}\n"
                           f"- Unique values: {stats['unique_values']}")
                
                # Download button
                if st.button("Download Segmented Image"):
                    # Convert result to PIL Image for download
                    if len(result.shape) == 3:
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(result_rgb)
                    else:
                        pil_image = Image.fromarray(result)
                    
                    # Save to bytes
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    st.download_button(
                        label="Download Image",
                        data=img_byte_arr,
                        file_name=f"segmented_{method.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")

if __name__ == "__main__":
    main()
