import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from image_segmentation import ImageSegmentation

def main():
    st.set_page_config(
        page_title="Image Segmentation with OpenCV",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Image Segmentation with OpenCV")
    st.markdown("Upload an image and apply various segmentation techniques")
    
    # Sidebar for controls
    st.sidebar.header("Segmentation Settings")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
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
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This application demonstrates various image segmentation techniques using OpenCV:
    
    - **Threshold**: Simple binary thresholding
    - **Adaptive Threshold**: Adaptive thresholding for varying lighting
    - **Watershed**: Watershed algorithm for object separation
    - **K-Means**: Color-based clustering
    - **Felzenszwalb**: Graph-based segmentation
    - **SLIC**: Superpixel segmentation
    - **GrabCut**: Interactive foreground extraction
    """)
    
    # Example usage
    if not uploaded_file:
        st.markdown("### Example Usage")
        st.code("""
# Command line usage:
python main.py image.jpg --method threshold --display
python main.py image.jpg --method kmeans --params 5
python main.py image.jpg --method slic --params 200 20
        """)

if __name__ == "__main__":
    main()
