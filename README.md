# Image Segmentation Project with OpenCV

A comprehensive Python project demonstrating various image segmentation techniques using OpenCV and scikit-image.

## 🚀 Features

- **Multiple Segmentation Methods**: 7 different segmentation algorithms
- **Interactive Web Interface**: Streamlit-based web application
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Comprehensive Examples**: Ready-to-run demonstration scripts
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📋 Segmentation Methods

1. **Threshold Segmentation**: Simple binary thresholding
2. **Adaptive Threshold**: Adaptive thresholding for varying lighting conditions
3. **Watershed**: Watershed algorithm for object separation
4. **K-Means Clustering**: Color-based clustering segmentation
5. **Felzenszwalb**: Graph-based segmentation algorithm
6. **SLIC**: Simple Linear Iterative Clustering (superpixels)
7. **GrabCut**: Interactive foreground extraction

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Automatic Installation (Windows)

#### Option 1: Batch File (Recommended)
```bash
# Double-click the install.bat file or run:
install.bat
```

#### Option 2: PowerShell
```powershell
# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```

### Manual Installation
```bash
# Navigate to D drive
D:

# Create virtual environment
python -m venv "D:\Image Segmentation OpenCV\venv"

# Activate virtual environment
"D:\Image Segmentation OpenCV\venv\Scripts\activate.bat"

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Web Interface (Recommended for beginners)
```bash
# Activate virtual environment
"D:\Image Segmentation OpenCV\venv\Scripts\activate.bat"

# Run basic Streamlit app
streamlit run streamlit_app.py

# Run enhanced app with camera features
streamlit run streamlit_camera_app.py
```

The web interface will open in your browser at `http://localhost:8501`

### Command Line Interface
```bash
# Basic usage
python main.py image.jpg --method threshold

# With custom parameters
python main.py image.jpg --method kmeans --params 5

# Display results
python main.py image.jpg --method slic --params 200 20 --display

# Save to specific output
python main.py image.jpg --method watershed --output result.jpg
```

### Examples and Demonstrations
```bash
# Run all demonstrations
python examples.py

# Quick demo with sample image
python quick_start.py

# View generated demo images
# - threshold_demo.png
# - kmeans_demo.png
# - watershed_demo.png
# - slic_demo.png
# - quick_demo_results.png
```

## 📖 API Reference

### ImageSegmentation Class

```python
from image_segmentation import ImageSegmentation

# Initialize
seg = ImageSegmentation()

# Load image
seg.load_image("path/to/image.jpg")

# Apply segmentation
result = seg.threshold_segmentation(threshold_value=127)
result = seg.kmeans_segmentation(k=3)
result = seg.watershed_segmentation()
result = seg.slic_segmentation(n_segments=100, compactness=10)

# Save result
seg.save_segmented_image("output.jpg")

# Display results
seg.display_results()

# Get statistics
stats = seg.get_segmentation_stats()
```

## 🎯 Examples

### Threshold Segmentation
```python
seg = ImageSegmentation()
seg.load_image("image.jpg")
result = seg.threshold_segmentation(threshold_value=150)
seg.save_segmented_image("threshold_result.jpg")
```

### K-Means Clustering
```python
seg = ImageSegmentation()
seg.load_image("image.jpg")
result = seg.kmeans_segmentation(k=4)
seg.save_segmented_image("kmeans_result.jpg")
```

### SLIC Superpixels
```python
seg = ImageSegmentation()
seg.load_image("image.jpg")
result = seg.slic_segmentation(n_segments=200, compactness=15)
seg.save_segmented_image("slic_result.jpg")
```

### Camera-based Segmentation
```bash
# Real-time camera segmentation
python camera_segmentation.py

# With specific camera
python camera_segmentation.py --camera 1
```

## 📁 Project Structure

```
Image Segmentation OpenCV/
├── image_segmentation.py    # Core segmentation module
├── main.py                  # Command line interface
├── streamlit_app.py         # Basic web application
├── streamlit_camera_app.py  # Enhanced web app with camera features
├── camera_segmentation.py   # Real-time camera segmentation
├── examples.py              # Demonstration scripts
├── quick_start.py           # Quick demo script
├── requirements.txt         # Python dependencies
├── install.bat             # Windows batch installer
├── install.ps1             # PowerShell installer
└── README.md               # This file
```

## 🔧 Parameters Guide

### Threshold Segmentation
- `threshold_value`: Pixel value threshold (0-255)
- `max_value`: Maximum value for binary output

### Adaptive Threshold
- `block_size`: Size of pixel neighborhood (must be odd)
- `c`: Constant subtracted from mean

### Watershed
- `markers_count`: Number of marker regions

### K-Means
- `k`: Number of clusters/segments

### Felzenszwalb
- `scale`: Higher values mean larger segments
- `sigma`: Gaussian smoothing parameter
- `min_size`: Minimum component size

### SLIC
- `n_segments`: Target number of segments
- `compactness`: Balance between color and spatial proximity

## 🎨 Supported Image Formats

- **Input**: JPG, PNG, BMP, TIFF, JPEG
- **Output**: JPG, PNG (depending on method)

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure virtual environment is activated
2. **OpenCV Error**: Verify OpenCV installation with `pip list | findstr opencv`
3. **Memory Error**: Reduce image size or use smaller parameters
4. **Display Issues**: Use `--display` flag for matplotlib windows

### Getting Help
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test OpenCV
python -c "import cv2; print(cv2.__version__)"
```

## 📚 Learning Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-image Documentation](https://scikit-image.org/)
- [Image Segmentation Tutorials](https://opencv-python-tutroals.readthedocs.io/)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenCV community for the excellent computer vision library
- scikit-image team for advanced image processing algorithms
- Streamlit for the interactive web framework

---
