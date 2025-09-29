# Advanced Character Illustration Vectoriser

A sophisticated system for converting detailed character illustrations (PNG images) into professional CAD-compatible vector formats (DWG) with intelligent masking, edge detection, and professional styling suitable for technical drawing applications.

![Vectoriser Preview](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.13+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Project Overview

Transform hand-drawn character illustrations into clean, scalable vector drawings while preserving anatomical details, edge quality, and artistic intent. Perfect for character design, technical documentation, animation preparation, and archival digitization.

## ✨ Key Features

### 🔍 Advanced Masking System
- **Depth-Texture Fusion**: Revolutionary algorithm combining depth awareness with texture analysis
- **Transparency Handling**: Sophisticated detection and removal of checkerboard patterns
- **Alpha Channel Support**: Proper RGBA compositing and constraint-based refinement
- **Anatomical Boundary Detection**: Specialized for character illustrations with body part boundaries

### 🎨 Professional CAD Styling
- **Multiple Industry Profiles**: Illustration, Architectural (AIA), Mechanical (ASME), Electrical (IEEE)
- **Intelligent Layer Classification**: Automatic assignment to appropriate CAD layers
- **Professional Line Weights**: Industry-standard line weights and colors
- **AutoCAD Compatibility**: DWG files compatible with AutoCAD 2000+ and similar software

### 🖥️ Real-Time Web Interface
- **Live Preview**: Multi-stage visualization (original → mask → edges → contours → DWG)
- **Parameter Controls**: Real-time adjustment of all processing parameters
- **Quality Metrics**: Instant feedback on masking quality and coverage statistics
- **Professional Workflow**: Upload → Process → Style → Preview → Download

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/vectoriser.git
cd vectoriser

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Start the web interface**:
   ```bash
   python web_preview.py
   ```

2. **Open your browser** to `http://localhost:3000`

3. **Upload a character illustration** (PNG recommended for best results)

4. **Process with standard masking** or use **Advanced Detailed Masking** for better results

5. **Adjust parameters** as needed and select appropriate CAD styling profile

6. **Download your professional DWG file**

## Usage

### Basic Usage

```bash
python image_to_dwg.py --input input_image.png --output output_file.dwg
```

### Advanced Usage with Parameters

```bash
python image_to_dwg.py \
    --input path/to/chair.png \
    --output path/to/chair.dwg \
    --canny_low 75 \
    --canny_high 200 \
    --min_len 30 \
    --epsilon 2.0
```

### Command-Line Arguments

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--input` | Yes | string | - | Path to the source raster image |
| `--output` | Yes | string | - | Path for the generated DWG file |
| `--canny_low` | No | int | 50 | Lower threshold for Canny edge detector |
| `--canny_high` | No | int | 150 | Upper threshold for Canny edge detector |
| `--min_len` | No | int | 25 | Minimum contour length (in pixels) to process |
| `--epsilon` | No | float | 1.5 | Epsilon value for RDP contour simplification |

### Parameter Tuning Guide

#### Canny Edge Detection Parameters
- **`--canny_low`**: Lower threshold for edge detection
  - Lower values = more edges detected (more sensitive)
  - Higher values = fewer edges detected (less sensitive)
  - Typical range: 30-100

- **`--canny_high`**: Upper threshold for edge detection
  - Should be 2-3x the lower threshold
  - Typical range: 100-300

#### Contour Processing Parameters
- **`--min_len`**: Minimum contour perimeter in pixels
  - Filters out small noise contours
  - Increase for cleaner output, decrease to capture more detail
  - Typical range: 15-50

- **`--epsilon`**: Contour simplification factor
  - Controls how much contours are simplified
  - Lower values = more detailed contours
  - Higher values = more simplified contours
  - Typical range: 0.5-5.0

## Output

The script generates a DWG file with two layers:

1. **PrimaryOutlines** (Red): Contains the contour with the largest area (main object outline)
2. **DetailLines** (White/Black): Contains all other detected contours (internal details)

## Workflow

1. **Image Loading**: Loads the source raster image
2. **Semantic Segmentation**: Uses DeepLabV3 to create an object mask
3. **Preprocessing**: Converts to grayscale, applies Gaussian blur, and masks background
4. **Edge Detection**: Applies Canny edge detection
5. **Contour Extraction**: Finds and filters contours by minimum length
6. **Vectorization**: Simplifies contours using Ramer-Douglas-Peucker algorithm
7. **DWG Generation**: Creates layered DWG file with vectorized contours

## Examples

### Example 1: Basic Conversion
```bash
python image_to_dwg.py --input furniture.jpg --output furniture.dwg
```

### Example 2: High Detail Settings
```bash
python image_to_dwg.py \
    --input detailed_object.png \
    --output detailed_object.dwg \
    --canny_low 30 \
    --canny_high 90 \
    --min_len 15 \
    --epsilon 0.8
```

### Example 3: Simplified Output
```bash
python image_to_dwg.py \
    --input complex_shape.jpg \
    --output simplified_shape.dwg \
    --canny_low 80 \
    --canny_high 200 \
    --min_len 40 \
    --epsilon 3.0
```

## Supported Image Formats

- PNG
- JPG/JPEG
- BMP
- TIFF
- GIF (static)
- And other formats supported by OpenCV

## Performance Notes

- **GPU Acceleration**: The script automatically uses CUDA if available for faster inference
- **Memory Usage**: Large images may require significant memory for processing
- **Processing Time**: Depends on image size and complexity (typically 10-60 seconds per image)

## Troubleshooting

### Common Issues

1. **"Could not load image"**: Check if the input file path is correct and the image format is supported
2. **CUDA out of memory**: Try with a smaller image or force CPU usage
3. **No contours found**: Adjust Canny thresholds or reduce minimum contour length
4. **Too many small contours**: Increase minimum contour length parameter

### Performance Optimization

- Use smaller images for faster processing
- Adjust parameters based on your specific use case
- Consider preprocessing images to enhance contrast before conversion

## Technical Details

### Model Information
- **Architecture**: DeepLabV3 with ResNet101 backbone
- **Training**: Pre-trained on COCO dataset
- **Classes**: 21 semantic classes (person, vehicle, furniture, etc.)

### Coordinate System
- Input images use standard image coordinates (origin top-left)
- Output DWG uses CAD coordinates (origin bottom-left, Y-axis flipped)

## License

This script is provided as-is for educational and research purposes.

## Version

Current version: 1.0.0 (2025-09-27)