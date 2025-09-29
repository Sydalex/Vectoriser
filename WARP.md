# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Vectoriser** is a Python computer vision project that converts raster images (PNG, JPEG, etc.) to DWG CAD files using semantic segmentation and contour vectorization. The pipeline isolates main subjects from backgrounds using deep learning models and vectorizes contours for CAD applications.

## Architecture

### Core Pipeline Components

1. **Image Loading & Validation** - Loads and validates input raster images
2. **Semantic Segmentation** - Uses DeepLabV3 with ResNet101 backbone for object isolation
3. **Image Preprocessing** - Applies Gaussian blur, CLAHE contrast enhancement, gamma correction, and unsharp masking
4. **Edge Detection** - Canny edge detection with bilateral filtering and morphological operations
5. **Contour Vectorization** - Ramer-Douglas-Peucker algorithm for contour simplification
6. **DWG Generation** - Creates layered CAD files with primary outlines and detail lines

### Key Classes

- **`ImageToDwgConverter`** - Main converter class encapsulating the entire pipeline
  - `convert()` - Primary entry point for end-to-end conversion
  - `_get_object_mask()` - Generates binary object masks (edge-based detection preferred over DeepLabV3)
  - `_preprocess_and_mask()` - Applies preprocessing and masking operations
  - `_detect_edges()` - Performs Canny edge detection with noise reduction
  - `_vectorize_contours()` - Extracts and simplifies contours for vectorization
  - `_create_dwg()` - Generates layered DWG files with primary outlines (red) and detail lines (white/black)

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Main Usage
```bash
# Basic conversion
python image_to_dwg.py --input input_image.png --output output_file.dwg

# Advanced usage with parameters
python image_to_dwg.py \
    --input path/to/image.png \
    --output path/to/output.dwg \
    --canny_low 75 \
    --canny_high 200 \
    --min_len 30 \
    --epsilon 2.0
```

### Testing and Development

```bash
# Run basic conversion test
python test_conversion.py

# Run comprehensive phase-by-phase testing
python test_phases.py

# Test Phase 2: Object mask generation with different methods
python test_phase2.py

# Test Phase 3: Image preprocessing variations
python test_phase3.py

# Test contrast enhancement variations
python test_contrast.py

# Test sharpening techniques
python test_sharpening.py
```

### Single Test Execution
```bash
# Test specific phase individually
python -c "from test_phases import test_phase_1_image_loading; test_phase_1_image_loading()"
```

### Parameter Tuning
```bash
# High detail settings (more sensitive)
python image_to_dwg.py --input image.png --output detailed.dwg --canny_low 30 --canny_high 90 --min_len 15 --epsilon 0.8

# Simplified output (less sensitive)
python image_to_dwg.py --input image.png --output simple.dwg --canny_low 80 --canny_high 200 --min_len 40 --epsilon 3.0
```

## AI/ML Integration

### ONNX Model Support
- Place ONNX models in the `/models` directory to enable AI features
- The `/models/.gitkeep` file documents this requirement
- Without ONNX models in `/models`, AI features will be disabled
- The current implementation uses PyTorch's pre-trained DeepLabV3 model, but is designed to support ONNX model inference

### Model Artifacts
- Partial UNet checkpoints (e.g., `epoch_0002.pt`) are intended to be saved in `ml/artifacts/` for testing
- ONNX models exported from these checkpoints should be placed in `/models` for production inference
- This allows testing the full PNG-to-vector pipeline without completing full model training

### Training and Inference Workflow
- Load partial checkpoints for ONNX export and testing
- Resume training later on GPU when full training is needed
- Use ONNX models for inference in the production pipeline

## Configuration Parameters

### Canny Edge Detection
- `--canny_low`: Lower threshold (30-100 range, default: 50)
- `--canny_high`: Upper threshold (100-300 range, default: 150, should be 2-3x lower threshold)

### Contour Processing
- `--min_len`: Minimum contour perimeter in pixels (15-50 range, default: 25)
- `--epsilon`: RDP simplification factor (0.5-5.0 range, default: 1.5)

## File Structure

```
Vectoriser/
├── image_to_dwg.py          # Main conversion script
├── test_conversion.py       # Basic conversion test
├── test_phases.py          # Comprehensive phase testing
├── test_phase2.py          # Object mask generation testing  
├── test_phase3.py          # Preprocessing variations testing
├── test_contrast.py        # Contrast enhancement testing
├── test_sharpening.py      # Sharpening technique testing
├── models/                 # ONNX models directory (required for AI features)
│   └── .gitkeep
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── venv/                  # Python virtual environment
```

## Dependencies

Core dependencies (see `requirements.txt`):
- **opencv-python>=4.5.0** - Image processing and computer vision
- **numpy>=1.21.0** - Numerical array operations
- **torch>=1.9.0** - Deep learning framework
- **torchvision>=0.10.0** - Pre-trained models (DeepLabV3)
- **ezdxf>=0.17.0** - DWG/DXF file creation and manipulation
- **Pillow>=8.0.0** - Image format support

## Performance Notes

- **GPU Acceleration**: Automatically uses CUDA if available for model inference
- **Memory Usage**: Large images require significant memory for processing
- **Processing Time**: Typically 10-60 seconds per image depending on size and complexity
- Use smaller images for faster processing during development

## Output Format

Generated DWG files contain two layers:
1. **PrimaryOutlines** (Red): Main object outline (largest contour)
2. **DetailLines** (White/Black): Internal details and smaller contours

## Coordinate Systems

- **Input**: Standard image coordinates (origin top-left)
- **Output**: CAD coordinates (origin bottom-left, Y-axis flipped)

## Supported Formats

**Input**: PNG, JPG/JPEG, BMP, TIFF, GIF (static), and other OpenCV-supported formats
**Output**: DWG (CAD format with layered vector graphics)

## Development Tips

- The project uses extensive phase-by-phase testing to validate each pipeline component
- Use the phase testing scripts to debug specific pipeline stages
- Parameter tuning is critical - start with default values and adjust based on image characteristics
- The edge-based object detection method often outperforms DeepLabV3 for many object types
- Consider preprocessing image contrast before conversion for better results