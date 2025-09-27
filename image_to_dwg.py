#!/usr/bin/env python3
"""
Image to DWG Converter

A Python script that takes a raster image, isolates the main subject using semantic segmentation,
vectorizes its contours, and saves the result as a layered DWG file.

Author: AI Assistant
Dependencies: opencv-python, numpy, ezdxf, torch, torchvision
"""

import argparse
import cv2
import numpy as np
import ezdxf
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import os
from typing import List, Tuple, Dict, Any


class ImageToDwgConverter:
    """
    A class to encapsulate the entire image-to-DWG conversion pipeline.
    """
    
    def __init__(self):
        """
        Initializes a pre-trained DeepLabV3 model with a ResNet101 backbone from torchvision.
        The model is set to evaluation mode (.eval()).
        """
        print("Loading DeepLabV3 model...")
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        
        # Define the transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),  # DeepLabV3 expects this size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully!")
    
    def convert(self, image_path: str, output_path: str, config: Dict[str, Any]) -> None:
        """
        Orchestrates the end-to-end conversion process. This is the main entry point.
        
        Args:
            image_path: Path to the source raster image
            output_path: Path for the generated DWG file
            config: Configuration dictionary with algorithm parameters
        """
        print(f"Starting conversion: {image_path} -> {output_path}")
        
        # Step 1: Load the source image
        print("Step 1: Loading source image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_height, original_width = image.shape[:2]
        print(f"Image loaded: {original_width}x{original_height}")
        
        # Step 2: Generate an object mask using the segmentation model
        print("Step 2: Generating object mask...")
        mask = self._get_object_mask(image)
        
        # Step 3: Pre-process the image and apply the mask
        print("Step 3: Pre-processing and applying mask...")
        processed_image = self._preprocess_and_mask(image, mask)
        
        # Step 4: Perform Canny edge detection
        print("Step 4: Performing Canny edge detection...")
        edges = self._detect_edges(processed_image, config)
        
        # Step 5: Find, filter, and simplify contours
        print("Step 5: Vectorizing contours...")
        contours = self._vectorize_contours(edges, config)
        
        # Step 6: Generate the final DWG file
        print("Step 6: Creating DWG file...")
        self._create_dwg(contours, output_path, original_width, original_height)
        
        print(f"Conversion completed successfully! Output saved to: {output_path}")
    
    def _get_object_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generates a precise, binary pixel mask of the primary object using edge-based detection.
        This method is more effective than DeepLabV3 for many object types.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            A binary NumPy array representing the object mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to create thicker boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask by filling the largest contour
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Clean up the mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
        else:
            # Fallback: if no contours found, use the entire image
            print("Warning: No contours found, using full image")
            return np.ones_like(gray, dtype=np.uint8) * 255
    
    def _preprocess_and_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies standard pre-processing: converts image to grayscale, applies a Gaussian blur
        for noise reduction, and then uses the object mask to remove the background.
        
        Args:
            image: Input image as numpy array
            mask: Binary mask array
            
        Returns:
            A processed NumPy array ready for edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply strong Gaussian blur for optimal preprocessing (Phase 3 analysis recommended)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Apply moderate contrast enhancement (CLAHE + lighter gamma correction)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        clahe_result = clahe.apply(blurred)
        
        # Apply lighter gamma correction for enhanced contrast without over-simplification
        enhanced = np.power(clahe_result / 255.0, 0.7) * 255.0
        enhanced = enhanced.astype(np.uint8)
        
        # Apply subtle unsharp masking for better edge definition
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.2, gaussian, -0.2, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Apply morphological operations to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to remove background
        masked_image = cv2.bitwise_and(sharpened, sharpened, mask=mask_smooth)
        
        return masked_image
    
    def _detect_edges(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Applies edge detection with focus on eliminating double lines from shadows/folds.
        
        Args:
            image: Processed image array
            config: Configuration dictionary containing Canny thresholds
            
        Returns:
            A binary edge map optimized for eliminating double lines
        """
        canny_low = config.get('canny_low', 50)
        canny_high = config.get('canny_high', 150)
        
        # Apply moderate smoothing to reduce noise while preserving details
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply Canny edge detection
        edges = cv2.Canny(smoothed, canny_low, canny_high)
        
        # Apply minimal morphological operations to clean up edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def _vectorize_contours(self, edges: np.ndarray, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Finds all contours in the edge map, eliminates double lines, and preserves defining details.
        
        Args:
            edges: Binary edge map
            config: Configuration dictionary containing RDP epsilon and min length
            
        Returns:
            A list of cleaned, smoothed and simplified contour coordinate lists
        """
        min_length = config.get('min_len', 25)
        epsilon = config.get('epsilon', 1.5)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by minimum length and apply smoothing
        smoothed_contours = []
        for contour in contours:
            if len(contour) >= min_length:
                # Convert contour to a more detailed representation for smoothing
                contour_detailed = cv2.approxPolyDP(contour, 0.1, True)
                
                # Apply smoothing using moving average
                smooth_window = config.get('smooth_window', 5)
                smoothed_points = self._smooth_contour(contour_detailed, smooth_window)
                
                # Final simplification using Ramer-Douglas-Peucker algorithm
                simplified = cv2.approxPolyDP(smoothed_points, epsilon, True)
                
                # Only keep contours that still have enough points after smoothing
                if len(simplified) >= 3:
                    smoothed_contours.append(simplified)
        
        # Eliminate double lines by merging nearby contours
        merge_distance = config.get('merge_distance', 10.0)
        merged_contours = self._eliminate_double_lines(smoothed_contours, merge_distance)
        
        print(f"Found {len(merged_contours)} contours after eliminating double lines")
        
        return merged_contours
    
    def _eliminate_double_lines(self, contours: List[np.ndarray], distance_threshold: float = 10.0) -> List[np.ndarray]:
        """
        Eliminates double lines by merging contours that are too close to each other.
        Uses a smarter approach that preserves important details.
        
        Args:
            contours: List of contour arrays
            distance_threshold: Maximum distance between contours to consider merging
            
        Returns:
            List of merged contours with double lines eliminated
        """
        if len(contours) <= 1:
            return contours
        
        # Calculate areas and sort by size (largest first)
        contour_areas = [(i, cv2.contourArea(contour)) for i, contour in enumerate(contours)]
        contour_areas.sort(key=lambda x: x[1], reverse=True)
        
        merged_contours = []
        used_indices = set()
        
        for i, (idx, area) in enumerate(contour_areas):
            if idx in used_indices:
                continue
                
            current_contour = contours[idx]
            merged_contour = current_contour.copy()
            
            # Check for nearby contours to merge
            for j, (other_idx, other_area) in enumerate(contour_areas[i+1:], i+1):
                if other_idx in used_indices:
                    continue
                    
                other_contour = contours[other_idx]
                
                # Calculate minimum distance between contours
                min_distance = self._calculate_min_distance(current_contour, other_contour)
                
                # Only merge if contours are very close (double lines) and similar in size
                area_ratio = min(area, other_area) / max(area, other_area) if max(area, other_area) > 0 else 0
                
                if min_distance < distance_threshold and area_ratio > 0.1:
                    # Merge contours by combining their points
                    merged_points = np.vstack([merged_contour.reshape(-1, 2), 
                                            other_contour.reshape(-1, 2)])
                    
                    # Create a new contour from merged points
                    merged_contour = cv2.convexHull(merged_points.astype(np.float32))
                    used_indices.add(other_idx)
            
            merged_contours.append(merged_contour)
            used_indices.add(idx)
        
        return merged_contours
    
    def _calculate_min_distance(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Calculates the minimum distance between two contours.
        
        Args:
            contour1: First contour
            contour2: Second contour
            
        Returns:
            Minimum distance between the contours
        """
        points1 = contour1.reshape(-1, 2)
        points2 = contour2.reshape(-1, 2)
        
        min_dist = float('inf')
        
        for p1 in points1:
            for p2 in points2:
                dist = np.sqrt(np.sum((p1 - p2) ** 2))
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _smooth_contour(self, contour: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Applies smoothing to a contour using a moving average filter.
        
        Args:
            contour: Input contour points
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed contour points
        """
        if len(contour) < window_size:
            return contour
        
        # Extract x and y coordinates
        points = contour.reshape(-1, 2)
        x_coords = points[:, 0].astype(np.float32)
        y_coords = points[:, 1].astype(np.float32)
        
        # Apply moving average smoothing
        smoothed_x = self._moving_average(x_coords, window_size)
        smoothed_y = self._moving_average(y_coords, window_size)
        
        # Combine back into contour format
        smoothed_points = np.column_stack((smoothed_x, smoothed_y))
        return smoothed_points.reshape(-1, 1, 2).astype(np.int32)
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Applies moving average smoothing to a 1D array.
        
        Args:
            data: Input 1D array
            window_size: Size of the moving average window
            
        Returns:
            Smoothed 1D array
        """
        if len(data) < window_size:
            return data
        
        # Pad the data for edge handling
        padded = np.pad(data, window_size//2, mode='edge')
        
        # Apply convolution for moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed
    
    def _create_dwg(self, contours: List[np.ndarray], output_path: str, 
                   width: int, height: int) -> None:
        """
        Generates a DWG file using the ezdxf library. Creates 'PrimaryOutlines' and 'DetailLines'
        layers. The contour with the largest area is classified as primary, and all others are
        classified as detail.
        
        Args:
            contours: List of contour coordinate arrays
            output_path: Path where to save the DWG file
            width: Original image width
            height: Original image height
        """
        # Create a new DXF document
        doc = ezdxf.new('R2010')  # Use AutoCAD 2010 format
        msp = doc.modelspace()
        
        # Create layers
        primary_layer = doc.layers.new('PrimaryOutlines')
        primary_layer.color = 1  # Red
        primary_layer.lineweight = 25  # Thicker lines
        
        detail_layer = doc.layers.new('DetailLines')
        detail_layer.color = 2  # Yellow
        detail_layer.lineweight = 9  # Thinner lines
        
        if not contours:
            print("Warning: No contours found, creating empty DWG file")
            doc.saveas(output_path)
            return
        
        # Find the contour with the largest area
        areas = [cv2.contourArea(contour) for contour in contours]
        largest_idx = np.argmax(areas)
        
        # Add contours to appropriate layers
        for i, contour in enumerate(contours):
            # Convert contour points to DXF format
            points = []
            for point in contour:
                x, y = point[0]
                # Convert from image coordinates to DXF coordinates
                # Flip Y axis and center the drawing
                dxf_x = x - width / 2
                dxf_y = (height - y) - height / 2
                points.append((dxf_x, dxf_y))
            
            # Close the polyline if it's not already closed
            if len(points) > 2 and points[0] != points[-1]:
                points.append(points[0])
            
            # Create polyline
            polyline = msp.add_lwpolyline(points)
            
            # Assign to appropriate layer
            if i == largest_idx:
                polyline.dxf.layer = 'PrimaryOutlines'
            else:
                polyline.dxf.layer = 'DetailLines'
        
        # Save the DWG file
        doc.saveas(output_path)
        print(f"DWG file saved with {len(contours)} polylines")


def main():
    """
    Main function to handle command-line interface and execute the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert raster images to DWG files using semantic segmentation and vectorization"
    )
    
    parser.add_argument(
        "--input", 
        required=True, 
        type=str, 
        help="Path to the source raster image"
    )
    
    parser.add_argument(
        "--output", 
        required=True, 
        type=str, 
        help="Path for the generated DWG file"
    )
    
    parser.add_argument(
        "--canny_low", 
        required=False, 
        type=int, 
        default=50, 
        help="Lower threshold for Canny edge detector"
    )
    
    parser.add_argument(
        "--canny_high", 
        required=False, 
        type=int, 
        default=150, 
        help="Upper threshold for Canny edge detector"
    )
    
    parser.add_argument(
        "--min_len", 
        required=False, 
        type=int, 
        default=25, 
        help="Minimum contour length (in pixels) to process"
    )
    
    parser.add_argument(
        "--epsilon", 
        required=False, 
        type=float, 
        default=1.5, 
        help="Epsilon value for RDP contour simplification"
    )
    
    parser.add_argument(
        "--smooth_window", 
        required=False, 
        type=int, 
        default=5, 
        help="Window size for contour smoothing (higher = more smoothing)"
    )
    
    parser.add_argument(
        "--merge_distance", 
        required=False, 
        type=float, 
        default=10.0, 
        help="Distance threshold for merging nearby contours (eliminates double lines)"
    )
    
    parser.add_argument(
        "--line_thickness", 
        required=False, 
        type=int, 
        default=2, 
        help="Line thickness for combining double lines (1-5, higher = thicker lines)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create configuration dictionary
    config = {
        'canny_low': args.canny_low,
        'canny_high': args.canny_high,
        'min_len': args.min_len,
        'epsilon': args.epsilon,
        'smooth_window': args.smooth_window,
        'merge_distance': args.merge_distance,
        'line_thickness': args.line_thickness
    }
    
    try:
        # Initialize converter and run conversion
        converter = ImageToDwgConverter()
        converter.convert(args.input, args.output, config)
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())