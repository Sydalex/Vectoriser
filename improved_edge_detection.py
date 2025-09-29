#!/usr/bin/env python3
"""
Improved Edge Detection Module for Phase 4

This module provides enhanced edge detection methods based on 
comprehensive testing and optimization.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional


class ImprovedEdgeDetector:
    """Enhanced edge detection with multiple strategies and automatic optimization"""
    
    def __init__(self):
        """Initialize the improved edge detector"""
        self.detection_methods = {
            'canny_bilateral': self.canny_with_bilateral,
            'canny_anisotropic': self.canny_with_anisotropic,
            'canny_nlm': self.canny_with_nlm,
            'multiscale': self.multiscale_canny,
            'adaptive': self.adaptive_canny,
            'hybrid': self.hybrid_edge_detection
        }
    
    def detect_edges(self, image: np.ndarray, method: str = 'auto', 
                    config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Main edge detection interface with automatic method selection
        
        Args:
            image: Preprocessed grayscale image
            method: Detection method ('auto' for automatic selection)
            config: Optional configuration parameters
            
        Returns:
            Binary edge map
        """
        if config is None:
            config = {}
        
        if method == 'auto':
            # Automatically select best method based on image characteristics
            method = self._select_best_method(image)
            print(f"Auto-selected method: {method}")
        
        if method in self.detection_methods:
            return self.detection_methods[method](image, config)
        else:
            # Fallback to standard Canny
            return self.canny_with_bilateral(image, config)
    
    def _select_best_method(self, image: np.ndarray) -> str:
        """
        Automatically select the best edge detection method based on image characteristics
        
        Args:
            image: Input image
            
        Returns:
            Name of the recommended method
        """
        # Calculate image statistics
        mean_val = np.mean(image[image > 0])
        std_val = np.std(image[image > 0])
        noise_estimate = self._estimate_noise(image)
        
        # Simple heuristics for method selection
        if noise_estimate > 10:
            # High noise - use stronger filtering
            return 'canny_nlm'
        elif std_val < 20:
            # Low contrast - use adaptive thresholds
            return 'adaptive'
        elif mean_val > 150:
            # Bright image - use anisotropic diffusion
            return 'canny_anisotropic'
        else:
            # Default to bilateral filter
            return 'canny_bilateral'
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level in the image using Laplacian variance
        
        Args:
            image: Input image
            
        Returns:
            Estimated noise level
        """
        # Calculate Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        # Estimate noise as variance of Laplacian
        noise = np.var(laplacian)
        return np.sqrt(noise)
    
    def canny_with_bilateral(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Standard Canny edge detection with bilateral filtering
        
        Args:
            image: Preprocessed image
            config: Configuration parameters
            
        Returns:
            Binary edge map
        """
        # Get parameters
        canny_low = config.get('canny_low', 50)
        canny_high = config.get('canny_high', 150)
        bilateral_d = config.get('bilateral_d', 9)
        bilateral_sigma = config.get('bilateral_sigma', 75)
        
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(image, bilateral_d, bilateral_sigma, bilateral_sigma)
        
        # Apply Canny edge detection
        edges = cv2.Canny(smoothed, canny_low, canny_high)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Remove small isolated pixels
        edges = self._remove_small_components(edges, min_size=10)
        
        return edges
    
    def canny_with_anisotropic(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Canny edge detection with anisotropic diffusion filtering
        
        Args:
            image: Preprocessed image
            config: Configuration parameters
            
        Returns:
            Binary edge map
        """
        # Apply anisotropic diffusion
        filtered = self._anisotropic_diffusion(image, iterations=10)
        
        # Get Canny parameters
        canny_low = config.get('canny_low', 50)
        canny_high = config.get('canny_high', 150)
        
        # Apply Canny edge detection
        edges = cv2.Canny(filtered, canny_low, canny_high)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def canny_with_nlm(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Canny edge detection with Non-Local Means denoising
        
        Args:
            image: Preprocessed image
            config: Configuration parameters
            
        Returns:
            Binary edge map
        """
        # Apply Non-Local Means denoising
        denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Get Canny parameters
        canny_low = config.get('canny_low', 50)
        canny_high = config.get('canny_high', 150)
        
        # Apply Canny edge detection
        edges = cv2.Canny(denoised, canny_low, canny_high)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def multiscale_canny(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Multi-scale Canny edge detection
        
        Args:
            image: Preprocessed image
            config: Configuration parameters
            
        Returns:
            Binary edge map combining multiple scales
        """
        scales = config.get('scales', [1.0, 0.75, 0.5])
        canny_low = config.get('canny_low', 50)
        canny_high = config.get('canny_high', 150)
        
        h, w = image.shape[:2]
        combined_edges = np.zeros_like(image, dtype=np.float32)
        
        for scale in scales:
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Apply bilateral filter at this scale
            smoothed = cv2.bilateralFilter(resized, 9, 75, 75)
            
            # Detect edges
            edges = cv2.Canny(smoothed, canny_low, canny_high)
            
            # Resize back to original size
            edges_resized = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Combine with weight based on scale
            weight = scale
            combined_edges += edges_resized.astype(np.float32) * weight
        
        # Normalize and threshold
        combined_edges = (combined_edges / combined_edges.max() * 255).astype(np.uint8)
        _, binary_edges = cv2.threshold(combined_edges, 127, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)
        
        return binary_edges
    
    def adaptive_canny(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Canny edge detection with adaptive threshold calculation
        
        Args:
            image: Preprocessed image
            config: Configuration parameters
            
        Returns:
            Binary edge map with automatically determined thresholds
        """
        method = config.get('threshold_method', 'otsu')  # 'otsu', 'median', or 'gradient'
        
        # Apply smoothing first
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        if method == 'otsu':
            # Otsu's method for automatic threshold
            blur = cv2.GaussianBlur(smoothed, (5, 5), 0)
            otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = 0.5 * otsu_thresh
            canny_high = otsu_thresh
            
        elif method == 'median':
            # Median-based thresholding
            median_val = np.median(smoothed[smoothed > 0])
            sigma = 0.33
            canny_low = int(max(0, (1.0 - sigma) * median_val))
            canny_high = int(min(255, (1.0 + sigma) * median_val))
            
        else:  # gradient
            # Gradient magnitude-based thresholding
            grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Use percentiles of gradient magnitude
            non_zero_grad = grad_mag[grad_mag > 0]
            if len(non_zero_grad) > 0:
                canny_low = np.percentile(non_zero_grad, 70)
                canny_high = np.percentile(non_zero_grad, 90)
            else:
                canny_low, canny_high = 50, 150
        
        # Apply Canny with calculated thresholds
        edges = cv2.Canny(smoothed, canny_low, canny_high)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def hybrid_edge_detection(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Hybrid approach combining multiple edge detection methods
        
        Args:
            image: Preprocessed image
            config: Configuration parameters
            
        Returns:
            Binary edge map combining multiple methods
        """
        # Apply multiple methods
        edges_canny = self.canny_with_bilateral(image, config)
        
        # Sobel edges
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        threshold = np.percentile(sobel_mag, 85)
        edges_sobel = (sobel_mag > threshold).astype(np.uint8) * 255
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        threshold = np.percentile(morph_grad, 90)
        edges_morph = (morph_grad > threshold).astype(np.uint8) * 255
        
        # Combine edges using voting
        combined = (edges_canny.astype(np.float32) + 
                   edges_sobel.astype(np.float32) * 0.5 + 
                   edges_morph.astype(np.float32) * 0.5)
        
        # Threshold to create binary edge map
        combined = (combined > 127).astype(np.uint8) * 255
        
        # Final cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = self._remove_small_components(combined, min_size=15)
        
        return combined
    
    def _anisotropic_diffusion(self, img: np.ndarray, iterations: int = 10,
                               kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """
        Apply Perona-Malik anisotropic diffusion
        
        Args:
            img: Input image
            iterations: Number of diffusion iterations
            kappa: Conduction coefficient
            gamma: Speed of diffusion
            
        Returns:
            Filtered image
        """
        img = img.astype(np.float32)
        imgout = img.copy()
        
        for _ in range(iterations):
            # Calculate gradients
            deltaN = np.roll(imgout, -1, axis=0) - imgout
            deltaS = np.roll(imgout, 1, axis=0) - imgout
            deltaE = np.roll(imgout, -1, axis=1) - imgout
            deltaW = np.roll(imgout, 1, axis=1) - imgout
            
            # Calculate diffusion coefficients
            cN = np.exp(-(deltaN/kappa)**2)
            cS = np.exp(-(deltaS/kappa)**2)
            cE = np.exp(-(deltaE/kappa)**2)
            cW = np.exp(-(deltaW/kappa)**2)
            
            # Update image
            imgout += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
        
        return np.clip(imgout, 0, 255).astype(np.uint8)
    
    def _remove_small_components(self, edges: np.ndarray, min_size: int = 10) -> np.ndarray:
        """
        Remove small connected components from edge map
        
        Args:
            edges: Binary edge map
            min_size: Minimum component size to keep
            
        Returns:
            Cleaned edge map
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        # Create output image
        output = np.zeros_like(edges)
        
        # Keep only components larger than min_size
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = 255
        
        return output
    
    def eliminate_double_lines(self, edges: np.ndarray, distance_threshold: int = 5) -> np.ndarray:
        """
        Eliminate double lines by thinning and merging nearby edges
        
        Args:
            edges: Binary edge map
            distance_threshold: Maximum distance to consider edges as doubles
            
        Returns:
            Edge map with double lines removed
        """
        # Apply morphological thinning
        thinned = cv2.ximgproc.thinning(edges) if hasattr(cv2, 'ximgproc') else edges
        
        # Dilate slightly to merge very close lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.dilate(thinned, kernel, iterations=1)
        
        # Thin again to get single lines
        if hasattr(cv2, 'ximgproc'):
            final = cv2.ximgproc.thinning(merged)
        else:
            # Fallback to morphological skeleton
            final = self._morphological_skeleton(merged)
        
        return final
    
    def _morphological_skeleton(self, img: np.ndarray) -> np.ndarray:
        """
        Compute morphological skeleton as fallback for thinning
        
        Args:
            img: Binary image
            
        Returns:
            Skeletonized image
        """
        skeleton = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        temp = img.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = eroded - opened
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        return skeleton


def test_improved_detector():
    """Test the improved edge detector"""
    print("Testing Improved Edge Detector")
    print("=" * 50)
    
    # Create test image
    test_img = np.ones((200, 200), dtype=np.uint8) * 255
    cv2.rectangle(test_img, (50, 50), (150, 150), 128, -1)
    cv2.circle(test_img, (100, 100), 30, 64, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_img.shape)
    test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)
    
    # Initialize detector
    detector = ImprovedEdgeDetector()
    
    # Test different methods
    methods = ['canny_bilateral', 'adaptive', 'multiscale', 'hybrid']
    
    for method in methods:
        print(f"\nTesting method: {method}")
        edges = detector.detect_edges(test_img, method=method)
        edge_count = np.sum(edges == 255)
        print(f"  Edge pixels: {edge_count}")
        
        # Save result
        output_path = f"test_improved_{method}.png"
        cv2.imwrite(output_path, edges)
        print(f"  Saved to: {output_path}")
    
    # Test automatic method selection
    print("\nTesting automatic method selection:")
    edges = detector.detect_edges(test_img, method='auto')
    edge_count = np.sum(edges == 255)
    print(f"  Edge pixels: {edge_count}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_improved_detector()