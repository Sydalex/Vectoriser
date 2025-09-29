#!/usr/bin/env python3
"""
Detailed Illustration Masking System

Advanced masking specifically designed for detailed illustrations including:
- Edge-preserving smoothing
- Multi-scale edge detection
- Feature-aware segmentation
- Anatomical boundary detection
- Depth and texture awareness

Author: AI Assistant
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class DetailedMaskingSystem:
    """Advanced masking system for detailed illustrations"""
    
    def __init__(self):
        self.debug_mode = True
        
    def _preprocess_transparency(self, image: np.ndarray) -> np.ndarray:
        """
        Handle PNG transparency and remove checkerboard patterns
        """
        # Check if image has alpha channel
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA image - extract alpha channel
            alpha = image[:, :, 3]
            
            # Create a white background where transparent
            white_bg = np.ones_like(image[:, :, :3]) * 255
            
            # Blend image with white background based on alpha
            alpha_norm = alpha.astype(np.float32) / 255.0
            alpha_3d = np.repeat(alpha_norm[:, :, np.newaxis], 3, axis=2)
            
            # Blend: result = foreground * alpha + background * (1 - alpha)
            blended = (image[:, :, :3].astype(np.float32) * alpha_3d + 
                      white_bg.astype(np.float32) * (1 - alpha_3d))
            
            processed_image = blended.astype(np.uint8)
            
            # Return the alpha mask for later use
            return processed_image, alpha
        
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image - check for checkerboard pattern (common transparency indicator)
            processed_image = self._remove_checkerboard_pattern(image)
            
            # If regular removal didn't work well, try aggressive removal
            # Check if there's still a lot of gray pattern
            gray_test = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            gray_pattern = cv2.inRange(processed_image, (150, 150, 150), (240, 240, 240))
            gray_ratio = np.sum(gray_pattern > 0) / (processed_image.shape[0] * processed_image.shape[1])
            
            if gray_ratio > 0.2:  # Still significant gray pattern
                processed_image = self._aggressive_checkerboard_removal(processed_image)
                
                # Final check - if there's still gray pattern, apply final cleanup
                final_gray_test = cv2.inRange(processed_image, (170, 170, 170), (230, 230, 230))
                final_gray_ratio = np.sum(final_gray_test > 0) / (processed_image.shape[0] * processed_image.shape[1])
                
                if final_gray_ratio > 0.15:
                    print(f"Final cleanup - removing remaining patterns (coverage: {final_gray_ratio*100:.1f}%)")
                    # Very aggressive final cleanup
                    processed_image = self._final_pattern_cleanup(processed_image)
            
            return processed_image, None
        
        else:
            # Grayscale image
            return image, None
    
    def _remove_checkerboard_pattern(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and remove checkerboard transparency patterns
        """
        # Convert to grayscale for pattern detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # More comprehensive checkerboard detection
        # Common transparency checkerboard colors
        patterns = [
            # Light gray and white pattern
            (cv2.inRange(image, (200, 200, 200), (255, 255, 255)), 
             cv2.inRange(image, (170, 170, 170), (199, 199, 199))),
            # Medium gray pattern  
            (cv2.inRange(image, (180, 180, 180), (220, 220, 220)),
             cv2.inRange(image, (150, 150, 150), (179, 179, 179))),
            # Common Photoshop transparency colors
            (cv2.inRange(image, (192, 192, 192), (255, 255, 255)),
             cv2.inRange(image, (128, 128, 128), (191, 191, 191))),
        ]
        
        best_checkerboard = None
        best_ratio = 0
        
        for light_mask, dark_mask in patterns:
            checkerboard_pixels = cv2.bitwise_or(light_mask, dark_mask)
            checkerboard_ratio = np.sum(checkerboard_pixels > 0) / (image.shape[0] * image.shape[1])
            
            if checkerboard_ratio > best_ratio:
                best_ratio = checkerboard_ratio
                best_checkerboard = checkerboard_pixels
        
        # Lower threshold for detection - even small checkerboard patterns should be removed
        if best_ratio > 0.15:  # If more than 15% is checkerboard pattern
            print(f"Detected checkerboard transparency pattern - removing... (coverage: {best_ratio*100:.1f}%)")
            
            # Additional step: detect actual content (non-transparent areas)
            # Look for areas with significant color variation or edges
            edges = cv2.Canny(gray, 30, 100)
            
            # Dilate edges to capture nearby content
            edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            content_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, edge_kernel)
            
            # Also look for areas that are not grayscale (have color)
            color_variance = np.std(image, axis=2)
            color_areas = (color_variance > 5).astype(np.uint8) * 255
            
            # Combine edge and color information to identify actual content
            actual_content = cv2.bitwise_or(content_edges, color_areas)
            
            # Dilate the actual content to include nearby pixels
            content_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            actual_content = cv2.morphologyEx(actual_content, cv2.MORPH_DILATE, content_kernel)
            
            # Create the final mask: remove checkerboard but keep actual content
            final_mask = cv2.bitwise_or(actual_content, cv2.bitwise_not(best_checkerboard))
            
            # Apply the mask
            result = image.copy()
            # Replace checkerboard areas that are NOT actual content with white
            replacement_mask = cv2.bitwise_and(best_checkerboard, cv2.bitwise_not(actual_content))
            result[replacement_mask > 0] = [255, 255, 255]
            
            return result
        
        return image
    
    def _aggressive_checkerboard_removal(self, image: np.ndarray) -> np.ndarray:
        """
        More aggressive checkerboard pattern removal for stubborn cases
        """
        print("Applying aggressive checkerboard removal...")
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect low saturation areas (typical of transparency patterns)
        low_saturation = hsv[:, :, 1] < 30
        
        # Detect areas with similar lightness values (checkerboard characteristic)
        lightness = lab[:, :, 0]
        light_areas = (lightness > 180) & (lightness < 255)
        medium_areas = (lightness > 120) & (lightness < 180)
        
        # Combine indicators
        potential_pattern = low_saturation & (light_areas | medium_areas)
        pattern_mask = potential_pattern.astype(np.uint8) * 255
        
        # Morphological operations to clean up the pattern detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pattern_mask = cv2.morphologyEx(pattern_mask, cv2.MORPH_CLOSE, kernel)
        
        # If significant pattern detected, remove it
        pattern_ratio = np.sum(pattern_mask > 0) / (image.shape[0] * image.shape[1])
        if pattern_ratio > 0.1:
            print(f"Removing checkerboard pattern (coverage: {pattern_ratio*100:.1f}%)")
            
            # Identify actual content by looking for edges and color variation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 60)
            
            # Dilate edges to preserve content
            edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            content_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, edge_kernel)
            
            # Preserve areas with significant color
            color_std = np.std(image.reshape(-1, 3), axis=1).reshape(image.shape[:2])
            color_content = (color_std > 10).astype(np.uint8) * 255
            
            # Combine content indicators
            preserve_mask = cv2.bitwise_or(content_edges, color_content)
            
            # Apply removal
            result = image.copy()
            removal_mask = cv2.bitwise_and(pattern_mask, cv2.bitwise_not(preserve_mask))
            result[removal_mask > 0] = [255, 255, 255]
            
            return result
        
        return image
    
    def _final_pattern_cleanup(self, image: np.ndarray) -> np.ndarray:
        """
        Final aggressive cleanup for any remaining background patterns
        """
        # Very aggressive approach - remove all grayish areas that don't have strong edges or colors
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find strong edges (actual content)
        strong_edges = cv2.Canny(gray, 50, 150)
        
        # Dilate strong edges significantly
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        content_areas = cv2.morphologyEx(strong_edges, cv2.MORPH_DILATE, large_kernel)
        
        # Find areas with significant color variation
        color_std = np.std(image, axis=2)
        colorful_areas = (color_std > 15).astype(np.uint8) * 255
        
        # Find very dark or very light areas (likely actual content)
        dark_areas = (gray < 50).astype(np.uint8) * 255
        bright_but_not_gray = ((gray > 240) & (color_std > 3)).astype(np.uint8) * 255
        
        # Combine all content indicators
        preserve_mask = cv2.bitwise_or(content_areas, colorful_areas)
        preserve_mask = cv2.bitwise_or(preserve_mask, dark_areas)
        preserve_mask = cv2.bitwise_or(preserve_mask, bright_but_not_gray)
        
        # Apply final cleanup - replace everything else with white
        result = image.copy()
        result[preserve_mask == 0] = [255, 255, 255]
        
        return result

    def object_cohesion_refinement(self, image: np.ndarray, mask: np.ndarray, iterations: int = 4,
                                   color_thr: float = 14.0, edge_thr: float = 20.0) -> np.ndarray:
        """
        Expand/refine the object mask to keep cohesive parts that are near in depth but may have been
        cut off by depth weighting. Uses color similarity in LAB space, edge-aware constraints, and
        iterative connectivity-based growth from the largest connected component.
        """
        if mask is None or image is None:
            return mask
        
        # Ensure mask is binary uint8 {0,255}
        m = (mask > 0).astype(np.uint8) * 255
        if np.sum(m) == 0:
            return m
        
        # Find largest connected component as the object core
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if num_labels <= 1:
            core = m.copy()
        else:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            core = ((labels == largest_label).astype(np.uint8)) * 255
        
        # Prepare color space and edges
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        
        L, A, B = lab[:, :, 0].astype(np.float32), lab[:, :, 1].astype(np.float32), lab[:, :, 2].astype(np.float32)
        core_inds = core > 0
        if np.sum(core_inds) == 0:
            return m
        Lm, Am, Bm = np.mean(L[core_inds]), np.mean(A[core_inds]), np.mean(B[core_inds])
        
        # Precompute color distance map (LAB Euclidean)
        color_dist = np.sqrt((L - Lm) ** 2 + (A - Am) ** 2 + (B - Bm) ** 2)
        
        # Edge map to avoid crossing strong boundaries
        edges = cv2.Canny(gray, 80, 200)
        edges_blur = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
        
        current = core.copy()
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = float(color_thr)
        for _ in range(max(1, iterations)):
            # Candidate ring: immediate neighbors
            dil = cv2.dilate(current, se)
            ring = cv2.bitwise_and(dil, cv2.bitwise_not(current))
            ring_bool = ring > 0
            if not np.any(ring_bool):
                break
            
            # Accept candidates similar in color and not across strong edges
            accept = (color_dist <= thr) & (edges_blur <= edge_thr) & ring_bool
            if not np.any(accept):
                # relax threshold slightly
                thr += 2.0
                continue
            
            current[accept] = 255
            # Gradually relax
            thr = min(thr + 1.0, color_thr + 6.0)
        
        # Fill small holes and smooth
        filled = ndimage.binary_fill_holes(current > 0).astype(np.uint8) * 255
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close)
        
        return refined
        
    def bilateral_edge_preserving_mask(self, image: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Edge-preserving mask that maintains fine details while smoothing regions
        Perfect for illustrations with fine lines and details
        """
        # Preprocess transparency
        processed_image, alpha = self._preprocess_transparency(image)
        
        # Convert to grayscale for processing
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_image.copy()
        
        # Apply bilateral filtering to preserve edges while smoothing
        filtered = gray.copy()
        for _ in range(iterations):
            filtered = cv2.bilateralFilter(filtered, 9, 80, 80)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 30, 90)  # Fine details
        edges_medium = cv2.Canny(filtered, 50, 150)  # Medium features
        edges_coarse = cv2.Canny(cv2.GaussianBlur(filtered, (7, 7), 0), 70, 200)  # Major boundaries
        
        # Combine multi-scale edges with different weights
        combined_edges = np.zeros_like(gray, dtype=np.float32)
        combined_edges += edges_fine.astype(np.float32) * 0.5  # Fine details
        combined_edges += edges_medium.astype(np.float32) * 0.8  # Medium features  
        combined_edges += edges_coarse.astype(np.float32) * 1.0  # Major boundaries
        
        # Normalize and convert back to uint8
        combined_edges = np.clip(combined_edges, 0, 255).astype(np.uint8)
        
        # Morphological operations to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and create mask from largest regions
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._fallback_threshold_mask(image)
        
        # Sort contours by area and keep significant ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Create mask by filling significant contours
        mask = np.zeros_like(gray)
        total_area = gray.shape[0] * gray.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Keep contours that are at least 1% of image area
            if area > total_area * 0.01:
                cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def texture_aware_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Texture-aware masking that differentiates between different surface types
        Good for detecting clothing vs skin, hair textures, etc.
        """
        # Preprocess transparency
        processed_image, alpha = self._preprocess_transparency(image)
        
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_image.copy()
        
        # Calculate local texture features
        # 1. Local standard deviation (texture measure)
        kernel = np.ones((7, 7), np.float32) / 49
        gray_f32 = gray.astype(np.float32)
        local_mean = cv2.filter2D(gray_f32, -1, kernel)
        local_sqr_mean = cv2.filter2D(gray_f32**2, -1, kernel)
        local_std = np.sqrt(np.abs(local_sqr_mean - local_mean**2))
        
        # 2. Gabor filter responses for texture
        gabor_responses = []
        for theta in [0, 45, 90, 135]:  # Different orientations
            for frequency in [0.1, 0.3]:  # Different frequencies
                real, _ = cv2.getGaborKernel((15, 15), 3, np.radians(theta), 2*np.pi*frequency, 0.5, 0)
                filtered = cv2.filter2D(gray_f32, cv2.CV_8UC3, real)
                gabor_responses.append(filtered)
        
        # Combine Gabor responses
        gabor_combined = np.zeros_like(gray_f32)
        for response in gabor_responses:
            gabor_combined += np.abs(response)
        gabor_combined = gabor_combined / len(gabor_responses)
        
        # 3. Edge density (how many edges in local neighborhood)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
        
        # Combine texture features
        texture_feature = local_std * 0.4 + gabor_combined * 0.4 + edge_density * 0.2
        
        # Normalize texture feature
        texture_feature = cv2.normalize(texture_feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply adaptive thresholding based on texture
        mask = cv2.adaptiveThreshold(texture_feature, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        
        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        # If we have an alpha channel, use it to refine the mask
        if alpha is not None:
            alpha_mask = (alpha > 64).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, alpha_mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def depth_aware_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Depth-aware masking that tries to separate foreground from background
        and different depth layers (like hair vs face vs clothing)
        """
        # Preprocess transparency
        processed_image, alpha = self._preprocess_transparency(image)
        
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        else:
            gray = processed_image.copy()
            lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        
        # 1. Brightness-based depth cues
        # Typically, closer objects are better lit
        brightness = lab[:, :, 0]  # L channel
        
        # 2. Contrast-based depth cues
        # Objects in focus have higher contrast
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        contrast = np.abs(laplacian)
        contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Color saturation cues
        # Foreground objects often have higher saturation
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV) if len(processed_image.shape) == 3 else cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # 4. Edge strength cues
        # Sharp edges indicate closer objects
        edges = cv2.Canny(gray, 100, 200)
        edge_strength = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
        
        # Combine depth cues
        depth_score = (brightness.astype(np.float32) * 0.25 + 
                      contrast.astype(np.float32) * 0.35 + 
                      saturation.astype(np.float32) * 0.2 + 
                      edge_strength * 0.2)
        
        # Normalize
        depth_score = cv2.normalize(depth_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Use Otsu's method to find optimal threshold for foreground/background
        _, mask = cv2.threshold(depth_score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If we have an alpha channel, use it to constrain the mask
        if alpha is not None:
            alpha_mask = (alpha > 64).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, alpha_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def anatomical_boundary_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Specialized for detecting anatomical boundaries in character illustrations
        Focuses on detecting boundaries between body parts, clothing, etc.
        """
        # Preprocess transparency
        processed_image, alpha = self._preprocess_transparency(image)
        
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_image.copy()
        
        # 1. Gradient-based boundary detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. Structure tensor for detecting oriented boundaries
        Ixx = cv2.GaussianBlur(grad_x * grad_x, (3, 3), 0)
        Ixy = cv2.GaussianBlur(grad_x * grad_y, (3, 3), 0)
        Iyy = cv2.GaussianBlur(grad_y * grad_y, (3, 3), 0)
        
        # Calculate coherence (how well-oriented the local structure is)
        determinant = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy
        coherence = np.zeros_like(determinant)
        mask = trace > 0
        coherence[mask] = (trace[mask] - 2*np.sqrt(determinant[mask])) / (trace[mask] + 1e-10)
        coherence = cv2.normalize(coherence, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Multi-scale boundary detection
        boundaries = []
        for sigma in [1.0, 2.0, 4.0]:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            edge = cv2.Canny(blurred, 50, 150)
            boundaries.append(edge)
        
        # Combine multi-scale boundaries
        combined_boundaries = np.zeros_like(gray, dtype=np.float32)
        for i, boundary in enumerate(boundaries):
            weight = 1.0 / (i + 1)  # Give more weight to finer scales
            combined_boundaries += boundary.astype(np.float32) * weight
        
        combined_boundaries = cv2.normalize(combined_boundaries, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 4. Combine all boundary information
        final_boundaries = np.maximum(gradient_magnitude, 
                          np.maximum(coherence, combined_boundaries))
        
        # 5. Morphological operations to connect boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_boundaries = cv2.morphologyEx(final_boundaries, cv2.MORPH_CLOSE, kernel)
        
        # 6. Find contours and create mask
        contours, _ = cv2.findContours(final_boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._fallback_threshold_mask(image)
        
        # Create mask from significant contours
        mask = np.zeros_like(gray)
        total_area = gray.shape[0] * gray.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > total_area * 0.005:  # Even smaller threshold for detailed work
                cv2.fillPoly(mask, [contour], 255)
        
        # If we have an alpha channel, use it to constrain the mask
        if alpha is not None:
            alpha_mask = (alpha > 64).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, alpha_mask)
        
        return mask
    
    def depth_texture_fusion_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced fusion of depth-aware and texture-aware masking
        This combines the best of both approaches for detailed character illustrations
        """
        # Preprocess transparency
        processed_image, alpha = self._preprocess_transparency(image)
        
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        else:
            gray = processed_image.copy()
            lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        
        # === DEPTH AWARENESS COMPONENTS ===
        
        # 1. Brightness-based depth (closer objects are often better lit)
        brightness = lab[:, :, 0].astype(np.float32)
        
        # 2. Focus/sharpness measure (closer objects are sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.abs(laplacian)
        
        # 3. Edge strength (sharp boundaries indicate foreground)
        edges = cv2.Canny(gray, 50, 150)
        edge_strength = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
        
        # 4. Color saturation (foreground often more saturated)
        saturation = hsv[:, :, 1].astype(np.float32)
        
        # === TEXTURE AWARENESS COMPONENTS ===
        
        # 1. Local standard deviation (texture measure)
        kernel = np.ones((7, 7), np.float32) / 49
        gray_f32 = gray.astype(np.float32)
        local_mean = cv2.filter2D(gray_f32, -1, kernel)
        local_sqr_mean = cv2.filter2D(gray_f32**2, -1, kernel)
        texture_variance = np.sqrt(np.abs(local_sqr_mean - local_mean**2))
        
        # 2. Multi-orientation Gabor filters for texture detection
        gabor_energy = np.zeros_like(gray_f32)
        for theta in [0, 30, 60, 90, 120, 150]:  # More orientations for better texture capture
            for frequency in [0.05, 0.15, 0.25]:  # Multiple frequencies
                gabor_kernel = cv2.getGaborKernel((15, 15), 4, np.radians(theta), 
                                                2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray_f32, cv2.CV_8UC3, gabor_kernel)
                gabor_energy += np.abs(filtered)
        
        gabor_energy = gabor_energy / (6 * 3)  # Normalize by number of filters
        
        # 3. Local Binary Pattern-like texture measure
        # Calculate intensity differences in 8 directions
        texture_consistency = np.zeros_like(gray_f32)
        directions = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        
        for dy, dx in directions:
            shifted = np.roll(np.roll(gray_f32, dy, axis=0), dx, axis=1)
            texture_consistency += np.abs(gray_f32 - shifted)
        
        texture_consistency = texture_consistency / 8
        
        # === FUSION OF DEPTH AND TEXTURE FEATURES ===
        
        # Normalize all features to 0-1 range
        def normalize_feature(feature):
            return cv2.normalize(feature, None, 0, 1, cv2.NORM_MINMAX)
        
        brightness_norm = normalize_feature(brightness)
        sharpness_norm = normalize_feature(sharpness)
        edge_strength_norm = normalize_feature(edge_strength)
        saturation_norm = normalize_feature(saturation)
        texture_variance_norm = normalize_feature(texture_variance)
        gabor_energy_norm = normalize_feature(gabor_energy)
        texture_consistency_norm = normalize_feature(texture_consistency)
        
        # Create composite depth score
        depth_score = (brightness_norm * 0.3 +        # Lighting
                      sharpness_norm * 0.4 +         # Focus/sharpness
                      edge_strength_norm * 0.2 +     # Edge strength
                      saturation_norm * 0.1)         # Color saturation
        
        # Create composite texture score
        texture_score = (texture_variance_norm * 0.4 +     # Local texture variation
                        gabor_energy_norm * 0.4 +          # Multi-scale texture
                        texture_consistency_norm * 0.2)     # Local consistency
        
        # === INTELLIGENT FUSION STRATEGY ===
        
        # Calculate local confidence for each method
        depth_confidence = (edge_strength_norm + sharpness_norm) / 2
        texture_confidence = (texture_variance_norm + gabor_energy_norm) / 2
        
        # Create adaptive weights based on local image characteristics
        total_confidence = depth_confidence + texture_confidence + 1e-10  # Avoid division by zero
        depth_weight = depth_confidence / total_confidence
        texture_weight = texture_confidence / total_confidence
        
        # Fused score with adaptive weighting
        fused_score = depth_score * depth_weight + texture_score * texture_weight
        
        # === POST-PROCESSING ===
        
        # Apply bilateral filtering to smooth while preserving edges
        fused_score_smooth = cv2.bilateralFilter(fused_score.astype(np.float32), 9, 0.1, 10)
        
        # Convert to uint8 for thresholding
        fused_score_uint8 = (fused_score_smooth * 255).astype(np.uint8)
        
        # Use adaptive thresholding to handle varying lighting conditions
        mask_adaptive = cv2.adaptiveThreshold(fused_score_uint8, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 15, 2)
        
        # Also try Otsu's method for comparison
        _, mask_otsu = cv2.threshold(fused_score_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine both thresholding approaches
        mask_combined = cv2.bitwise_or(mask_adaptive, mask_otsu)
        
        # Morphological operations to clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_open)
        
        # If we have an alpha channel, use it to constrain the mask
        if alpha is not None:
            print("Using alpha channel to refine mask...")
            # Create alpha mask (exclude fully transparent areas)
            alpha_mask = (alpha > 64).astype(np.uint8) * 255  # More lenient threshold for semi-transparency
            
            # Apply alpha constraint to the combined mask
            mask_combined = cv2.bitwise_and(mask_combined, alpha_mask)
        
        # Find largest connected component (assume it's the main subject)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_combined)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            final_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            final_mask = mask_combined
        
        # Final object-cohesion refinement to keep near-depth cohesive parts
        try:
            final_mask = self.object_cohesion_refinement(processed_image, final_mask, iterations=5,
                                                         color_thr=14.0, edge_thr=22.0)
        except Exception as _:
            pass
        
        return final_mask
    
    def mean_shift_segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Mean shift segmentation for region-based masking
        Good for separating regions with similar colors/textures
        """
        # Prepare image for mean shift
        if len(image.shape) == 3:
            # Convert to feature space (position + color)
            h, w, c = image.shape
            features = np.zeros((h * w, c + 2))  # color + spatial coordinates
            
            # Add spatial coordinates
            y, x = np.mgrid[0:h, 0:w]
            features[:, 0] = x.flatten() / w  # Normalize spatial coordinates
            features[:, 1] = y.flatten() / h
            
            # Add color features
            for i in range(c):
                features[:, i + 2] = image[:, :, i].flatten() / 255.0
        else:
            h, w = image.shape
            features = np.zeros((h * w, 3))  # grayscale + spatial coordinates
            
            y, x = np.mgrid[0:h, 0:w]
            features[:, 0] = x.flatten() / w
            features[:, 1] = y.flatten() / h
            features[:, 2] = image.flatten() / 255.0
        
        # Apply mean shift clustering
        try:
            ms = MeanShift(bandwidth=0.1, bin_seeding=True)
            ms.fit(features)
            labels = ms.labels_
            
            # Reshape labels back to image shape
            segmented = labels.reshape(h, w)
            
            # Find the most central cluster (assume it's the main object)
            center_y, center_x = h // 2, w // 2
            center_label = segmented[center_y, center_x]
            
            # Create mask from the central cluster and nearby clusters
            mask = np.zeros_like(segmented, dtype=np.uint8)
            
            # Include the center cluster
            mask[segmented == center_label] = 255
            
            # Also include clusters that are spatially connected to the center
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
            
        except Exception as e:
            print(f"Mean shift segmentation failed: {e}")
            return self._fallback_threshold_mask(image)
    
    def _fallback_threshold_mask(self, image: np.ndarray) -> np.ndarray:
        """Simple fallback mask when other methods fail"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple Otsu thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If we have an alpha channel, use it to refine the mask
        if alpha is not None:
            # Use alpha channel as additional constraint
            alpha_mask = (alpha > 128).astype(np.uint8) * 255
            # Combine with edge-based mask
            mask = cv2.bitwise_and(mask, alpha_mask)
        
        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        return mask
    
    def create_detailed_mask(self, image: np.ndarray, method: str = "combined") -> Dict[str, Any]:
        """
        Create a detailed mask using the specified method
        """
        print(f"Creating detailed mask using method: {method}")
        
        try:
            if method == "bilateral_edge":
                mask = self.bilateral_edge_preserving_mask(image)
                quality_score = self._evaluate_mask_quality(mask, image, "edge_preservation")
                
            elif method == "texture_aware":
                mask = self.texture_aware_mask(image)
                quality_score = self._evaluate_mask_quality(mask, image, "texture_consistency")
                
            elif method == "depth_aware":
                mask = self.depth_aware_mask(image)
                quality_score = self._evaluate_mask_quality(mask, image, "depth_separation")
                
            elif method == "anatomical":
                mask = self.anatomical_boundary_mask(image)
                quality_score = self._evaluate_mask_quality(mask, image, "boundary_detection")
                
            elif method == "mean_shift":
                mask = self.mean_shift_segmentation_mask(image)
                quality_score = self._evaluate_mask_quality(mask, image, "region_consistency")
                
            elif method == "depth_texture_fusion":
                mask = self.depth_texture_fusion_mask(image)
                quality_score = self._evaluate_mask_quality(mask, image, "fusion_quality")
                
            elif method == "combined":
                # Try multiple methods and select the best one, including the new fusion method
                methods_to_try = ["depth_texture_fusion", "bilateral_edge", "anatomical", "depth_aware"]
                best_mask = None
                best_quality = 0
                best_method = None
                
                for m in methods_to_try:
                    try:
                        test_result = self.create_detailed_mask(image, m)
                        if test_result['quality_score'] > best_quality:
                            best_quality = test_result['quality_score']
                            best_mask = test_result['mask']
                            best_method = m
                    except Exception as e:
                        print(f"Method {m} failed: {e}")
                        continue
                
                if best_mask is not None:
                    mask = best_mask
                    quality_score = best_quality
                    print(f"Selected best method: {best_method} with quality: {quality_score:.3f}")
                else:
                    mask = self._fallback_threshold_mask(image)
                    quality_score = 0.3
                    
            else:
                mask = self.bilateral_edge_preserving_mask(image)  # Default
                quality_score = self._evaluate_mask_quality(mask, image, "edge_preservation")
            
            return {
                'mask': mask,
                'quality_score': quality_score,
                'method': method,
                'mask_area_percentage': (np.sum(mask > 0) / mask.size) * 100
            }
            
        except Exception as e:
            print(f"Detailed masking failed with error: {e}")
            fallback_mask = self._fallback_threshold_mask(image)
            return {
                'mask': fallback_mask,
                'quality_score': 0.2,
                'method': 'fallback',
                'mask_area_percentage': (np.sum(fallback_mask > 0) / fallback_mask.size) * 100,
                'error': str(e)
            }
    
    def _evaluate_mask_quality(self, mask: np.ndarray, original_image: np.ndarray, 
                             focus: str = "general") -> float:
        """
        Evaluate mask quality with different focus areas
        """
        try:
            h, w = mask.shape
            mask_area = np.sum(mask > 0)
            
            if mask_area == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if len(original_image.shape) == 3:
                gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = original_image.copy()
            
            # Basic metrics
            coverage_ratio = mask_area / (h * w)
            
            # Edge alignment
            edges = cv2.Canny(gray, 50, 150)
            mask_edges = cv2.Canny(mask, 50, 150)
            edge_overlap = np.sum((edges > 0) & (mask_edges > 0))
            total_edges = np.sum(edges > 0)
            edge_alignment = edge_overlap / max(1, total_edges)
            
            # Mask smoothness (boundary quality)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            else:
                compactness = 0
            
            # Focus-specific quality metrics
            if focus == "edge_preservation":
                quality = edge_alignment * 0.6 + compactness * 0.2 + min(coverage_ratio * 2, 1.0) * 0.2
            elif focus == "texture_consistency":
                # Evaluate texture homogeneity within mask
                masked_region = gray[mask > 0]
                if len(masked_region) > 0:
                    texture_variance = np.var(masked_region) / 255.0
                    texture_consistency = 1.0 / (1.0 + texture_variance)
                else:
                    texture_consistency = 0
                quality = texture_consistency * 0.5 + edge_alignment * 0.3 + compactness * 0.2
            elif focus == "depth_separation":
                # Evaluate how well the mask separates different depth layers
                quality = edge_alignment * 0.4 + compactness * 0.4 + min(coverage_ratio * 2, 1.0) * 0.2
            elif focus == "boundary_detection":
                # Focus on boundary quality
                quality = edge_alignment * 0.7 + compactness * 0.3
            elif focus == "region_consistency":
                # Focus on region homogeneity
                quality = compactness * 0.5 + edge_alignment * 0.3 + min(coverage_ratio * 2, 1.0) * 0.2
            elif focus == "fusion_quality":
                # Comprehensive quality for depth-texture fusion
                # This method should excel in both edge preservation and region coherence
                quality = edge_alignment * 0.5 + compactness * 0.3 + min(coverage_ratio * 2, 1.0) * 0.2
            else:
                # General quality
                quality = edge_alignment * 0.4 + compactness * 0.3 + min(coverage_ratio * 2, 1.0) * 0.3
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            print(f"Quality evaluation failed: {e}")
            return 0.1