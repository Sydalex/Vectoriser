#!/usr/bin/env python3
"""
Advanced Masking System for Vectoriser

Provides multiple masking algorithms for improved object isolation including
semantic segmentation, adaptive thresholding, GrabCut, and edge-based methods.

Author: AI Assistant
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class MaskingMethod(Enum):
    """Available masking methods"""
    SEMANTIC_DEEPLAB = "semantic_deeplab"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    GRABCUT = "grabcut"
    WATERSHED = "watershed"
    EDGE_BASED = "edge_based"
    KMEANS_CLUSTERING = "kmeans_clustering"
    COMBINED = "combined"


class MaskQualityMetrics:
    """Metrics for evaluating mask quality"""
    
    @staticmethod
    def calculate_metrics(mask: np.ndarray, original_image: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for a mask"""
        h, w = mask.shape
        
        # Basic coverage metrics
        mask_area = np.sum(mask > 0)
        coverage_ratio = mask_area / (h * w)
        
        # Edge alignment metric
        edges = cv2.Canny(original_image, 50, 150)
        mask_edges = cv2.Canny(mask, 50, 150)
        edge_alignment = np.sum((edges > 0) & (mask_edges > 0)) / max(1, np.sum(edges > 0))
        
        # Compactness metric (how well-formed the mask is)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        else:
            compactness = 0
        
        # Smoothness metric (how smooth the boundaries are)
        smoothness = 1.0 - (np.sum(cv2.Laplacian(mask, cv2.CV_64F) ** 2) / mask_area) if mask_area > 0 else 0
        smoothness = max(0, min(1, smoothness / 1000))  # Normalize
        
        return {
            'coverage_ratio': coverage_ratio,
            'edge_alignment': edge_alignment,
            'compactness': compactness,
            'smoothness': smoothness,
            'overall_quality': (edge_alignment * 0.4 + compactness * 0.3 + smoothness * 0.3)
        }


class AdvancedMaskingSystem:
    """Advanced masking system with multiple algorithms"""
    
    def __init__(self):
        self.deeplab_model = None
        self.current_method = MaskingMethod.SEMANTIC_DEEPLAB
        self.mask_cache = {}
        
    # ---- Transparency and checkerboard preprocessing ----
    def _remove_checkerboard_pattern(self, image: np.ndarray) -> np.ndarray:
        # Detect common transparency checkerboard patterns and replace with white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patterns = [
            (cv2.inRange(image, (200, 200, 200), (255, 255, 255)),
             cv2.inRange(image, (170, 170, 170), (199, 199, 199))),
            (cv2.inRange(image, (180, 180, 180), (220, 220, 220)),
             cv2.inRange(image, (150, 150, 150), (179, 179, 179))),
            (cv2.inRange(image, (192, 192, 192), (255, 255, 255)),
             cv2.inRange(image, (128, 128, 128), (191, 191, 191))),
        ]
        best = None
        best_ratio = 0
        for light_mask, dark_mask in patterns:
            cb = cv2.bitwise_or(light_mask, dark_mask)
            ratio = float(np.sum(cb > 0)) / (image.shape[0] * image.shape[1])
            if ratio > best_ratio:
                best_ratio, best = ratio, cb
        if best is None or best_ratio <= 0.15:
            return image
        # Preserve areas likely to be content
        edges = cv2.Canny(gray, 30, 100)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        content_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, edge_kernel)
        color_var = np.std(image, axis=2)
        color_areas = (color_var > 5).astype(np.uint8) * 255
        actual_content = cv2.bitwise_or(content_edges, color_areas)
        actual_content = cv2.morphologyEx(actual_content, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # Replace checkerboard-only areas with white
        result = image.copy()
        replacement = cv2.bitwise_and(best, cv2.bitwise_not(actual_content))
        result[replacement > 0] = [255, 255, 255]
        return result

    def _aggressive_checkerboard_removal(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        low_sat = hsv[:, :, 1] < 30
        L = lab[:, :, 0]
        light_areas = (L > 120) & (L < 255)
        potential = (low_sat & light_areas).astype(np.uint8) * 255
        potential = cv2.morphologyEx(potential, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ratio = float(np.sum(potential > 0)) / (image.shape[0] * image.shape[1])
        if ratio <= 0.1:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 60)
        content_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        color_std = np.std(image.reshape(-1, 3), axis=1).reshape(image.shape[:2])
        color_content = (color_std > 10).astype(np.uint8) * 255
        preserve = cv2.bitwise_or(content_edges, color_content)
        result = image.copy()
        removal = cv2.bitwise_and(potential, cv2.bitwise_not(preserve))
        result[removal == 0] = result[removal == 0]
        result[removal > 0] = [255, 255, 255]
        return result

    def _preprocess_transparency(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Handle RGBA and remove checkerboard for RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            white_bg = np.ones_like(image[:, :, :3]) * 255
            alpha_norm = (alpha.astype(np.float32) / 255.0)[..., None]
            blended = (image[:, :, :3].astype(np.float32) * alpha_norm + white_bg.astype(np.float32) * (1 - alpha_norm)).astype(np.uint8)
            return blended, alpha
        elif len(image.shape) == 3 and image.shape[2] == 3:
            processed = self._remove_checkerboard_pattern(image)
            # If still a lot of gray, try aggressive
            gray_mask = cv2.inRange(processed, (150, 150, 150), (240, 240, 240))
            ratio = float(np.sum(gray_mask > 0)) / (processed.shape[0] * processed.shape[1])
            if ratio > 0.2:
                processed = self._aggressive_checkerboard_removal(processed)
            return processed, None
        else:
            return image, None
        
    def _load_deeplab_model(self):
        """Lazy load the DeepLabV3 model"""
        if self.deeplab_model is None:
            print("Loading DeepLabV3 model for semantic segmentation...")
            self.deeplab_model = deeplabv3_resnet101(pretrained=True)
            self.deeplab_model.eval()
    
    def semantic_segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """Enhanced semantic segmentation using DeepLabV3"""
        self._load_deeplab_model()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = self.deeplab_model(input_tensor)
            predictions = torch.argmax(output['out'], dim=1).squeeze(0).cpu().numpy()
        
        # Create mask for person, vehicle, furniture, and other object classes
        object_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        mask = np.isin(predictions, object_classes).astype(np.uint8) * 255
        
        # Resize back to original dimensions
        original_h, original_w = image.shape[:2]
        mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # Post-processing: fill holes and smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
        
        return mask_resized
    
    def adaptive_threshold_mask(self, image: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for high-contrast objects"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply multiple adaptive threshold methods and combine
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Find the largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined)
        
        if num_labels > 1:
            # Get the largest component (excluding background)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            mask = combined
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def grabcut_mask(self, image: np.ndarray, iterations: int = 5) -> np.ndarray:
        """GrabCut algorithm for foreground extraction"""
        h, w = image.shape[:2]
        
        # Create initial rectangle (assume object is in center 60% of image)
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
        
        # Initialize GrabCut
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result_mask = mask2 * 255
        
        # Post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
        
        return result_mask
    
    def watershed_mask(self, image: np.ndarray) -> np.ndarray:
        """Watershed segmentation for object separation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Noise removal
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Sure background area
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create mask from watershed result
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255
        
        return mask
    
    def edge_based_mask(self, image: np.ndarray) -> np.ndarray:
        """Edge-based masking with contour detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Enhanced edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask by filling the largest contour
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
        else:
            # Fallback to adaptive threshold
            return self.adaptive_threshold_mask(image)
    
    def kmeans_clustering_mask(self, image: np.ndarray, k: int = 3) -> np.ndarray:
        """K-means clustering for color-based segmentation"""
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to image shape
        labels = labels.reshape(image.shape[:2])
        
        # Find the most central cluster (assume it's the object)
        h, w = image.shape[:2]
        center_region = labels[h//3:2*h//3, w//3:2*w//3]
        most_common_label = np.bincount(center_region.flatten()).argmax()
        
        # Create mask
        mask = (labels == most_common_label).astype(np.uint8) * 255
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def combined_mask(self, image: np.ndarray) -> np.ndarray:
        """Combine multiple masking methods for best results"""
        methods = [
            (self.semantic_segmentation_mask, "semantic"),
            (self.edge_based_mask, "edge"),
            (self.adaptive_threshold_mask, "adaptive"),
        ]
        
        masks = []
        qualities = []
        
        for method_func, method_name in methods:
            try:
                mask = method_func(image)
                quality = MaskQualityMetrics.calculate_metrics(mask, image)
                masks.append(mask)
                qualities.append(quality['overall_quality'])
                print(f"Method {method_name}: quality = {quality['overall_quality']:.3f}")
            except Exception as e:
                print(f"Method {method_name} failed: {str(e)}")
                continue
        
        if not masks:
            # Fallback to simple edge-based method
            return self.edge_based_mask(image)
        
        # Select the best mask based on quality
        best_idx = np.argmax(qualities)
        best_mask = masks[best_idx]
        
        print(f"Selected best mask with quality: {qualities[best_idx]:.3f}")
        return best_mask
    
    def generate_mask(self, image: np.ndarray, method: MaskingMethod = None, **kwargs) -> Dict[str, Any]:
        """Generate mask using specified method with transparency preprocessing"""
        if method is None:
            method = self.current_method
        
        # Preprocess image for transparency/checkerboard
        processed_image, alpha = self._preprocess_transparency(image)
        
        # Cache key for performance (use processed image bytes)
        cache_key = f"{method.value}_{hash(processed_image.data.tobytes())}"
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        try:
            img = processed_image
            if method == MaskingMethod.SEMANTIC_DEEPLAB:
                mask = self.semantic_segmentation_mask(img)
            elif method == MaskingMethod.ADAPTIVE_THRESHOLD:
                mask = self.adaptive_threshold_mask(img)
            elif method == MaskingMethod.GRABCUT:
                mask = self.grabcut_mask(img, **kwargs)
            elif method == MaskingMethod.WATERSHED:
                mask = self.watershed_mask(img)
            elif method == MaskingMethod.EDGE_BASED:
                mask = self.edge_based_mask(img)
            elif method == MaskingMethod.KMEANS_CLUSTERING:
                mask = self.kmeans_clustering_mask(img, **kwargs)
            elif method == MaskingMethod.COMBINED:
                mask = self.combined_mask(img)
            else:
                mask = self.edge_based_mask(img)  # Fallback
            
            # Constrain by alpha if present
            if alpha is not None:
                alpha_mask = (alpha > 64).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask, alpha_mask)
            
            # Calculate quality metrics against processed image
            quality_metrics = MaskQualityMetrics.calculate_metrics(mask, img)
            
            result = {
                'mask': mask,
                'method': method,
                'quality_metrics': quality_metrics
            }
            
            # Cache the result
            self.mask_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Masking method {method} failed: {str(e)}")
            # Fallback to edge-based method
            fallback_mask = self.edge_based_mask(image)
            return {
                'mask': fallback_mask,
                'method': MaskingMethod.EDGE_BASED,
                'quality_metrics': MaskQualityMetrics.calculate_metrics(fallback_mask, image),
                'error': str(e)
            }
    
    def set_method(self, method: MaskingMethod):
        """Set the default masking method"""
        self.current_method = method
    
    def get_available_methods(self) -> List[str]:
        """Get list of available masking methods"""
        return [method.value for method in MaskingMethod]
    
    def clear_cache(self):
        """Clear the mask cache"""
        self.mask_cache.clear()


def create_mask_comparison_image(image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Create a comparison image showing different masking methods"""
    n_methods = len(masks)
    
    if n_methods == 0:
        return image
    
    # Calculate grid size
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (method_name, mask) in enumerate(masks.items()):
        if i < len(axes):
            axes[i].imshow(mask, cmap='gray')
            axes[i].set_title(f'{method_name}', fontsize=12)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(masks), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    import io
    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    
    # Convert buffer to numpy array
    import PIL.Image
    pil_image = PIL.Image.open(buffer)
    comparison_image = np.array(pil_image)
    
    return comparison_image