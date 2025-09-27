#!/usr/bin/env python3
"""
Phase 2 Testing Script: Object Mask Generation

This script tests different object mask generation methods to improve
semantic segmentation and object isolation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_to_dwg import ImageToDwgConverter
import os


def test_phase2_variations():
    """Test different object mask generation methods"""
    print("=" * 60)
    print("PHASE 2: OBJECT MASK GENERATION - DETAILED ANALYSIS")
    print("=" * 60)
    
    # Load image
    image = cv2.imread("Swimmer.png")
    if image is None:
        print("❌ Could not load Swimmer.png")
        return None
    
    print(f"✅ Original image: {image.shape}, range: {image.min()}-{image.max()}")
    
    # Test different mask generation approaches
    variations = [
        {
            "name": "Current DeepLabV3",
            "description": "DeepLabV3 ResNet101 (current method)",
            "mask_method": "deeplabv3"
        },
        {
            "name": "Threshold-based",
            "description": "Simple threshold-based segmentation",
            "mask_method": "threshold"
        },
        {
            "name": "Otsu Threshold",
            "description": "Otsu's automatic threshold",
            "mask_method": "otsu"
        },
        {
            "name": "Adaptive Threshold",
            "description": "Adaptive threshold segmentation",
            "mask_method": "adaptive"
        },
        {
            "name": "GrabCut",
            "description": "GrabCut interactive segmentation",
            "mask_method": "grabcut"
        },
        {
            "name": "Watershed",
            "description": "Watershed segmentation",
            "mask_method": "watershed"
        },
        {
            "name": "Edge-based",
            "description": "Edge-based object detection",
            "mask_method": "edge_based"
        },
        {
            "name": "Color-based",
            "description": "Color-based segmentation",
            "mask_method": "color_based"
        }
    ]
    
    results = {}
    
    for variation in variations:
        print(f"\n--- Testing: {variation['name']} ---")
        print(f"Description: {variation['description']}")
        
        try:
            mask = generate_mask_variation(image, variation)
            
            # Analyze results
            stats = analyze_mask_result(mask, variation['name'])
            results[variation['name']] = {
                'mask': mask,
                'stats': stats,
                'variation': variation
            }
            
            print(f"✅ {variation['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ Error in {variation['name']}: {str(e)}")
    
    return results


def generate_mask_variation(image, variation):
    """Generate mask using specific method"""
    
    mask_method = variation['mask_method']
    
    if mask_method == "deeplabv3":
        # Current DeepLabV3 method
        converter = ImageToDwgConverter()
        mask = converter._get_object_mask(image)
        return mask
    
    elif mask_method == "threshold":
        # Simple threshold-based segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use median value as threshold
        median_val = np.median(gray)
        _, mask = cv2.threshold(gray, median_val, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    elif mask_method == "otsu":
        # Otsu's automatic threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's threshold
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    elif mask_method == "adaptive":
        # Adaptive threshold segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    elif mask_method == "grabcut":
        # GrabCut interactive segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create initial mask (assume object is in center)
        h, w = gray.shape
        mask = np.zeros((h, w), np.uint8)
        
        # Define rectangle around center (rough object area)
        margin = min(h, w) // 4
        rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        # Apply GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Extract foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask = mask2 * 255
        
        return mask
    
    elif mask_method == "watershed":
        # Watershed segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create mask (markers > 1 are objects)
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255
        
        return mask
    
    elif mask_method == "edge_based":
        # Edge-based object detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to create thicker boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Fill the largest contour
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
        else:
            mask = np.zeros_like(gray)
        
        return mask
    
    elif mask_method == "color_based":
        # Color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges (this is a simple approach)
        # For darker objects (like silhouettes)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    else:
        raise ValueError(f"Unknown mask method: {mask_method}")


def analyze_mask_result(mask, name):
    """Analyze the quality of mask generation results"""
    
    stats = {
        'name': name,
        'shape': mask.shape,
        'dtype': mask.dtype,
        'min': mask.min(),
        'max': mask.max(),
        'unique_values': np.unique(mask),
        'white_pixels': np.sum(mask == 255),
        'black_pixels': np.sum(mask == 0),
        'total_pixels': mask.shape[0] * mask.shape[1]
    }
    
    stats['white_percentage'] = (stats['white_pixels'] / stats['total_pixels']) * 100
    stats['black_percentage'] = (stats['black_pixels'] / stats['total_pixels']) * 100
    
    # Calculate mask quality metrics
    stats['coverage'] = stats['white_percentage']
    
    # Calculate mask connectivity (number of connected components)
    num_labels, labels = cv2.connectedComponents(mask)
    stats['connected_components'] = num_labels - 1  # Subtract background
    
    print(f"  📊 Shape: {stats['shape']}")
    print(f"  📊 Range: {stats['min']}-{stats['max']}")
    print(f"  📊 Unique values: {stats['unique_values']}")
    print(f"  📊 White pixels: {stats['white_pixels']} ({stats['white_percentage']:.1f}%)")
    print(f"  📊 Black pixels: {stats['black_pixels']} ({stats['black_percentage']:.1f}%)")
    print(f"  📊 Connected components: {stats['connected_components']}")
    
    return stats


def test_downstream_effects(results):
    """Test how different masks affect the downstream processing"""
    print("\n" + "=" * 60)
    print("TESTING DOWNSTREAM EFFECTS OF DIFFERENT MASKS")
    print("=" * 60)
    
    converter = ImageToDwgConverter()
    image = cv2.imread("Swimmer.png")
    
    for name, result in results.items():
        print(f"\n--- Downstream Analysis: {name} ---")
        
        mask = result['mask']
        
        try:
            # Test preprocessing with this mask
            processed = converter._preprocess_and_mask(image, mask)
            
            # Test edge detection
            edges = converter._detect_edges(processed, {'canny_low': 50, 'canny_high': 150})
            
            # Test contour vectorization
            config = {
                'epsilon': 1.5,
                'min_len': 15,
                'smooth_window': 3,
                'merge_distance': 5
            }
            
            contours = converter._vectorize_contours(edges, config)
            
            # Analyze results
            edge_pixels = np.sum(edges == 255)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_percentage = (edge_pixels / total_pixels) * 100
            
            total_points = sum(len(contour) for contour in contours)
            avg_points = total_points / len(contours) if len(contours) > 0 else 0
            
            print(f"  ✅ Edge pixels: {edge_pixels} ({edge_percentage:.2f}%)")
            print(f"  ✅ Contours: {len(contours)}")
            print(f"  ✅ Total points: {total_points}")
            print(f"  ✅ Avg points/contour: {avg_points:.1f}")
            
            result['downstream_stats'] = {
                'edge_pixels': edge_pixels,
                'edge_percentage': edge_percentage,
                'contour_count': len(contours),
                'total_points': total_points,
                'avg_points': avg_points
            }
            
        except Exception as e:
            print(f"  ❌ Error in downstream analysis: {str(e)}")


def visualize_mask_variations(results):
    """Create visualizations of different mask generation variations"""
    print("\n" + "=" * 60)
    print("CREATING MASK GENERATION VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Create a large figure with subplots
        n_variations = len(results)
        fig, axes = plt.subplots(2, n_variations, figsize=(4*n_variations, 8))
        fig.suptitle('Phase 2: Object Mask Generation Analysis', fontsize=16)
        
        # Load original image
        original_image = cv2.imread("Swimmer.png")
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        for i, (name, result) in enumerate(results.items()):
            mask = result['mask']
            
            # Original image
            axes[0, i].imshow(original_gray, cmap='gray')
            axes[0, i].set_title(f'{name}\nOriginal')
            
            # Mask
            axes[1, i].imshow(mask, cmap='gray')
            coverage = result['stats']['white_percentage']
            axes[1, i].set_title(f'Mask ({coverage:.1f}%)')
            
            # Remove axis ticks
            for ax in axes[:, i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('phase2_mask_generation_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Mask generation analysis visualization saved as 'phase2_mask_generation_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")


def recommend_best_mask_method(results):
    """Analyze results and recommend the best mask generation approach"""
    print("\n" + "=" * 60)
    print("MASK GENERATION RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze each variation
    scores = {}
    
    for name, result in results.items():
        score = 0
        stats = result['stats']
        
        # Score based on mask quality
        # Prefer reasonable coverage (20-50%)
        coverage = stats['white_percentage']
        if 20 <= coverage <= 50:
            score += 3
        elif 15 <= coverage <= 60:
            score += 2
        else:
            score += 1
        
        # Prefer fewer connected components (more coherent object)
        components = stats['connected_components']
        if components == 1:
            score += 3
        elif components <= 3:
            score += 2
        elif components <= 5:
            score += 1
        
        # Score based on downstream results
        if 'downstream_stats' in result:
            downstream = result['downstream_stats']
            
            # Prefer good edge density (2-4%)
            edge_pct = downstream['edge_percentage']
            if 2.0 <= edge_pct <= 4.0:
                score += 3
            elif 1.5 <= edge_pct <= 5.0:
                score += 2
            else:
                score += 1
            
            # Prefer good number of contours (3-8)
            contour_count = downstream['contour_count']
            if 3 <= contour_count <= 8:
                score += 2
            elif 2 <= contour_count <= 10:
                score += 1
        
        scores[name] = score
        print(f"📊 {name}: Score = {score}")
    
    # Find best approach
    best_approach = max(scores, key=scores.get)
    print(f"\n🏆 RECOMMENDED APPROACH: {best_approach}")
    print(f"   Score: {scores[best_approach]}")
    
    # Show details of best approach
    best_result = results[best_approach]
    print(f"\n📋 Details of {best_approach}:")
    print(f"   - Coverage: {best_result['stats']['white_percentage']:.1f}%")
    print(f"   - Connected components: {best_result['stats']['connected_components']}")
    
    if 'downstream_stats' in best_result:
        downstream = best_result['downstream_stats']
        print(f"   - Edge pixels: {downstream['edge_percentage']:.2f}%")
        print(f"   - Contours: {downstream['contour_count']}")
    
    return best_approach, best_result


def main():
    """Run Phase 2 analysis"""
    print("🔍 PHASE 2: OBJECT MASK GENERATION ANALYSIS")
    print("Testing different mask generation methods to improve object isolation...")
    
    # Test different mask generation variations
    results = test_phase2_variations()
    
    if results is None:
        return
    
    # Test downstream effects
    test_downstream_effects(results)
    
    # Create visualizations
    visualize_mask_variations(results)
    
    # Recommend best approach
    best_approach, best_result = recommend_best_mask_method(results)
    
    print("\n" + "=" * 60)
    print("🎉 PHASE 2 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check the generated files:")
    print("- phase2_mask_generation_analysis.png for visual analysis")
    print(f"- Best approach: {best_approach}")


if __name__ == "__main__":
    main()