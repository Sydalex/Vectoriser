#!/usr/bin/env python3
"""
Contrast Enhancement Testing Script

This script tests different contrast enhancement methods in Phase 3
to make the lines clearer and more defined.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_to_dwg import ImageToDwgConverter
import os


def test_contrast_enhancement():
    """Test different contrast enhancement methods"""
    print("=" * 60)
    print("CONTRAST ENHANCEMENT ANALYSIS")
    print("=" * 60)
    
    # Load image and mask
    converter = ImageToDwgConverter()
    image = cv2.imread("Swimmer.png")
    mask = converter._get_object_mask(image)
    
    print(f"✅ Original image: {image.shape}, range: {image.min()}-{image.max()}")
    print(f"✅ Original mask: {mask.shape}, unique: {np.unique(mask)}")
    
    # Test different contrast enhancement approaches
    variations = [
        {
            "name": "Current Method",
            "description": "No contrast enhancement",
            "contrast_method": "none"
        },
        {
            "name": "Histogram Equalization",
            "description": "Global histogram equalization",
            "contrast_method": "hist_eq"
        },
        {
            "name": "CLAHE",
            "description": "Contrast Limited Adaptive Histogram Equalization",
            "contrast_method": "clahe"
        },
        {
            "name": "Gamma Correction",
            "description": "Gamma correction (γ=0.5)",
            "contrast_method": "gamma",
            "gamma": 0.5
        },
        {
            "name": "Gamma Correction Bright",
            "description": "Gamma correction (γ=0.3)",
            "contrast_method": "gamma",
            "gamma": 0.3
        },
        {
            "name": "Linear Stretch",
            "description": "Linear contrast stretching",
            "contrast_method": "linear_stretch"
        },
        {
            "name": "Unsharp Masking",
            "description": "Unsharp masking for edge enhancement",
            "contrast_method": "unsharp"
        },
        {
            "name": "Combined CLAHE + Gamma",
            "description": "CLAHE followed by gamma correction",
            "contrast_method": "combined"
        }
    ]
    
    results = {}
    
    for variation in variations:
        print(f"\n--- Testing: {variation['name']} ---")
        print(f"Description: {variation['description']}")
        
        try:
            processed = preprocess_with_contrast(image, mask, variation)
            
            # Analyze results
            stats = analyze_contrast_result(processed, variation['name'])
            results[variation['name']] = {
                'processed': processed,
                'stats': stats,
                'variation': variation
            }
            
            print(f"✅ {variation['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ Error in {variation['name']}: {str(e)}")
    
    return results


def preprocess_with_contrast(image, mask, variation):
    """Apply preprocessing with specific contrast enhancement"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply strong Gaussian blur (optimized from Phase 3)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply contrast enhancement
    contrast_method = variation['contrast_method']
    
    if contrast_method == "none":
        enhanced = blurred
    elif contrast_method == "hist_eq":
        enhanced = cv2.equalizeHist(blurred)
    elif contrast_method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
    elif contrast_method == "gamma":
        gamma = variation.get('gamma', 0.5)
        # Apply gamma correction
        enhanced = np.power(blurred / 255.0, gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)
    elif contrast_method == "linear_stretch":
        # Linear contrast stretching
        min_val, max_val = blurred.min(), blurred.max()
        if max_val > min_val:
            enhanced = ((blurred - min_val) / (max_val - min_val)) * 255.0
            enhanced = enhanced.astype(np.uint8)
        else:
            enhanced = blurred
    elif contrast_method == "unsharp":
        # Unsharp masking
        gaussian = cv2.GaussianBlur(blurred, (0, 0), 2.0)
        enhanced = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    elif contrast_method == "combined":
        # CLAHE followed by gamma correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_result = clahe.apply(blurred)
        enhanced = np.power(clahe_result / 255.0, 0.5) * 255.0
        enhanced = enhanced.astype(np.uint8)
    
    # Apply morphological operations to smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to remove background
    masked_image = cv2.bitwise_and(enhanced, enhanced, mask=mask_smooth)
    
    return masked_image


def analyze_contrast_result(processed, name):
    """Analyze the quality of contrast enhancement results"""
    
    stats = {
        'name': name,
        'shape': processed.shape,
        'dtype': processed.dtype,
        'min': processed.min(),
        'max': processed.max(),
        'mean': processed.mean(),
        'std': processed.std(),
        'non_zero_pixels': np.sum(processed > 0),
        'total_pixels': processed.shape[0] * processed.shape[1],
        'zero_pixels': np.sum(processed == 0),
        'unique_values': len(np.unique(processed))
    }
    
    stats['non_zero_percentage'] = (stats['non_zero_pixels'] / stats['total_pixels']) * 100
    stats['zero_percentage'] = (stats['zero_pixels'] / stats['total_pixels']) * 100
    
    # Calculate contrast metrics
    stats['contrast_ratio'] = stats['max'] / stats['min'] if stats['min'] > 0 else float('inf')
    stats['dynamic_range'] = stats['max'] - stats['min']
    
    print(f"  📊 Shape: {stats['shape']}")
    print(f"  📊 Range: {stats['min']}-{stats['max']} (dynamic range: {stats['dynamic_range']})")
    print(f"  📊 Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
    print(f"  📊 Contrast ratio: {stats['contrast_ratio']:.2f}")
    print(f"  📊 Non-zero pixels: {stats['non_zero_pixels']} ({stats['non_zero_percentage']:.1f}%)")
    print(f"  📊 Unique values: {stats['unique_values']}")
    
    return stats


def test_edge_detection_on_contrast_variations(results):
    """Test how different contrast enhancements affect edge detection"""
    print("\n" + "=" * 60)
    print("TESTING EDGE DETECTION ON CONTRAST ENHANCEMENT VARIATIONS")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n--- Edge Detection on: {name} ---")
        
        processed = result['processed']
        
        # Test with different Canny thresholds
        canny_configs = [
            {"low": 50, "high": 150, "name": "Default"},
            {"low": 30, "high": 100, "name": "Sensitive"},
            {"low": 80, "high": 200, "name": "Conservative"}
        ]
        
        edge_results = {}
        
        for config in canny_configs:
            try:
                # Apply Canny edge detection
                edges = cv2.Canny(processed, config['low'], config['high'])
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # Calculate statistics
                edge_pixels = np.sum(edges == 255)
                total_pixels = edges.shape[0] * edges.shape[1]
                edge_percentage = (edge_pixels / total_pixels) * 100
                
                edge_results[config['name']] = {
                    'edges': edges,
                    'pixels': edge_pixels,
                    'percentage': edge_percentage
                }
                
                print(f"  ✅ {config['name']}: {edge_pixels} pixels ({edge_percentage:.2f}%)")
                
            except Exception as e:
                print(f"  ❌ Error in {config['name']}: {str(e)}")
        
        result['edge_results'] = edge_results


def test_contour_quality_on_contrast_variations(results):
    """Test how different contrast enhancements affect contour quality"""
    print("\n" + "=" * 60)
    print("TESTING CONTOUR QUALITY ON CONTRAST ENHANCEMENT VARIATIONS")
    print("=" * 60)
    
    converter = ImageToDwgConverter()
    
    for name, result in results.items():
        if 'edge_results' not in result:
            continue
            
        print(f"\n--- Contour Analysis on: {name} ---")
        
        # Test with the best edge detection result (usually Default)
        if 'Default' in result['edge_results']:
            edges = result['edge_results']['Default']['edges']
            
            # Test contour vectorization
            config = {
                'epsilon': 1.5,
                'min_len': 15,
                'smooth_window': 3,
                'merge_distance': 5
            }
            
            try:
                contours = converter._vectorize_contours(edges, config)
                
                # Analyze contours
                total_points = sum(len(contour) for contour in contours)
                avg_points = total_points / len(contours) if len(contours) > 0 else 0
                
                # Calculate contour areas
                areas = [cv2.contourArea(contour) for contour in contours]
                total_area = sum(areas)
                avg_area = total_area / len(contours) if len(contours) > 0 else 0
                
                print(f"  ✅ Contours: {len(contours)}")
                print(f"  ✅ Total points: {total_points}")
                print(f"  ✅ Avg points/contour: {avg_points:.1f}")
                print(f"  ✅ Total area: {total_area:.0f}")
                print(f"  ✅ Avg area/contour: {avg_area:.0f}")
                
                result['contour_stats'] = {
                    'count': len(contours),
                    'total_points': total_points,
                    'avg_points': avg_points,
                    'total_area': total_area,
                    'avg_area': avg_area
                }
                
            except Exception as e:
                print(f"  ❌ Error in contour analysis: {str(e)}")


def visualize_contrast_variations(results):
    """Create visualizations of different contrast enhancement variations"""
    print("\n" + "=" * 60)
    print("CREATING CONTRAST ENHANCEMENT VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Create a large figure with subplots
        n_variations = len(results)
        fig, axes = plt.subplots(3, n_variations, figsize=(4*n_variations, 12))
        fig.suptitle('Contrast Enhancement Analysis', fontsize=16)
        
        for i, (name, result) in enumerate(results.items()):
            processed = result['processed']
            
            # Original grayscale for comparison
            if i == 0:
                original_gray = cv2.cvtColor(cv2.imread("Swimmer.png"), cv2.COLOR_BGR2GRAY)
                axes[0, i].imshow(original_gray, cmap='gray')
                axes[0, i].set_title(f'Original Grayscale')
            else:
                axes[0, i].axis('off')
            
            # Processed image
            axes[1, i].imshow(processed, cmap='gray')
            axes[1, i].set_title(f'{name}\nRange: {processed.min()}-{processed.max()}')
            
            # Edge detection result
            if 'edge_results' in result and 'Default' in result['edge_results']:
                edges = result['edge_results']['Default']['edges']
                axes[2, i].imshow(edges, cmap='gray')
                edge_pct = result['edge_results']['Default']['percentage']
                axes[2, i].set_title(f'Edges ({edge_pct:.2f}%)')
            else:
                axes[2, i].axis('off')
            
            # Remove axis ticks
            for ax in axes[:, i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('contrast_enhancement_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Contrast enhancement analysis visualization saved as 'contrast_enhancement_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")


def recommend_best_contrast(results):
    """Analyze results and recommend the best contrast enhancement approach"""
    print("\n" + "=" * 60)
    print("CONTRAST ENHANCEMENT RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze each variation
    scores = {}
    
    for name, result in results.items():
        score = 0
        stats = result['stats']
        
        # Score based on contrast metrics
        # Prefer higher dynamic range
        dynamic_range = stats['dynamic_range']
        if dynamic_range > 200:
            score += 3
        elif dynamic_range > 150:
            score += 2
        else:
            score += 1
        
        # Prefer higher standard deviation (more contrast)
        if stats['std'] > 70:
            score += 3
        elif stats['std'] > 60:
            score += 2
        elif stats['std'] > 50:
            score += 1
        
        # Score based on edge detection results
        if 'edge_results' in result and 'Default' in result['edge_results']:
            edge_pct = result['edge_results']['Default']['percentage']
            if 2.0 <= edge_pct <= 3.0:  # Good edge density
                score += 3
            elif 1.5 <= edge_pct <= 3.5:
                score += 2
            else:
                score += 1
        
        # Score based on contour quality
        if 'contour_stats' in result:
            contour_count = result['contour_stats']['count']
            if 3 <= contour_count <= 6:  # Good number of contours
                score += 2
            elif 2 <= contour_count <= 8:
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
    print(f"   - Dynamic range: {best_result['stats']['dynamic_range']}")
    print(f"   - Standard deviation: {best_result['stats']['std']:.1f}")
    
    if 'edge_results' in best_result and 'Default' in best_result['edge_results']:
        edge_pct = best_result['edge_results']['Default']['percentage']
        print(f"   - Edge pixels: {edge_pct:.2f}%")
    
    if 'contour_stats' in best_result:
        contour_count = best_result['contour_stats']['count']
        print(f"   - Contours: {contour_count}")
    
    return best_approach, best_result


def main():
    """Run contrast enhancement analysis"""
    print("🔍 CONTRAST ENHANCEMENT ANALYSIS")
    print("Testing different contrast enhancement methods to make lines clearer...")
    
    # Test different contrast enhancement variations
    results = test_contrast_enhancement()
    
    # Test edge detection on each variation
    test_edge_detection_on_contrast_variations(results)
    
    # Test contour quality on each variation
    test_contour_quality_on_contrast_variations(results)
    
    # Create visualizations
    visualize_contrast_variations(results)
    
    # Recommend best approach
    best_approach, best_result = recommend_best_contrast(results)
    
    print("\n" + "=" * 60)
    print("🎉 CONTRAST ENHANCEMENT ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check the generated files:")
    print("- contrast_enhancement_analysis.png for visual analysis")
    print(f"- Best approach: {best_approach}")


if __name__ == "__main__":
    main()