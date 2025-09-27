#!/usr/bin/env python3
"""
Phase 3 Testing Script: Image Preprocessing and Masking

This script tests and optimizes Phase 3 of the conversion pipeline
to ensure the best possible preprocessing and masking results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_to_dwg import ImageToDwgConverter
import os


def test_phase3_variations():
    """Test different preprocessing variations"""
    print("=" * 60)
    print("PHASE 3: IMAGE PREPROCESSING AND MASKING - DETAILED ANALYSIS")
    print("=" * 60)
    
    # Load image and mask
    converter = ImageToDwgConverter()
    image = cv2.imread("Swimmer.png")
    mask = converter._get_object_mask(image)
    
    print(f"✅ Original image: {image.shape}, range: {image.min()}-{image.max()}")
    print(f"✅ Original mask: {mask.shape}, unique: {np.unique(mask)}")
    
    # Test different preprocessing approaches
    variations = [
        {
            "name": "Current Method",
            "blur_kernel": (9, 9),
            "morph_ops": True,
            "description": "Current implementation"
        },
        {
            "name": "Minimal Blur",
            "blur_kernel": (3, 3),
            "morph_ops": True,
            "description": "Less aggressive blurring"
        },
        {
            "name": "Strong Blur",
            "blur_kernel": (15, 15),
            "morph_ops": True,
            "description": "More aggressive blurring"
        },
        {
            "name": "No Morphology",
            "blur_kernel": (9, 9),
            "morph_ops": False,
            "description": "No morphological operations on mask"
        },
        {
            "name": "Bilateral Filter",
            "blur_kernel": (9, 9),
            "morph_ops": True,
            "bilateral": True,
            "description": "Use bilateral filter instead of Gaussian"
        }
    ]
    
    results = {}
    
    for variation in variations:
        print(f"\n--- Testing: {variation['name']} ---")
        print(f"Description: {variation['description']}")
        
        try:
            processed = preprocess_variation(image, mask, variation)
            
            # Analyze results
            stats = analyze_preprocessing_result(processed, variation['name'])
            results[variation['name']] = {
                'processed': processed,
                'stats': stats,
                'variation': variation
            }
            
            print(f"✅ {variation['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ Error in {variation['name']}: {str(e)}")
    
    return results


def preprocess_variation(image, mask, variation):
    """Apply a specific preprocessing variation"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply blur
    if variation.get('bilateral', False):
        # Use bilateral filter
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    else:
        # Use Gaussian blur
        blur_kernel = variation['blur_kernel']
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Apply morphological operations to mask if enabled
    if variation['morph_ops']:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
    else:
        mask_smooth = mask
    
    # Apply mask to remove background
    masked_image = cv2.bitwise_and(blurred, blurred, mask=mask_smooth)
    
    return masked_image


def analyze_preprocessing_result(processed, name):
    """Analyze the quality of preprocessing results"""
    
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
    
    print(f"  📊 Shape: {stats['shape']}")
    print(f"  📊 Range: {stats['min']}-{stats['max']}")
    print(f"  📊 Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
    print(f"  📊 Non-zero pixels: {stats['non_zero_pixels']} ({stats['non_zero_percentage']:.1f}%)")
    print(f"  📊 Zero pixels: {stats['zero_pixels']} ({stats['zero_percentage']:.1f}%)")
    print(f"  📊 Unique values: {stats['unique_values']}")
    
    return stats


def test_edge_detection_on_variations(results):
    """Test how different preprocessing affects edge detection"""
    print("\n" + "=" * 60)
    print("TESTING EDGE DETECTION ON DIFFERENT PREPROCESSING VARIATIONS")
    print("=" * 60)
    
    converter = ImageToDwgConverter()
    
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


def test_contour_quality_on_variations(results):
    """Test how different preprocessing affects contour quality"""
    print("\n" + "=" * 60)
    print("TESTING CONTOUR QUALITY ON DIFFERENT PREPROCESSING VARIATIONS")
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


def visualize_preprocessing_variations(results):
    """Create visualizations of different preprocessing variations"""
    print("\n" + "=" * 60)
    print("CREATING PREPROCESSING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Create a large figure with subplots
        n_variations = len(results)
        fig, axes = plt.subplots(3, n_variations, figsize=(4*n_variations, 12))
        fig.suptitle('Phase 3: Preprocessing Variations Analysis', fontsize=16)
        
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
        plt.savefig('phase3_preprocessing_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Preprocessing analysis visualization saved as 'phase3_preprocessing_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")


def recommend_best_preprocessing(results):
    """Analyze results and recommend the best preprocessing approach"""
    print("\n" + "=" * 60)
    print("PREPROCESSING RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze each variation
    scores = {}
    
    for name, result in results.items():
        score = 0
        stats = result['stats']
        
        # Score based on preprocessing stats
        # Prefer moderate non-zero percentage (30-40%)
        non_zero_pct = stats['non_zero_percentage']
        if 30 <= non_zero_pct <= 40:
            score += 3
        elif 25 <= non_zero_pct <= 45:
            score += 2
        else:
            score += 1
        
        # Prefer good contrast (higher std)
        if stats['std'] > 30:
            score += 2
        elif stats['std'] > 20:
            score += 1
        
        # Score based on edge detection results
        if 'edge_results' in result and 'Default' in result['edge_results']:
            edge_pct = result['edge_results']['Default']['percentage']
            if 1.5 <= edge_pct <= 2.5:  # Good edge density
                score += 3
            elif 1.0 <= edge_pct <= 3.0:
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
    print(f"   - Non-zero pixels: {best_result['stats']['non_zero_percentage']:.1f}%")
    print(f"   - Standard deviation: {best_result['stats']['std']:.1f}")
    
    if 'edge_results' in best_result and 'Default' in best_result['edge_results']:
        edge_pct = best_result['edge_results']['Default']['percentage']
        print(f"   - Edge pixels: {edge_pct:.2f}%")
    
    if 'contour_stats' in best_result:
        contour_count = best_result['contour_stats']['count']
        print(f"   - Contours: {contour_count}")
    
    return best_approach, best_result


def main():
    """Run Phase 3 analysis"""
    print("🔍 PHASE 3: IMAGE PREPROCESSING AND MASKING ANALYSIS")
    print("Testing different preprocessing approaches to find the optimal method...")
    
    # Test different preprocessing variations
    results = test_phase3_variations()
    
    # Test edge detection on each variation
    test_edge_detection_on_variations(results)
    
    # Test contour quality on each variation
    test_contour_quality_on_variations(results)
    
    # Create visualizations
    visualize_preprocessing_variations(results)
    
    # Recommend best approach
    best_approach, best_result = recommend_best_preprocessing(results)
    
    print("\n" + "=" * 60)
    print("🎉 PHASE 3 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check the generated files:")
    print("- phase3_preprocessing_analysis.png for visual analysis")
    print(f"- Best approach: {best_approach}")


if __name__ == "__main__":
    main()