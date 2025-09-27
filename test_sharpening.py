#!/usr/bin/env python3
"""
Sharpening Filter Testing Script

This script tests different sharpening methods to improve edge detection
and make the lines even clearer and more defined.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_to_dwg import ImageToDwgConverter
import os


def test_sharpening_methods():
    """Test different sharpening methods"""
    print("=" * 60)
    print("SHARPENING FILTER ANALYSIS")
    print("=" * 60)
    
    # Load image and mask
    converter = ImageToDwgConverter()
    image = cv2.imread("Swimmer.png")
    mask = converter._get_object_mask(image)
    
    print(f"✅ Original image: {image.shape}, range: {image.min()}-{image.max()}")
    print(f"✅ Original mask: {mask.shape}, unique: {np.unique(mask)}")
    
    # Test different sharpening approaches
    variations = [
        {
            "name": "Current Method",
            "description": "No sharpening",
            "sharpening_method": "none"
        },
        {
            "name": "Unsharp Masking",
            "description": "Unsharp masking (σ=1.0)",
            "sharpening_method": "unsharp",
            "sigma": 1.0,
            "amount": 1.5
        },
        {
            "name": "Unsharp Masking Strong",
            "description": "Strong unsharp masking (σ=2.0)",
            "sharpening_method": "unsharp",
            "sigma": 2.0,
            "amount": 2.0
        },
        {
            "name": "Laplacian Sharpening",
            "description": "Laplacian kernel sharpening",
            "sharpening_method": "laplacian"
        },
        {
            "name": "High Pass Filter",
            "description": "High pass filter sharpening",
            "sharpening_method": "highpass"
        },
        {
            "name": "Gaussian + Subtract",
            "description": "Gaussian blur subtraction method",
            "sharpening_method": "gaussian_subtract"
        },
        {
            "name": "Edge Enhancement",
            "description": "Sobel-based edge enhancement",
            "sharpening_method": "edge_enhance"
        },
        {
            "name": "Combined CLAHE + Unsharp",
            "description": "CLAHE + Unsharp masking",
            "sharpening_method": "combined"
        }
    ]
    
    results = {}
    
    for variation in variations:
        print(f"\n--- Testing: {variation['name']} ---")
        print(f"Description: {variation['description']}")
        
        try:
            processed = preprocess_with_sharpening(image, mask, variation)
            
            # Analyze results
            stats = analyze_sharpening_result(processed, variation['name'])
            results[variation['name']] = {
                'processed': processed,
                'stats': stats,
                'variation': variation
            }
            
            print(f"✅ {variation['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ Error in {variation['name']}: {str(e)}")
    
    return results


def preprocess_with_sharpening(image, mask, variation):
    """Apply preprocessing with specific sharpening method"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply strong Gaussian blur (optimized from Phase 3)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply contrast enhancement (CLAHE + Gamma)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_result = clahe.apply(blurred)
    enhanced = np.power(clahe_result / 255.0, 0.7) * 255.0
    enhanced = enhanced.astype(np.uint8)
    
    # Apply sharpening
    sharpening_method = variation['sharpening_method']
    
    if sharpening_method == "none":
        sharpened = enhanced
    elif sharpening_method == "unsharp":
        sigma = variation.get('sigma', 1.0)
        amount = variation.get('amount', 1.5)
        # Unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        sharpened = cv2.addWeighted(enhanced, 1 + amount, gaussian, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif sharpening_method == "laplacian":
        # Laplacian sharpening
        laplacian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, laplacian_kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif sharpening_method == "highpass":
        # High pass filter
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif sharpening_method == "gaussian_subtract":
        # Gaussian blur subtraction method
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif sharpening_method == "edge_enhance":
        # Sobel-based edge enhancement
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        sharpened = cv2.addWeighted(enhanced, 1.0, edges, 0.3, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif sharpening_method == "combined":
        # CLAHE + Unsharp masking
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        clahe_result = clahe.apply(blurred)
        enhanced = np.power(clahe_result / 255.0, 0.7) * 255.0
        enhanced = enhanced.astype(np.uint8)
        
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Apply morphological operations to smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to remove background
    masked_image = cv2.bitwise_and(sharpened, sharpened, mask=mask_smooth)
    
    return masked_image


def analyze_sharpening_result(processed, name):
    """Analyze the quality of sharpening results"""
    
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
    
    # Calculate sharpening metrics
    stats['dynamic_range'] = stats['max'] - stats['min']
    
    # Calculate local variance (measure of sharpness)
    gray = processed.astype(np.float32)
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
    stats['local_variance'] = np.mean(local_variance)
    
    print(f"  📊 Shape: {stats['shape']}")
    print(f"  📊 Range: {stats['min']}-{stats['max']} (dynamic range: {stats['dynamic_range']})")
    print(f"  📊 Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
    print(f"  📊 Local variance: {stats['local_variance']:.1f}")
    print(f"  📊 Non-zero pixels: {stats['non_zero_pixels']} ({stats['non_zero_percentage']:.1f}%)")
    print(f"  📊 Unique values: {stats['unique_values']}")
    
    return stats


def test_edge_detection_on_sharpening_variations(results):
    """Test how different sharpening methods affect edge detection"""
    print("\n" + "=" * 60)
    print("TESTING EDGE DETECTION ON SHARPENING VARIATIONS")
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


def test_contour_quality_on_sharpening_variations(results):
    """Test how different sharpening methods affect contour quality"""
    print("\n" + "=" * 60)
    print("TESTING CONTOUR QUALITY ON SHARPENING VARIATIONS")
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


def visualize_sharpening_variations(results):
    """Create visualizations of different sharpening variations"""
    print("\n" + "=" * 60)
    print("CREATING SHARPENING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Create a large figure with subplots
        n_variations = len(results)
        fig, axes = plt.subplots(3, n_variations, figsize=(4*n_variations, 12))
        fig.suptitle('Sharpening Filter Analysis', fontsize=16)
        
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
        plt.savefig('sharpening_filter_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Sharpening filter analysis visualization saved as 'sharpening_filter_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")


def recommend_best_sharpening(results):
    """Analyze results and recommend the best sharpening approach"""
    print("\n" + "=" * 60)
    print("SHARPENING FILTER RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze each variation
    scores = {}
    
    for name, result in results.items():
        score = 0
        stats = result['stats']
        
        # Score based on sharpening metrics
        # Prefer higher local variance (sharper edges)
        local_variance = stats['local_variance']
        if local_variance > 1000:
            score += 3
        elif local_variance > 800:
            score += 2
        elif local_variance > 600:
            score += 1
        
        # Prefer higher standard deviation (more contrast)
        if stats['std'] > 80:
            score += 3
        elif stats['std'] > 70:
            score += 2
        elif stats['std'] > 60:
            score += 1
        
        # Score based on edge detection results
        if 'edge_results' in result and 'Default' in result['edge_results']:
            edge_pct = result['edge_results']['Default']['percentage']
            if 2.5 <= edge_pct <= 4.0:  # Good edge density for sharpening
                score += 3
            elif 2.0 <= edge_pct <= 4.5:
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
    print(f"   - Local variance: {best_result['stats']['local_variance']:.1f}")
    print(f"   - Standard deviation: {best_result['stats']['std']:.1f}")
    
    if 'edge_results' in best_result and 'Default' in best_result['edge_results']:
        edge_pct = best_result['edge_results']['Default']['percentage']
        print(f"   - Edge pixels: {edge_pct:.2f}%")
    
    if 'contour_stats' in best_result:
        contour_count = best_result['contour_stats']['count']
        print(f"   - Contours: {contour_count}")
    
    return best_approach, best_result


def main():
    """Run sharpening filter analysis"""
    print("🔍 SHARPENING FILTER ANALYSIS")
    print("Testing different sharpening methods to improve edge detection...")
    
    # Test different sharpening variations
    results = test_sharpening_methods()
    
    # Test edge detection on each variation
    test_edge_detection_on_sharpening_variations(results)
    
    # Test contour quality on each variation
    test_contour_quality_on_sharpening_variations(results)
    
    # Create visualizations
    visualize_sharpening_variations(results)
    
    # Recommend best approach
    best_approach, best_result = recommend_best_sharpening(results)
    
    print("\n" + "=" * 60)
    print("🎉 SHARPENING FILTER ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check the generated files:")
    print("- sharpening_filter_analysis.png for visual analysis")
    print(f"- Best approach: {best_approach}")


if __name__ == "__main__":
    main()