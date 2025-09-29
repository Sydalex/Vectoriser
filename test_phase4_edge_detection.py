#!/usr/bin/env python3
"""
Phase 4 Advanced Testing Script: Edge Detection Optimization

This script provides comprehensive testing and optimization for Phase 4 
of the conversion pipeline, focusing on improving edge detection quality.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_to_dwg import ImageToDwgConverter
import os
from typing import Dict, List, Tuple, Any
import time


def test_phase4_comprehensive():
    """Comprehensive testing of Phase 4 edge detection methods"""
    print("=" * 70)
    print("PHASE 4: EDGE DETECTION - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    # Initialize converter
    converter = ImageToDwgConverter()
    
    # Load test image
    test_images = ["Swimmer.png", "test_image.png", "input_image.png"]
    image = None
    image_path = None
    
    for img_path in test_images:
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image_path = img_path
            break
    
    if image is None:
        print("❌ No test image found. Please provide one of:", test_images)
        print("Creating a test image for demonstration...")
        image = create_test_image()
        image_path = "generated_test.png"
    
    print(f"✅ Using image: {image_path}")
    print(f"✅ Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Generate mask and preprocessed image
    print("\n📋 Preparing image through phases 1-3...")
    mask = converter._get_object_mask(image)
    processed = converter._preprocess_and_mask(image, mask)
    
    # Test different edge detection approaches
    results = {}
    
    # 1. Current method variations
    print("\n" + "=" * 70)
    print("1. TESTING CURRENT METHOD WITH PARAMETER VARIATIONS")
    print("=" * 70)
    results['current_variations'] = test_current_method_variations(processed, converter)
    
    # 2. Alternative edge detection methods
    print("\n" + "=" * 70)
    print("2. TESTING ALTERNATIVE EDGE DETECTION METHODS")
    print("=" * 70)
    results['alternative_methods'] = test_alternative_methods(processed)
    
    # 3. Advanced filtering techniques
    print("\n" + "=" * 70)
    print("3. TESTING ADVANCED FILTERING TECHNIQUES")
    print("=" * 70)
    results['advanced_filters'] = test_advanced_filtering(processed)
    
    # 4. Multi-scale edge detection
    print("\n" + "=" * 70)
    print("4. TESTING MULTI-SCALE EDGE DETECTION")
    print("=" * 70)
    results['multiscale'] = test_multiscale_edge_detection(processed)
    
    # 5. Adaptive thresholding
    print("\n" + "=" * 70)
    print("5. TESTING ADAPTIVE THRESHOLD CALCULATION")
    print("=" * 70)
    results['adaptive'] = test_adaptive_thresholds(processed)
    
    # Analyze and compare results
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    best_config = analyze_results(results)
    
    # Visualize results
    if input("\n📊 Create visualizations? (y/n): ").lower() == 'y':
        visualize_edge_detection_results(image, mask, processed, results, best_config)
    
    return results, best_config


def test_current_method_variations(processed: np.ndarray, converter) -> Dict:
    """Test variations of the current edge detection method"""
    
    variations = [
        {
            "name": "Ultra Sensitive",
            "canny_low": 20,
            "canny_high": 60,
            "bilateral_d": 9,
            "bilateral_sigma": 50,
            "morph_kernel": 2
        },
        {
            "name": "Sensitive",
            "canny_low": 30,
            "canny_high": 90,
            "bilateral_d": 9,
            "bilateral_sigma": 75,
            "morph_kernel": 3
        },
        {
            "name": "Balanced (Current)",
            "canny_low": 50,
            "canny_high": 150,
            "bilateral_d": 9,
            "bilateral_sigma": 75,
            "morph_kernel": 3
        },
        {
            "name": "Conservative",
            "canny_low": 70,
            "canny_high": 180,
            "bilateral_d": 11,
            "bilateral_sigma": 100,
            "morph_kernel": 3
        },
        {
            "name": "Ultra Conservative",
            "canny_low": 100,
            "canny_high": 250,
            "bilateral_d": 13,
            "bilateral_sigma": 150,
            "morph_kernel": 4
        }
    ]
    
    results = {}
    
    for var in variations:
        print(f"\n📍 Testing: {var['name']}")
        
        # Apply bilateral filter
        smoothed = cv2.bilateralFilter(
            processed, 
            var['bilateral_d'], 
            var['bilateral_sigma'], 
            var['bilateral_sigma']
        )
        
        # Apply Canny edge detection
        edges = cv2.Canny(smoothed, var['canny_low'], var['canny_high'])
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (var['morph_kernel'], var['morph_kernel'])
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Calculate metrics
        metrics = calculate_edge_metrics(edges)
        
        results[var['name']] = {
            'edges': edges,
            'config': var,
            'metrics': metrics
        }
        
        print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
        print(f"  ✅ Connected components: {metrics['num_components']}")
        print(f"  ✅ Avg component size: {metrics['avg_component_size']:.1f} pixels")
    
    return results


def test_alternative_methods(processed: np.ndarray) -> Dict:
    """Test alternative edge detection methods"""
    
    results = {}
    
    # 1. Sobel edge detection
    print("\n📍 Testing: Sobel Edge Detection")
    sobel_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edges = (sobel_magnitude > np.percentile(sobel_magnitude, 90)).astype(np.uint8) * 255
    
    metrics = calculate_edge_metrics(sobel_edges)
    results['Sobel'] = {
        'edges': sobel_edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 2. Laplacian edge detection
    print("\n📍 Testing: Laplacian Edge Detection")
    laplacian = cv2.Laplacian(processed, cv2.CV_64F, ksize=3)
    laplacian_edges = (np.abs(laplacian) > np.percentile(np.abs(laplacian), 90)).astype(np.uint8) * 255
    
    metrics = calculate_edge_metrics(laplacian_edges)
    results['Laplacian'] = {
        'edges': laplacian_edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 3. Scharr edge detection
    print("\n📍 Testing: Scharr Edge Detection")
    scharr_x = cv2.Scharr(processed, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(processed, cv2.CV_64F, 0, 1)
    scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_edges = (scharr_magnitude > np.percentile(scharr_magnitude, 90)).astype(np.uint8) * 255
    
    metrics = calculate_edge_metrics(scharr_edges)
    results['Scharr'] = {
        'edges': scharr_edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 4. Structured Edge Detection (if available)
    print("\n📍 Testing: Morphological Gradient")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_gradient = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
    morph_edges = (morph_gradient > np.percentile(morph_gradient, 85)).astype(np.uint8) * 255
    
    metrics = calculate_edge_metrics(morph_edges)
    results['Morphological'] = {
        'edges': morph_edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    return results


def test_advanced_filtering(processed: np.ndarray) -> Dict:
    """Test advanced filtering techniques before edge detection"""
    
    results = {}
    
    # 1. Guided filter + Canny (skip if opencv-contrib not available)
    try:
        print("\n📍 Testing: Guided Filter + Canny")
        guided = cv2.ximgproc.guidedFilter(
            guide=processed,
            src=processed,
            radius=5,
            eps=10
        )
        edges = cv2.Canny(guided, 50, 150)
        
        metrics = calculate_edge_metrics(edges)
        results['Guided Filter'] = {
            'edges': edges,
            'metrics': metrics
        }
        print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    except AttributeError:
        print("  ⚠️ Guided Filter skipped (opencv-contrib-python not installed)")
    
    # 2. Anisotropic diffusion + Canny
    print("\n📍 Testing: Anisotropic Diffusion + Canny")
    anisotropic = anisotropic_diffusion(processed, iterations=10)
    edges = cv2.Canny(anisotropic, 50, 150)
    
    metrics = calculate_edge_metrics(edges)
    results['Anisotropic'] = {
        'edges': edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 3. Non-local means + Canny
    print("\n📍 Testing: Non-local Means + Canny")
    nlm = cv2.fastNlMeansDenoising(processed, h=10, templateWindowSize=7, searchWindowSize=21)
    edges = cv2.Canny(nlm, 50, 150)
    
    metrics = calculate_edge_metrics(edges)
    results['Non-local Means'] = {
        'edges': edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 4. Median filter + Canny (for comparison)
    print("\n📍 Testing: Median Filter + Canny")
    median = cv2.medianBlur(processed, 5)
    edges = cv2.Canny(median, 50, 150)
    
    metrics = calculate_edge_metrics(edges)
    results['Median Filter'] = {
        'edges': edges,
        'metrics': metrics
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    return results


def test_multiscale_edge_detection(processed: np.ndarray) -> Dict:
    """Test multi-scale edge detection approach"""
    
    results = {}
    scales = [1.0, 0.75, 0.5]
    
    print("\n📍 Testing: Multi-scale Edge Detection")
    
    combined_edges = np.zeros_like(processed)
    
    for scale in scales:
        # Resize image
        h, w = processed.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Detect edges at this scale
        edges = cv2.Canny(resized, 50, 150)
        
        # Resize back to original size
        edges_resized = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Combine with weight based on scale
        weight = scale
        combined_edges = cv2.addWeighted(
            combined_edges.astype(np.float32), 
            1.0, 
            edges_resized.astype(np.float32) * weight, 
            1.0, 
            0
        )
    
    # Normalize and threshold
    combined_edges = ((combined_edges / combined_edges.max()) * 255).astype(np.uint8)
    combined_edges = (combined_edges > 127).astype(np.uint8) * 255
    
    metrics = calculate_edge_metrics(combined_edges)
    results['Multi-scale'] = {
        'edges': combined_edges,
        'metrics': metrics,
        'scales': scales
    }
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    print(f"  ✅ Scales used: {scales}")
    
    return results


def test_adaptive_thresholds(processed: np.ndarray) -> Dict:
    """Test adaptive threshold calculation methods"""
    
    results = {}
    
    # 1. Otsu's method for automatic threshold
    print("\n📍 Testing: Otsu's Automatic Thresholding")
    blur = cv2.GaussianBlur(processed, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    low_thresh = 0.5 * otsu_thresh
    high_thresh = otsu_thresh
    edges = cv2.Canny(processed, low_thresh, high_thresh)
    
    metrics = calculate_edge_metrics(edges)
    results['Otsu'] = {
        'edges': edges,
        'metrics': metrics,
        'thresholds': (low_thresh, high_thresh)
    }
    print(f"  ✅ Computed thresholds: Low={low_thresh:.1f}, High={high_thresh:.1f}")
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 2. Median-based thresholding
    print("\n📍 Testing: Median-based Thresholding")
    median_val = np.median(processed[processed > 0])  # Only consider non-zero pixels
    sigma = 0.33
    
    low_thresh = int(max(0, (1.0 - sigma) * median_val))
    high_thresh = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(processed, low_thresh, high_thresh)
    
    metrics = calculate_edge_metrics(edges)
    results['Median'] = {
        'edges': edges,
        'metrics': metrics,
        'thresholds': (low_thresh, high_thresh)
    }
    print(f"  ✅ Computed thresholds: Low={low_thresh}, High={high_thresh}")
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    # 3. Gradient magnitude-based thresholding
    print("\n📍 Testing: Gradient Magnitude-based Thresholding")
    grad_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Use percentiles of gradient magnitude
    low_thresh = np.percentile(grad_mag[grad_mag > 0], 70)
    high_thresh = np.percentile(grad_mag[grad_mag > 0], 90)
    edges = cv2.Canny(processed, int(low_thresh), int(high_thresh))
    
    metrics = calculate_edge_metrics(edges)
    results['Gradient'] = {
        'edges': edges,
        'metrics': metrics,
        'thresholds': (low_thresh, high_thresh)
    }
    print(f"  ✅ Computed thresholds: Low={low_thresh:.1f}, High={high_thresh:.1f}")
    print(f"  ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
    
    return results


def calculate_edge_metrics(edges: np.ndarray) -> Dict:
    """Calculate quality metrics for edge detection results"""
    
    # Basic metrics
    edge_pixels = np.sum(edges == 255)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    # Connected components analysis
    num_labels, labels = cv2.connectedComponents(edges)
    
    # Component size statistics
    component_sizes = []
    for i in range(1, num_labels):  # Skip background (label 0)
        component_size = np.sum(labels == i)
        component_sizes.append(component_size)
    
    avg_component_size = np.mean(component_sizes) if component_sizes else 0
    std_component_size = np.std(component_sizes) if component_sizes else 0
    max_component_size = max(component_sizes) if component_sizes else 0
    
    # Edge continuity (ratio of largest component to total edge pixels)
    continuity = (max_component_size / edge_pixels * 100) if edge_pixels > 0 else 0
    
    # Edge density in regions with edges
    if edge_pixels > 0:
        edge_mask = edges > 0
        y_coords, x_coords = np.where(edge_mask)
        
        # Bounding box of edge regions
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        
        edge_density = edge_pixels / bbox_area * 100
    else:
        edge_density = 0
    
    return {
        'edge_pixels': edge_pixels,
        'total_pixels': total_pixels,
        'edge_percentage': edge_percentage,
        'num_components': num_labels - 1,  # Exclude background
        'avg_component_size': avg_component_size,
        'std_component_size': std_component_size,
        'max_component_size': max_component_size,
        'continuity': continuity,
        'edge_density': edge_density
    }


def anisotropic_diffusion(img: np.ndarray, iterations: int = 10, 
                          kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
    """Apply anisotropic diffusion filter (Perona-Malik)"""
    
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


def analyze_results(results: Dict) -> Dict:
    """Analyze all results and determine the best configuration"""
    
    print("\n📊 METRICS COMPARISON")
    print("-" * 70)
    
    all_configs = []
    
    # Collect all configurations with their metrics
    for category, category_results in results.items():
        for name, result in category_results.items():
            metrics = result['metrics']
            
            # Calculate a quality score
            # Balance between having enough edges and not too many
            # Prefer continuous edges and reasonable density
            score = calculate_quality_score(metrics)
            
            config_info = {
                'category': category,
                'name': name,
                'metrics': metrics,
                'score': score
            }
            
            if 'config' in result:
                config_info['config'] = result['config']
            if 'thresholds' in result:
                config_info['thresholds'] = result['thresholds']
            
            all_configs.append(config_info)
    
    # Sort by score
    all_configs.sort(key=lambda x: x['score'], reverse=True)
    
    # Display top 5 configurations
    print("\n🏆 TOP 5 CONFIGURATIONS:")
    print("-" * 70)
    
    for i, config in enumerate(all_configs[:5], 1):
        print(f"\n{i}. {config['category']} - {config['name']}")
        print(f"   Score: {config['score']:.2f}")
        print(f"   Edge pixels: {config['metrics']['edge_pixels']} ({config['metrics']['edge_percentage']:.2f}%)")
        print(f"   Components: {config['metrics']['num_components']}")
        print(f"   Continuity: {config['metrics']['continuity']:.1f}%")
        print(f"   Edge density: {config['metrics']['edge_density']:.1f}%")
    
    return all_configs[0] if all_configs else None


def calculate_quality_score(metrics: Dict) -> float:
    """Calculate a quality score for edge detection results"""
    
    # Ideal values (can be tuned)
    ideal_edge_percentage = 2.0  # 2% of pixels as edges
    ideal_continuity = 50.0  # 50% of edges in largest component
    ideal_components = 20  # Around 20 significant components
    ideal_density = 30.0  # 30% density in edge regions
    
    # Calculate component scores
    edge_score = 100 * np.exp(-((metrics['edge_percentage'] - ideal_edge_percentage) / ideal_edge_percentage) ** 2)
    continuity_score = 100 * (metrics['continuity'] / 100)  # Linear scale
    component_score = 100 * np.exp(-((metrics['num_components'] - ideal_components) / ideal_components) ** 2)
    density_score = 100 * np.exp(-((metrics['edge_density'] - ideal_density) / ideal_density) ** 2)
    
    # Weight the scores
    weights = {
        'edge': 0.25,
        'continuity': 0.30,
        'components': 0.20,
        'density': 0.25
    }
    
    total_score = (
        weights['edge'] * edge_score +
        weights['continuity'] * continuity_score +
        weights['components'] * component_score +
        weights['density'] * density_score
    )
    
    return total_score


def visualize_edge_detection_results(image: np.ndarray, mask: np.ndarray, 
                                    processed: np.ndarray, results: Dict, 
                                    best_config: Dict) -> None:
    """Create comprehensive visualizations of edge detection results"""
    
    print("\n🎨 Creating visualizations...")
    
    # Select representative results for visualization
    viz_configs = []
    
    # Get best from each category
    for category in ['current_variations', 'alternative_methods', 'advanced_filters', 'adaptive']:
        if category in results:
            # Get the one with highest score from this category
            category_results = results[category]
            best_in_category = None
            best_score = -1
            
            for name, result in category_results.items():
                score = calculate_quality_score(result['metrics'])
                if score > best_score:
                    best_score = score
                    best_in_category = (name, result)
            
            if best_in_category:
                viz_configs.append((category, best_in_category[0], best_in_category[1]))
    
    # Create figure
    n_methods = len(viz_configs)
    fig = plt.figure(figsize=(20, 5 * ((n_methods + 3) // 4)))
    
    # Original, mask, and processed in first row
    ax1 = plt.subplot((n_methods + 3) // 4 + 1, 4, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2 = plt.subplot((n_methods + 3) // 4 + 1, 4, 2)
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Object Mask')
    ax2.axis('off')
    
    ax3 = plt.subplot((n_methods + 3) // 4 + 1, 4, 3)
    ax3.imshow(processed, cmap='gray')
    ax3.set_title('Preprocessed')
    ax3.axis('off')
    
    # Edge detection results
    for i, (category, name, result) in enumerate(viz_configs, 4):
        ax = plt.subplot((n_methods + 3) // 4 + 1, 4, i + 1)
        ax.imshow(result['edges'], cmap='gray')
        
        metrics = result['metrics']
        title = f"{name}\n"
        title += f"Edges: {metrics['edge_percentage']:.1f}%, "
        title += f"Components: {metrics['num_components']}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Phase 4: Edge Detection Method Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = 'phase4_edge_detection_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    plt.show()


def create_test_image() -> np.ndarray:
    """Create a synthetic test image if no test image is available"""
    
    # Create a 500x500 image with some geometric shapes
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    # Add a circle
    cv2.circle(img, (150, 150), 80, (100, 100, 100), -1)
    cv2.circle(img, (150, 150), 70, (200, 200, 200), -1)
    
    # Add a rectangle
    cv2.rectangle(img, (300, 100), (450, 250), (50, 50, 50), -1)
    cv2.rectangle(img, (310, 110), (440, 240), (150, 150, 150), -1)
    
    # Add a triangle
    pts = np.array([[250, 350], [150, 450], [350, 450]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (75, 75, 75))
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save the test image
    cv2.imwrite('generated_test.png', img)
    
    return img


def run_improved_edge_detection(image_path: str, output_path: str, best_config: Dict) -> None:
    """Run the conversion with improved edge detection based on test results"""
    
    print("\n" + "=" * 70)
    print("APPLYING IMPROVED EDGE DETECTION")
    print("=" * 70)
    
    converter = ImageToDwgConverter()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Prepare configuration based on best results
    config = {
        'canny_low': 50,
        'canny_high': 150,
        'min_len': 25,
        'epsilon': 1.5
    }
    
    # Update with best configuration if available
    if best_config and 'config' in best_config:
        config.update(best_config['config'])
    elif best_config and 'thresholds' in best_config:
        config['canny_low'] = int(best_config['thresholds'][0])
        config['canny_high'] = int(best_config['thresholds'][1])
    
    print(f"Using configuration:")
    print(f"  - Canny Low: {config['canny_low']}")
    print(f"  - Canny High: {config['canny_high']}")
    print(f"  - Min Length: {config['min_len']}")
    print(f"  - Epsilon: {config['epsilon']}")
    
    # Run conversion
    try:
        converter.convert(image_path, output_path, config)
        print(f"✅ Conversion completed: {output_path}")
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")


def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("PHASE 4 EDGE DETECTION - ADVANCED TESTING & OPTIMIZATION")
    print("=" * 70)
    
    # Run comprehensive tests
    results, best_config = test_phase4_comprehensive()
    
    # Ask if user wants to apply the best configuration
    if best_config:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print(f"\n🎯 Best configuration found: {best_config['category']} - {best_config['name']}")
        print(f"   Quality score: {best_config['score']:.2f}")
        
        if input("\n💡 Apply this configuration to a conversion? (y/n): ").lower() == 'y':
            image_path = input("Enter input image path: ").strip()
            output_path = input("Enter output DWG path: ").strip()
            
            if os.path.exists(image_path):
                run_improved_edge_detection(image_path, output_path, best_config)
            else:
                print(f"❌ Image not found: {image_path}")
    
    print("\n✅ Phase 4 testing completed!")


if __name__ == "__main__":
    main()