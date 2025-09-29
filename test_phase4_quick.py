#!/usr/bin/env python3
"""
Quick non-interactive Phase 4 edge detection test
"""

import sys
import os
# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_phase4_edge_detection import (
    test_phase4_comprehensive,
    create_test_image,
    analyze_results,
    calculate_edge_metrics
)
import cv2
import numpy as np
from image_to_dwg import ImageToDwgConverter

def quick_test():
    """Run a quick non-interactive test of Phase 4"""
    
    print("\n" + "=" * 70)
    print("PHASE 4 EDGE DETECTION - QUICK TEST")
    print("=" * 70)
    
    # Create or load test image
    test_images = ["Swimmer.png", "test_image.png", "input_image.png", "generated_test.png"]
    image = None
    image_path = None
    
    for img_path in test_images:
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image_path = img_path
            break
    
    if image is None:
        print("Creating synthetic test image...")
        image = create_test_image()
        image_path = "generated_test.png"
    
    print(f"✅ Using image: {image_path}")
    print(f"✅ Image shape: {image.shape}")
    
    # Initialize converter
    converter = ImageToDwgConverter()
    
    # Run through phases 1-3
    print("\n📋 Running phases 1-3...")
    mask = converter._get_object_mask(image)
    processed = converter._preprocess_and_mask(image, mask)
    
    print(f"✅ Mask generated: {mask.shape}")
    print(f"✅ Preprocessing complete: {processed.shape}")
    
    # Test a few edge detection variations
    print("\n" + "=" * 70)
    print("TESTING EDGE DETECTION VARIATIONS")
    print("=" * 70)
    
    test_configs = [
        {
            "name": "Current Default",
            "canny_low": 50,
            "canny_high": 150,
            "description": "Current implementation defaults"
        },
        {
            "name": "Sensitive",
            "canny_low": 30,
            "canny_high": 90,
            "description": "More sensitive to edges"
        },
        {
            "name": "Conservative",
            "canny_low": 70,
            "canny_high": 180,
            "description": "Less sensitive, cleaner edges"
        },
        {
            "name": "Auto (Otsu)",
            "auto": True,
            "description": "Automatic threshold using Otsu's method"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n📍 Testing: {config['name']}")
        print(f"   {config['description']}")
        
        if config.get('auto'):
            # Automatic threshold calculation
            blur = cv2.GaussianBlur(processed, (5, 5), 0)
            otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = 0.5 * otsu_thresh
            canny_high = otsu_thresh
            print(f"   Auto thresholds: Low={canny_low:.1f}, High={canny_high:.1f}")
        else:
            canny_low = config['canny_low']
            canny_high = config['canny_high']
            print(f"   Thresholds: Low={canny_low}, High={canny_high}")
        
        # Apply edge detection
        edges = converter._detect_edges(processed, {
            'canny_low': canny_low,
            'canny_high': canny_high
        })
        
        # Calculate metrics
        metrics = calculate_edge_metrics(edges)
        
        print(f"   ✅ Edge pixels: {metrics['edge_pixels']} ({metrics['edge_percentage']:.2f}%)")
        print(f"   ✅ Components: {metrics['num_components']}")
        print(f"   ✅ Largest component: {metrics['max_component_size']} pixels")
        print(f"   ✅ Continuity: {metrics['continuity']:.1f}%")
        
        results.append({
            'name': config['name'],
            'config': config,
            'metrics': metrics,
            'edges': edges
        })
    
    # Find best configuration based on metrics
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Simple scoring: prefer moderate edge percentage with good continuity
    best_score = -1
    best_config = None
    
    for result in results:
        metrics = result['metrics']
        # Simple scoring function
        edge_score = 100 if 1.0 <= metrics['edge_percentage'] <= 3.0 else 50
        continuity_score = metrics['continuity']
        component_score = 100 if 10 <= metrics['num_components'] <= 50 else 50
        
        total_score = (edge_score * 0.3 + continuity_score * 0.5 + component_score * 0.2)
        
        print(f"\n{result['name']}:")
        print(f"  Score: {total_score:.1f}")
        print(f"  Edge %: {metrics['edge_percentage']:.2f}")
        print(f"  Continuity: {metrics['continuity']:.1f}%")
        print(f"  Components: {metrics['num_components']}")
        
        if total_score > best_score:
            best_score = total_score
            best_config = result
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"🎯 Best configuration: {best_config['name']}")
    print(f"   Score: {best_score:.1f}")
    
    # Save the best edge detection result
    output_path = f"phase4_best_edges_{best_config['name'].replace(' ', '_')}.png"
    cv2.imwrite(output_path, best_config['edges'])
    print(f"\n✅ Best edge detection result saved to: {output_path}")
    
    # Also save comparison image if possible
    try:
        comparison = np.hstack([
            cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(results[0]['edges'], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(best_config['edges'], cv2.COLOR_GRAY2BGR)
        ])
        comparison_path = "phase4_comparison.png"
        cv2.imwrite(comparison_path, comparison)
        print(f"✅ Comparison image saved to: {comparison_path}")
        print("   (Preprocessed | Default Edges | Best Edges)")
    except Exception as e:
        print(f"⚠️ Could not create comparison image: {e}")
    
    return best_config

if __name__ == "__main__":
    quick_test()
    print("\n✅ Phase 4 quick test completed!")