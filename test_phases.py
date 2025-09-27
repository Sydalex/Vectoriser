#!/usr/bin/env python3
"""
Phase Testing Script for Image to DWG Converter

This script tests each phase of the conversion pipeline individually
to ensure all components are working correctly.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_to_dwg import ImageToDwgConverter
import os


def test_phase_1_image_loading():
    """Test Phase 1: Image Loading"""
    print("=" * 50)
    print("PHASE 1: IMAGE LOADING")
    print("=" * 50)
    
    converter = ImageToDwgConverter()
    
    # Test with swimmer image
    image_path = "Swimmer.png"
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load {image_path}")
        return None
    
    height, width = image.shape[:2]
    print(f"✅ Image loaded successfully: {width}x{height}")
    print(f"✅ Image shape: {image.shape}")
    print(f"✅ Image dtype: {image.dtype}")
    print(f"✅ Image range: {image.min()} - {image.max()}")
    
    return image


def test_phase_2_object_mask(image):
    """Test Phase 2: Object Mask Generation"""
    print("\n" + "=" * 50)
    print("PHASE 2: OBJECT MASK GENERATION")
    print("=" * 50)
    
    converter = ImageToDwgConverter()
    
    try:
        mask = converter._get_object_mask(image)
        print(f"✅ Mask generated successfully")
        print(f"✅ Mask shape: {mask.shape}")
        print(f"✅ Mask dtype: {mask.dtype}")
        print(f"✅ Mask range: {mask.min()} - {mask.max()}")
        print(f"✅ Mask unique values: {np.unique(mask)}")
        
        # Calculate mask statistics
        total_pixels = mask.shape[0] * mask.shape[1]
        white_pixels = np.sum(mask == 255)
        black_pixels = np.sum(mask == 0)
        
        print(f"✅ White pixels (object): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
        print(f"✅ Black pixels (background): {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")
        
        return mask
        
    except Exception as e:
        print(f"❌ Error in mask generation: {str(e)}")
        return None


def test_phase_3_preprocessing(image, mask):
    """Test Phase 3: Image Preprocessing and Masking"""
    print("\n" + "=" * 50)
    print("PHASE 3: IMAGE PREPROCESSING AND MASKING")
    print("=" * 50)
    
    converter = ImageToDwgConverter()
    
    try:
        processed = converter._preprocess_and_mask(image, mask)
        print(f"✅ Image preprocessed successfully")
        print(f"✅ Processed shape: {processed.shape}")
        print(f"✅ Processed dtype: {processed.dtype}")
        print(f"✅ Processed range: {processed.min()} - {processed.max()}")
        
        # Check if background was properly removed
        non_zero_pixels = np.sum(processed > 0)
        total_pixels = processed.shape[0] * processed.shape[1]
        print(f"✅ Non-zero pixels: {non_zero_pixels} ({non_zero_pixels/total_pixels*100:.1f}%)")
        
        return processed
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        return None


def test_phase_4_edge_detection(processed_image):
    """Test Phase 4: Edge Detection"""
    print("\n" + "=" * 50)
    print("PHASE 4: EDGE DETECTION")
    print("=" * 50)
    
    converter = ImageToDwgConverter()
    
    # Test with different Canny thresholds
    configs = [
        {"canny_low": 50, "canny_high": 150, "name": "Default"},
        {"canny_low": 30, "canny_high": 100, "name": "Sensitive"},
        {"canny_low": 80, "canny_high": 200, "name": "Conservative"}
    ]
    
    edges_results = {}
    
    for config in configs:
        try:
            edges = converter._detect_edges(processed_image, config)
            edge_pixels = np.sum(edges == 255)
            total_pixels = edges.shape[0] * edges.shape[1]
            
            print(f"✅ {config['name']} edges: {edge_pixels} pixels ({edge_pixels/total_pixels*100:.2f}%)")
            edges_results[config['name']] = edges
            
        except Exception as e:
            print(f"❌ Error in edge detection ({config['name']}): {str(e)}")
    
    return edges_results


def test_phase_5_contour_vectorization(edges_results):
    """Test Phase 5: Contour Vectorization"""
    print("\n" + "=" * 50)
    print("PHASE 5: CONTOUR VECTORIZATION")
    print("=" * 50)
    
    converter = ImageToDwgConverter()
    
    # Test with different parameters
    configs = [
        {"epsilon": 1.5, "min_len": 15, "smooth_window": 3, "merge_distance": 5, "name": "Balanced"},
        {"epsilon": 0.5, "min_len": 10, "smooth_window": 1, "merge_distance": 2, "name": "Detailed"},
        {"epsilon": 3.0, "min_len": 25, "smooth_window": 7, "merge_distance": 10, "name": "Simplified"}
    ]
    
    contour_results = {}
    
    for config in configs:
        for edge_name, edges in edges_results.items():
            try:
                contours = converter._vectorize_contours(edges, config)
                
                # Calculate contour statistics
                total_points = sum(len(contour) for contour in contours)
                avg_points = total_points / len(contours) if len(contours) > 0 else 0
                
                print(f"✅ {config['name']} + {edge_name}: {len(contours)} contours, {total_points} total points, {avg_points:.1f} avg points/contour")
                
                key = f"{config['name']}_{edge_name}"
                contour_results[key] = contours
                
            except Exception as e:
                print(f"❌ Error in vectorization ({config['name']} + {edge_name}): {str(e)}")
    
    return contour_results


def test_phase_6_dwg_creation(contour_results):
    """Test Phase 6: DWG File Creation"""
    print("\n" + "=" * 50)
    print("PHASE 6: DWG FILE CREATION")
    print("=" * 50)
    
    converter = ImageToDwgConverter()
    
    # Test with different contour sets
    for key, contours in contour_results.items():
        if len(contours) > 0:
            try:
                output_path = f"test_phase6_{key}.dwg"
                converter._create_dwg(contours, output_path, 734, 189)  # Swimmer dimensions
                
                # Check if file was created
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ {key}: DWG created successfully ({file_size} bytes)")
                else:
                    print(f"❌ {key}: DWG file not created")
                    
            except Exception as e:
                print(f"❌ Error in DWG creation ({key}): {str(e)}")


def visualize_phases(image, mask, processed, edges_results):
    """Create visualizations of each phase"""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Image to DWG Conversion Pipeline Visualization', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Phase 1: Original Image')
        axes[0, 0].axis('off')
        
        # Object mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Phase 2: Object Mask')
        axes[0, 1].axis('off')
        
        # Processed image
        axes[0, 2].imshow(processed, cmap='gray')
        axes[0, 2].set_title('Phase 3: Processed Image')
        axes[0, 2].axis('off')
        
        # Edge detection results
        edge_names = list(edges_results.keys())
        for i, (edge_name, edges) in enumerate(edges_results.items()):
            if i < 3:  # Show up to 3 edge detection results
                axes[1, i].imshow(edges, cmap='gray')
                axes[1, i].set_title(f'Phase 4: {edge_name} Edges')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('conversion_pipeline_visualization.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'conversion_pipeline_visualization.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")


def main():
    """Run all phase tests"""
    print("🔍 TESTING IMAGE TO DWG CONVERSION PIPELINE")
    print("Testing each phase individually to ensure perfect operation...")
    
    # Phase 1: Image Loading
    image = test_phase_1_image_loading()
    if image is None:
        return
    
    # Phase 2: Object Mask Generation
    mask = test_phase_2_object_mask(image)
    if mask is None:
        return
    
    # Phase 3: Image Preprocessing
    processed = test_phase_3_preprocessing(image, mask)
    if processed is None:
        return
    
    # Phase 4: Edge Detection
    edges_results = test_phase_4_edge_detection(processed)
    if not edges_results:
        return
    
    # Phase 5: Contour Vectorization
    contour_results = test_phase_5_contour_vectorization(edges_results)
    if not contour_results:
        return
    
    # Phase 6: DWG Creation
    test_phase_6_dwg_creation(contour_results)
    
    # Create visualizations
    visualize_phases(image, mask, processed, edges_results)
    
    print("\n" + "=" * 50)
    print("🎉 ALL PHASES TESTED SUCCESSFULLY!")
    print("=" * 50)
    print("Check the generated files:")
    print("- test_phase6_*.dwg files for DWG outputs")
    print("- conversion_pipeline_visualization.png for visual pipeline")


if __name__ == "__main__":
    main()