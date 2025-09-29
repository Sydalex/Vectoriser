#!/usr/bin/env python3
"""
Web-based Live Preview for Vectoriser Development

A Flask web application that provides real-time preview of the image-to-DWG conversion process
with adjustable parameters and multi-stage visualization.

Author: AI Assistant
Dependencies: flask, opencv-python, numpy, PIL, matplotlib, base64
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import base64
import io
import os
import json
from pathlib import Path
import tempfile
import threading
import time
import ezdxf

# Import your existing converter and new styling system
from image_to_dwg import ImageToDwgConverter
from cad_styles import CADStyleManager, aci_to_rgb
from advanced_masking import AdvancedMaskingSystem, MaskingMethod
from detailed_masking import DetailedMaskingSystem

app = Flask(__name__)
app.secret_key = 'vectoriser_preview_key'

# Global variables for the current processing state
current_state = {
    'converter': None,
    'image': None,
    'mask': None,
    'mask_result': None,  # Full mask result with quality metrics
    'processed': None,
    'edges': None,
    'contours': None,
    'dwg_path': None,
    'dwg_preview': None,
    'filename': None,
    'stats': {},
    'cad_style_manager': CADStyleManager(),
    'styled_contours': None,
    'masking_system': AdvancedMaskingSystem(),
    'detailed_masking_system': DetailedMaskingSystem()
}

# Default parameters
default_params = {
    'canny_low': 50,
    'canny_high': 150,
    'min_len': 25,
    'epsilon': 1.5,
    'blur_kernel': 15
}

def image_to_base64(image, format='PNG'):
    """Convert numpy array to base64 encoded image"""
    if image is None:
        return None
    
    # Handle different image types
    if len(image.shape) == 3:
        # Color image - convert BGR to RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
    else:
        # Grayscale image
        pil_image = Image.fromarray(image)
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{encoded}"

def create_contour_visualization(image, contours):
    """Create a visualization of contours overlaid on the original image.
    Robust to empty/invalid contours and shapes. Returns None if nothing to draw.
    """
    if image is None or contours is None:
        return None

    # Normalize to list
    try:
        contour_list = list(contours)
    except Exception:
        return None

    # Filter and normalize contours to proper shape and dtype
    valid_contours = []
    for c in contour_list:
        if c is None:
            continue
        c = np.asarray(c)
        # Accept (N,1,2) or (N,2); attempt to reshape flattened pairs
        if c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2:
            pass  # already (N,1,2)
        elif c.ndim == 2 and c.shape[1] == 2:
            c = c.reshape((-1, 1, 2))
        else:
            # Try to coerce if data length is even
            if c.size % 2 == 0:
                try:
                    c = c.reshape((-1, 1, 2))
                except Exception:
                    continue
            else:
                continue
        if c.shape[0] > 0:
            valid_contours.append(c.astype(np.int32))

    if len(valid_contours) == 0:
        return None

    contour_image = image.copy()

    # Find primary contour (largest area)
    areas = [cv2.contourArea(c) for c in valid_contours]
    primary_idx = int(np.argmax(areas)) if len(areas) > 0 else -1

    # Draw detail contours in white
    for idx, c in enumerate(valid_contours):
        if idx != primary_idx:
            cv2.drawContours(contour_image, [c], -1, (255, 255, 255), 2)

    # Draw primary contour in red
    if primary_idx >= 0:
        cv2.drawContours(contour_image, [valid_contours[primary_idx]], -1, (0, 0, 255), 3)

    return contour_image

def calculate_stats():
    """Calculate processing statistics"""
    stats = {}
    
    if current_state['image'] is not None:
        h, w = current_state['image'].shape[:2]
        stats['image_size'] = f"{w}x{h}"
        
        if current_state['mask'] is not None:
            mask_area = int(np.sum(current_state['mask'] > 0))
            mask_percentage = (mask_area / (h * w)) * 100
            stats['mask_area'] = f"{mask_percentage:.1f}%"
        
        if current_state['edges'] is not None:
            edge_pixels = int(np.sum(current_state['edges'] > 0))
            stats['edge_pixels'] = edge_pixels
        
        if current_state['contours'] is not None and len(current_state['contours']) > 0:
            stats['contour_count'] = len(current_state['contours'])
            areas = [cv2.contourArea(c) for c in current_state['contours']]
            stats['primary_area'] = f"{max(areas):.0f}"
            stats['total_points'] = int(sum(len(c) for c in current_state['contours']))
    
    current_state['stats'] = stats
    return stats

def render_dwg_to_image(dwg_path, width=800, height=600, use_styling=True):
    """Convert DWG file to a styled preview image showing CAD layers and colors"""
    try:
        # Read the DXF/DWG file
        doc = ezdxf.readfile(dwg_path)
        msp = doc.modelspace()
        
        print(f"DWG Preview Debug: Processing {len(list(msp))} entities")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(width/100, height/100), dpi=100)
        ax.set_aspect('equal')
        ax.set_facecolor('white')  # White background
        
        # Track bounds for auto-scaling
        all_x, all_y = [], []
        
        # Get layer information for styling
        layers = {}
        for layer in doc.layers:
            layers[layer.dxf.name] = {
                'color': layer.dxf.color,
                'lineweight': layer.dxf.lineweight
            }
        
        # Process all entities with styling
        entity_count = 0
        for entity in msp:
            entity_count += 1
            print(f"Processing entity {entity_count}: {entity.dxftype()} on layer {entity.dxf.layer}")
            
            if entity.dxftype() == 'LWPOLYLINE':
                # Get polyline points
                points = list(entity.get_points())
                if len(points) < 2:
                    continue
                    
                # Extract coordinates
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                print(f"Polyline has {len(points)} points, x_range: {min(x_coords):.1f} to {max(x_coords):.1f}, y_range: {min(y_coords):.1f} to {max(y_coords):.1f}")
                
                all_x.extend(x_coords)
                all_y.extend(y_coords)
                
                # Get styling from layer - always use black for visibility on white background
                layer_name = entity.dxf.layer
                color = 'black'  # Force black for now to ensure visibility
                
                if use_styling and layer_name in layers:
                    layer_info = layers[layer_name]
                    # Enhanced line weight scaling for better visibility
                    raw_weight = layer_info['lineweight']
                    if raw_weight <= 15:  # Very thin lines (0.1-0.15mm)
                        lineweight = 1.5
                    elif raw_weight <= 25:  # Thin lines (0.15-0.25mm)  
                        lineweight = 2.0
                    elif raw_weight <= 40:  # Medium lines (0.25-0.4mm)
                        lineweight = 2.5
                    else:  # Thick lines (0.4mm+)
                        lineweight = 3.0
                    print(f"Using styled line: weight={raw_weight} -> {lineweight}")
                else:
                    lineweight = 2.0
                    print("Using default styling")
                
                # Draw the polyline with styling
                ax.plot(x_coords, y_coords, 
                       color=color, 
                       linewidth=lineweight,
                       solid_capstyle='round',
                       solid_joinstyle='round',
                       antialiased=True,
                       alpha=0.8)
            
            elif entity.dxftype() == 'LINE':
                # Handle line entities
                start = entity.dxf.start
                end = entity.dxf.end
                
                all_x.extend([start[0], end[0]])
                all_y.extend([start[1], end[1]])
                
                # Get styling from layer - force black for visibility
                layer_name = entity.dxf.layer
                color = 'black'
                
                if use_styling and layer_name in layers:
                    layer_info = layers[layer_name]
                    raw_weight = layer_info['lineweight']
                    if raw_weight <= 15:
                        lineweight = 1.5
                    elif raw_weight <= 25:
                        lineweight = 2.0
                    elif raw_weight <= 40:
                        lineweight = 2.5
                    else:
                        lineweight = 3.0
                else:
                    lineweight = 2.0
                
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       color=color,
                       linewidth=lineweight,
                       antialiased=True,
                       alpha=0.8)
        
        # Set appropriate bounds with small margin
        if all_x and all_y:
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            
            # Ensure minimum range for very small objects
            if x_range < 1:
                x_range = 100
            if y_range < 1:
                y_range = 100
                
            margin = max(x_range, y_range) * 0.1
            
            x_center = (max(all_x) + min(all_x)) / 2
            y_center = (max(all_y) + min(all_y)) / 2
            
            ax.set_xlim(x_center - x_range/2 - margin, x_center + x_range/2 + margin)
            ax.set_ylim(y_center - y_range/2 - margin, y_center + y_range/2 + margin)
            
            print(f"Set bounds: x=[{x_center - x_range/2 - margin:.1f}, {x_center + x_range/2 + margin:.1f}], y=[{y_center - y_range/2 - margin:.1f}, {y_center + y_range/2 + margin:.1f}]")
        else:
            # Fallback bounds if no coordinates found
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            print("Warning: No coordinates found, using fallback bounds")
        
        # Remove all decorations - clean preview
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save to buffer with no padding
        buffer = io.BytesIO()
        fig.savefig(buffer, format='PNG', dpi=150, 
                   facecolor='white', edgecolor='none',
                   bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except Exception as e:
        # Create error image
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error rendering DWG:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='red')
        ax.set_facecolor('lightgray')
        ax.set_xticks([])
        ax.set_yticks([])
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer.getvalue()

@app.route('/')
def index():
    """Main preview interface"""
    return render_template('preview.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image data
        image_data = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Initialize converter if needed
        if current_state['converter'] is None:
            current_state['converter'] = ImageToDwgConverter()
        
        # Clean up previous temporary files
        if current_state['dwg_path'] and os.path.exists(current_state['dwg_path']):
            try:
                os.unlink(current_state['dwg_path'])
            except:
                pass
        
        # Store image
        current_state['image'] = image
        current_state['filename'] = file.filename
        
        # Reset other states
        for key in ['mask', 'processed', 'edges', 'contours', 'dwg_path', 'dwg_preview']:
            current_state[key] = None
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': file.filename,
            'size': f"{image.shape[1]}x{image.shape[0]}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    """Process image with given parameters"""
    if current_state['image'] is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    try:
        # Get parameters from request
        params = request.json or default_params
        
        # Create config
        config = {
            'canny_low': params.get('canny_low', default_params['canny_low']),
            'canny_high': params.get('canny_high', default_params['canny_high']),
            'min_len': params.get('min_len', default_params['min_len']),
            'epsilon': params.get('epsilon', default_params['epsilon'])
        }
        
        # Process in stages with error handling  
        converter = current_state['converter']
        masking_system = current_state['masking_system']
        
        try:
            # Use advanced masking system
            mask_result = masking_system.generate_mask(current_state['image'])
            current_state['mask'] = mask_result['mask']
            current_state['mask_result'] = mask_result
            print(f"Masking method: {mask_result['method'].value}, Quality: {mask_result['quality_metrics']['overall_quality']:.3f}")
        except Exception as e:
            return jsonify({'error': f'Mask generation failed: {str(e)}'}), 500
            
        try:
            current_state['processed'] = converter._preprocess_and_mask(current_state['image'], current_state['mask'])
        except Exception as e:
            return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 500
            
        try:
            current_state['edges'] = converter._detect_edges(current_state['processed'], config)
        except Exception as e:
            return jsonify({'error': f'Edge detection failed: {str(e)}'}), 500
            
        try:
            current_state['contours'] = converter._vectorize_contours(current_state['edges'], config)
        except Exception as e:
            return jsonify({'error': f'Contour vectorization failed: {str(e)}'}), 500
        
        # Generate styled DWG preview
        try:
            if current_state['contours'] is not None and len(current_state['contours']) > 0:
                # Create temporary DWG file using CAD style manager
                with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
                    temp_dwg_path = tmp.name
                
                # Generate styled DWG using CAD style manager
                h, w = current_state['image'].shape[:2]
                style_manager = current_state['cad_style_manager']
                styled_contours = style_manager.create_styled_dwg(
                    current_state['contours'], temp_dwg_path, w, h
                )
                current_state['styled_contours'] = styled_contours
                
                # Render DWG to image with styling
                dwg_image_data = render_dwg_to_image(temp_dwg_path, use_styling=True)
                
                # Convert to base64 for preview
                encoded = base64.b64encode(dwg_image_data).decode('utf-8')
                current_state['dwg_preview'] = f"data:image/png;base64,{encoded}"
                current_state['dwg_path'] = temp_dwg_path
            else:
                current_state['dwg_preview'] = None
                current_state['dwg_path'] = None
                current_state['styled_contours'] = None
        except Exception as e:
            print(f"DWG preview generation failed: {str(e)}")
            current_state['dwg_preview'] = None
            current_state['dwg_path'] = None
            current_state['styled_contours'] = None
        
        # Calculate statistics
        stats = calculate_stats()
        
        return jsonify({
            'message': 'Processing completed',
            'stats': stats,
            'config': config
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Processing error: {error_details}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/preview/<stage>')
def get_preview(stage):
    """Get preview image for a specific stage"""
    if current_state['image'] is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    try:
        image_data = None
        
        if stage == 'original':
            image_data = image_to_base64(current_state['image'])
            
        elif stage == 'mask' and current_state['mask'] is not None:
            image_data = image_to_base64(current_state['mask'])
            
        elif stage == 'processed' and current_state['processed'] is not None:
            image_data = image_to_base64(current_state['processed'])
            
        elif stage == 'edges' and current_state['edges'] is not None:
            image_data = image_to_base64(current_state['edges'])
            
        elif stage == 'contours' and current_state['contours'] is not None:
            contour_viz = create_contour_visualization(current_state['image'], current_state['contours'])
            image_data = image_to_base64(contour_viz)
            
        elif stage == 'dwg' and current_state['dwg_preview'] is not None:
            return jsonify({'image': current_state['dwg_preview']})
        
        if image_data is None:
            return jsonify({'error': f'Stage {stage} not available'}), 404
        
        return jsonify({'image': image_data})
        
    except Exception as e:
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Get current processing statistics"""
    stats = calculate_stats() if current_state['image'] is not None else {}
    
    # Add styling information
    style_manager = current_state['cad_style_manager']
    stats['cad_profile'] = style_manager.current_profile
    
    # Add masking information
    if current_state['mask_result']:
        mask_result = current_state['mask_result']
        stats['mask_method'] = mask_result['method'].value
        stats['mask_quality'] = round(mask_result['quality_metrics']['overall_quality'], 3)
        stats['mask_coverage'] = f"{mask_result['quality_metrics']['coverage_ratio']*100:.1f}%"
        stats['mask_edge_alignment'] = round(mask_result['quality_metrics']['edge_alignment'], 3)
    
    if current_state['styled_contours']:
        # Count contours by classification
        classification_counts = {}
        for styled_contour in current_state['styled_contours']:
            classification = styled_contour['classification'].value
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        stats['line_classifications'] = classification_counts
    
    return jsonify(stats)

@app.route('/cad_profiles')
def get_cad_profiles():
    """Get available CAD styling profiles"""
    style_manager = current_state['cad_style_manager']
    profiles = style_manager.get_available_profiles()
    current_profile = style_manager.current_profile
    
    return jsonify({
        'profiles': profiles,
        'current': current_profile
    })

@app.route('/set_cad_profile', methods=['POST'])
def set_cad_profile():
    """Set the active CAD styling profile"""
    data = request.json
    if not data or 'profile' not in data:
        return jsonify({'error': 'Profile name required'}), 400
    
    try:
        style_manager = current_state['cad_style_manager']
        style_manager.set_profile(data['profile'])
        
        # Regenerate DWG preview with new style if we have contours
        if current_state['contours'] is not None and len(current_state['contours']) > 0:
            # Re-process with new styling
            h, w = current_state['image'].shape[:2]
            
            # Create new temporary DWG file
            with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
                temp_dwg_path = tmp.name
            
            # Clean up old file
            if current_state['dwg_path'] and os.path.exists(current_state['dwg_path']):
                try:
                    os.unlink(current_state['dwg_path'])
                except:
                    pass
            
            # Generate new styled DWG
            styled_contours = style_manager.create_styled_dwg(
                current_state['contours'], temp_dwg_path, w, h
            )
            current_state['styled_contours'] = styled_contours
            current_state['dwg_path'] = temp_dwg_path
            
            # Update preview
            dwg_image_data = render_dwg_to_image(temp_dwg_path, use_styling=True)
            encoded = base64.b64encode(dwg_image_data).decode('utf-8')
            current_state['dwg_preview'] = f"data:image/png;base64,{encoded}"
        else:
            current_state['styled_contours'] = None
            current_state['dwg_path'] = None
            current_state['dwg_preview'] = None
        
        return jsonify({
            'message': 'Profile updated successfully',
            'profile': data['profile']
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to set profile: {str(e)}'}), 500

@app.route('/masking_methods')
def get_masking_methods():
    """Get available masking methods"""
    masking_system = current_state['masking_system']
    methods = masking_system.get_available_methods()
    current_method = masking_system.current_method.value
    
    return jsonify({
        'methods': methods,
        'current': current_method
    })

@app.route('/set_masking_method', methods=['POST'])
def set_masking_method():
    """Set the active masking method"""
    data = request.json
    if not data or 'method' not in data:
        return jsonify({'error': 'Method name required'}), 400
    
    try:
        masking_system = current_state['masking_system']
        method = MaskingMethod(data['method'])
        masking_system.set_method(method)
        
        # Regenerate mask with new method if we have an image
        if current_state['image'] is not None:
            mask_result = masking_system.generate_mask(current_state['image'])
            current_state['mask'] = mask_result['mask']
            current_state['mask_result'] = mask_result
            
            # Reprocess the rest of the pipeline
            if current_state['converter']:
                config = {
                    'canny_low': 50,
                    'canny_high': 150,
                    'min_len': 25,
                    'epsilon': 1.5
                }
                
                current_state['processed'] = current_state['converter']._preprocess_and_mask(
                    current_state['image'], current_state['mask']
                )
                current_state['edges'] = current_state['converter']._detect_edges(
                    current_state['processed'], config
                )
                current_state['contours'] = current_state['converter']._vectorize_contours(
                    current_state['edges'], config
                )
                
                # Regenerate DWG preview if we have contours
                if current_state['contours'] is not None and len(current_state['contours']) > 0:
                    h, w = current_state['image'].shape[:2]
                    style_manager = current_state['cad_style_manager']
                    
                    with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
                        temp_dwg_path = tmp.name
                    
                    styled_contours = style_manager.create_styled_dwg(
                        current_state['contours'], temp_dwg_path, w, h
                    )
                    current_state['styled_contours'] = styled_contours
                    current_state['dwg_path'] = temp_dwg_path
                    
                    # Update DWG preview
                    dwg_image_data = render_dwg_to_image(temp_dwg_path, use_styling=True)
                    encoded = base64.b64encode(dwg_image_data).decode('utf-8')
                    current_state['dwg_preview'] = f"data:image/png;base64,{encoded}"
                else:
                    current_state['styled_contours'] = None
                    current_state['dwg_path'] = None
                    current_state['dwg_preview'] = None
        
        return jsonify({
            'message': 'Masking method updated successfully',
            'method': data['method'],
            'quality': current_state['mask_result']['quality_metrics']['overall_quality'] if current_state['mask_result'] else None
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to set masking method: {str(e)}'}), 500

@app.route('/detailed_masking_methods')
def get_detailed_masking_methods():
    """Get available detailed masking methods"""
    methods = [
        {'id': 'depth_texture_fusion', 'name': 'Depth + Texture Fusion', 'description': 'Advanced fusion of depth and texture awareness - BEST for character illustrations'},
        {'id': 'bilateral_edge', 'name': 'Edge Preserving', 'description': 'Maintains fine details while smoothing regions'},
        {'id': 'texture_aware', 'name': 'Texture Aware', 'description': 'Differentiates surface textures (skin, clothing, hair)'},
        {'id': 'depth_aware', 'name': 'Depth Aware', 'description': 'Separates foreground/background layers'},
        {'id': 'anatomical', 'name': 'Anatomical Boundary', 'description': 'Detects boundaries between body parts'},
        {'id': 'mean_shift', 'name': 'Mean Shift', 'description': 'Region-based color/texture segmentation'},
        {'id': 'combined', 'name': 'Auto-Select Best', 'description': 'Automatically selects the highest quality method'}
    ]
    
    return jsonify({
        'methods': methods,
        'current': 'combined'  # Default method
    })

@app.route('/use_detailed_masking', methods=['POST'])
def use_detailed_masking():
    """Switch to detailed masking for current image"""
    if current_state['image'] is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    data = request.json
    method = data.get('method', 'combined')
    
    try:
        detailed_masking = current_state['detailed_masking_system']
        
        # Generate detailed mask
        mask_result = detailed_masking.create_detailed_mask(current_state['image'], method)
        
        # Update current state with detailed mask
        current_state['mask'] = mask_result['mask']
        current_state['mask_result'] = {
            'mask': mask_result['mask'],
            'quality_metrics': {
                'overall_quality': mask_result['quality_score'],
                'coverage_ratio': mask_result['mask_area_percentage'] / 100,
                'edge_alignment': mask_result['quality_score'],  # Simplified for now
            },
            'method': type('Method', (), {'value': f"detailed_{mask_result['method']}"})()  # Mock method object
        }
        
        # Reprocess the pipeline with the new detailed mask
        if current_state['converter']:
            config = {
                'canny_low': 50,
                'canny_high': 150,
                'min_len': 25,
                'epsilon': 1.5
            }
            
            current_state['processed'] = current_state['converter']._preprocess_and_mask(
                current_state['image'], current_state['mask']
            )
            current_state['edges'] = current_state['converter']._detect_edges(
                current_state['processed'], config
            )
            current_state['contours'] = current_state['converter']._vectorize_contours(
                current_state['edges'], config
            )
            
            # Regenerate DWG preview with improved masking
            if current_state['contours']:
                h, w = current_state['image'].shape[:2]
                style_manager = current_state['cad_style_manager']
                
                # Clean up old DWG file
                if current_state['dwg_path'] and os.path.exists(current_state['dwg_path']):
                    try:
                        os.unlink(current_state['dwg_path'])
                    except:
                        pass
                
                with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
                    temp_dwg_path = tmp.name
                
                styled_contours = style_manager.create_styled_dwg(
                    current_state['contours'], temp_dwg_path, w, h
                )
                current_state['styled_contours'] = styled_contours
                current_state['dwg_path'] = temp_dwg_path
                
                # Update DWG preview
                dwg_image_data = render_dwg_to_image(temp_dwg_path, use_styling=True)
                encoded = base64.b64encode(dwg_image_data).decode('utf-8')
                current_state['dwg_preview'] = f"data:image/png;base64,{encoded}"
        
        return jsonify({
            'message': f'Detailed masking applied successfully using {method}',
            'method': method,
            'quality_score': mask_result['quality_score'],
            'mask_coverage': f"{mask_result['mask_area_percentage']:.1f}%",
            'improved': True
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed masking error: {error_details}")
        return jsonify({'error': f'Detailed masking failed: {str(e)}'}), 500

@app.route('/save_dwg', methods=['POST'])
def save_dwg():
    """Save current result as DWG file"""
    if current_state['contours'] is None:
        return jsonify({'error': 'No processed data to save'}), 400
    
    try:
        # Use existing DWG file if available, otherwise create new one
        if current_state['dwg_path'] and os.path.exists(current_state['dwg_path']):
            temp_path = current_state['dwg_path']
        else:
            # Create temporary DWG file
            with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
                temp_path = tmp.name
            
            # Generate DWG
            h, w = current_state['image'].shape[:2]
            current_state['converter']._create_dwg(current_state['contours'], temp_path, w, h)
        
        # Return file for download
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"{Path(current_state.get('filename', 'image')).stem}_vectorized.dwg",
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({'error': f'Save failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vectoriser Live Preview</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 350px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #ddd;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .preview-area {
            flex: 1;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            margin: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 100%;
            border-radius: 4px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        .section {
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border: 1px solid #eee;
        }
        
        .section h3 {
            margin-bottom: 10px;
            color: #555;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .param-group {
            margin-bottom: 15px;
        }
        
        .param-group label {
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            font-weight: 500;
            color: #666;
        }
        
        input[type="range"] {
            width: 100%;
            margin-bottom: 5px;
        }
        
        .param-value {
            font-size: 12px;
            color: #888;
            text-align: right;
        }
        
        button {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: none;
            border-radius: 4px;
            background: #007AFF;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        button:hover {
            background: #0056CC;
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .stage-button {
            background: #6C757D;
            margin: 2px 0;
            padding: 8px;
            font-size: 12px;
        }
        
        .stage-button:hover {
            background: #5A6268;
        }
        
        .stage-button.active {
            background: #28A745;
        }
        
        .stats {
            font-size: 12px;
            line-height: 1.4;
        }
        
        .stats div {
            margin-bottom: 4px;
            padding: 2px 0;
        }
        
        .status {
            padding: 10px;
            background: #e3f2fd;
            color: #1976d2;
            text-align: center;
            font-size: 14px;
            border-radius: 4px;
            margin: 20px;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
        }
        
        .success {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 8px;
            border: 2px dashed #ddd;
            border-radius: 4px;
            background: white;
        }
        
        select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        #auto-update {
            width: auto;
            margin-right: 10px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        
        .mask-quality {
            font-size: 11px;
            padding: 4px 8px;
            margin-top: 5px;
            border-radius: 3px;
            text-align: center;
        }
        
        .quality-excellent {
            background: #d4edda;
            color: #155724;
        }
        
        .quality-good {
            background: #fff3cd;
            color: #856404;
        }
        
        .quality-poor {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1 style="font-size: 18px; margin-bottom: 20px; color: #333;">Vectoriser Preview</h1>
            
            <div class="section">
                <h3>File Operations</h3>
                <input type="file" id="imageInput" accept="image/*">
                <button onclick="processCurrent()">Process Image</button>
                <button onclick="saveDWG()">Save DWG</button>
            </div>
            
            <div class="section">
                <h3>🎨 CAD Style Profile</h3>
                <select id="cadProfile" onchange="setCadProfile()">
                    <option value="illustration">✏️ Illustration Style</option>
                    <option value="architectural">🏗️ Architectural (AIA)</option>
                    <option value="mechanical">⚙️ Mechanical (ASME)</option>
                    <option value="electrical">⚡ Electrical (IEEE)</option>
                </select>
            </div>
            
            <div class="section">
                <h3>🎭 Masking Method</h3>
                <select id="maskingMethod" onchange="setMaskingMethod()">
                    <option value="semantic_deeplab">🧠 AI Segmentation</option>
                    <option value="edge_based">✂️ Edge Detection</option>
                    <option value="adaptive_threshold">🌅 Adaptive Threshold</option>
                    <option value="grabcut">✂️ GrabCut</option>
                    <option value="watershed">🌊 Watershed</option>
                    <option value="kmeans_clustering">🎨 K-Means</option>
                    <option value="combined">✨ Smart Combined</option>
                </select>
                <div id="maskQuality" class="mask-quality"></div>
            </div>
            
            <div class="section">
                <h3>🎯 Detailed Illustration Masking</h3>
                <p style="font-size: 11px; color: #666; margin-bottom: 10px;">For detailed character illustrations requiring fine edge preservation and anatomical boundaries.</p>
                <select id="detailedMaskingMethod">
                    <option value="depth_texture_fusion">🎯 Depth + Texture Fusion (BEST)</option>
                    <option value="combined">✨ Auto-Select Best</option>
                    <option value="bilateral_edge">🔍 Edge Preserving</option>
                    <option value="anatomical">🫀 Anatomical Boundary</option>
                    <option value="depth_aware">🌊 Depth Aware</option>
                    <option value="texture_aware">🧵 Texture Aware</option>
                    <option value="mean_shift">🎨 Mean Shift</option>
                </select>
                <button onclick="useDetailedMasking()" style="margin-top: 10px; background: #FF6B35;">🎯 Apply Detailed Masking</button>
                <div id="detailedMaskResult" class="mask-quality" style="margin-top: 10px;"></div>
            </div>
            
            <div class="section">
                <h3>Processing Parameters</h3>
                <div class="param-group">
                    <label>Canny Low Threshold</label>
                    <input type="range" id="cannyLow" min="10" max="200" value="50" oninput="updateParamValue('cannyLow')">
                    <div class="param-value" id="cannyLowValue">50</div>
                </div>
                
                <div class="param-group">
                    <label>Canny High Threshold</label>
                    <input type="range" id="cannyHigh" min="50" max="400" value="150" oninput="updateParamValue('cannyHigh')">
                    <div class="param-value" id="cannyHighValue">150</div>
                </div>
                
                <div class="param-group">
                    <label>Min Contour Length</label>
                    <input type="range" id="minLen" min="5" max="100" value="25" oninput="updateParamValue('minLen')">
                    <div class="param-value" id="minLenValue">25</div>
                </div>
                
                <div class="param-group">
                    <label>Epsilon (Simplification)</label>
                    <input type="range" id="epsilon" min="0.1" max="10" step="0.1" value="1.5" oninput="updateParamValue('epsilon')">
                    <div class="param-value" id="epsilonValue">1.5</div>
                </div>
                
                <div class="checkbox-group">
                    <input type="checkbox" id="auto-update" checked>
                    <label for="auto-update" style="margin-bottom: 0;">Auto Update</label>
                </div>
            </div>
            
            <div class="section">
                <h3>Processing Stages</h3>
                <button class="stage-button" onclick="showStage('original')">Original</button>
                <button class="stage-button" onclick="showStage('mask')">Mask</button>
                <button class="stage-button" onclick="showStage('processed')">Processed</button>
                <button class="stage-button" onclick="showStage('edges')">Edges</button>
                <button class="stage-button" onclick="showStage('contours')">Contours</button>
                <button class="stage-button" onclick="showStage('dwg')" style="background: #FF6B35;">⚪⚫ Vector Output</button>
            </div>
            
            <div class="section">
                <h3>Statistics</h3>
                <div class="stats" id="stats">
                    Load an image to see statistics
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="status" id="status">Ready - Upload an image to start</div>
            <div class="preview-area" id="previewArea">
                <div style="text-align: center; color: #888;">
                    <h3>Upload an image to begin</h3>
                    <p>Select an image file and click "Process Image" to see the vectorization stages</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentStage = 'original';
        let autoUpdateTimer = null;
        let isProcessing = false;
        
        // Initialize parameter value displays
        document.addEventListener('DOMContentLoaded', function() {
            updateParamValue('cannyLow');
            updateParamValue('cannyHigh');
            updateParamValue('minLen');
            updateParamValue('epsilon');
            
            // Load CAD profiles and masking methods
            loadCadProfiles();
            loadMaskingMethods();
            
            // Add parameter change listeners for auto-update
            ['cannyLow', 'cannyHigh', 'minLen', 'epsilon'].forEach(param => {
                document.getElementById(param).addEventListener('input', () => {
                    if (document.getElementById('auto-update').checked) {
                        scheduleAutoUpdate();
                    }
                });
            });
        });
        
        function updateParamValue(paramId) {
            const slider = document.getElementById(paramId);
            const valueDiv = document.getElementById(paramId + 'Value');
            valueDiv.textContent = slider.value;
        }
        
        function scheduleAutoUpdate() {
            if (autoUpdateTimer) {
                clearTimeout(autoUpdateTimer);
            }
            autoUpdateTimer = setTimeout(() => {
                if (!isProcessing) {
                    processCurrent();
                }
            }, 500);
        }
        
        function setStatus(message, type = 'info') {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + (type === 'error' ? 'error' : type === 'success' ? 'success' : '');
        }
        
        // Handle file upload
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('image', file);
            
            setStatus('Uploading image...');
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    setStatus(data.error, 'error');
                } else {
                    setStatus(`Uploaded: ${data.filename} (${data.size})`, 'success');
                    showStage('original');
                }
            })
            .catch(error => {
                setStatus('Upload failed: ' + error.message, 'error');
            });
        });
        
        function processCurrent() {
            if (isProcessing) return;
            
            isProcessing = true;
            setStatus('Processing image...');
            
            const params = {
                canny_low: parseInt(document.getElementById('cannyLow').value),
                canny_high: parseInt(document.getElementById('cannyHigh').value),
                min_len: parseInt(document.getElementById('minLen').value),
                epsilon: parseFloat(document.getElementById('epsilon').value)
            };
            
            fetch('/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                isProcessing = false;
                if (data.error) {
                    setStatus(data.error, 'error');
                } else {
                    setStatus('Processing completed', 'success');
                    updateStats();
                    showStage(currentStage); // Refresh current view
                }
            })
            .catch(error => {
                isProcessing = false;
                setStatus('Processing failed: ' + error.message, 'error');
            });
        }
        
        function showStage(stage) {
            currentStage = stage;
            
            // Update button states
            document.querySelectorAll('.stage-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event?.target?.classList.add('active');
            
            fetch(`/preview/${stage}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('previewArea').innerHTML = 
                        `<div style="text-align: center; color: #888;">
                            <h3>Stage: ${stage}</h3>
                            <p>${data.error}</p>
                        </div>`;
                } else {
                    document.getElementById('previewArea').innerHTML = 
                        `<img src="${data.image}" alt="${stage}" class="preview-image">`;
                }
            })
            .catch(error => {
                document.getElementById('previewArea').innerHTML = 
                    `<div style="text-align: center; color: #f44;">
                        <h3>Error loading ${stage}</h3>
                        <p>${error.message}</p>
                    </div>`;
            });
        }
        
        function updateStats() {
            fetch('/stats')
            .then(response => response.json())
            .then(stats => {
                let statsHTML = '';
                Object.keys(stats).forEach(key => {
                    const label = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                    statsHTML += `<div><strong>${label}:</strong> ${stats[key]}</div>`;
                });
                
                if (statsHTML === '') {
                    statsHTML = 'No statistics available';
                }
                
                document.getElementById('stats').innerHTML = statsHTML;
            });
        }
        
        function setCadProfile() {
            const profileSelect = document.getElementById('cadProfile');
            const selectedProfile = profileSelect.value;
            
            setStatus('Updating CAD profile...');
            
            fetch('/set_cad_profile', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({'profile': selectedProfile})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    setStatus(data.error, 'error');
                } else {
                    setStatus('CAD profile updated: ' + data.profile, 'success');
                    updateStats();
                    // Refresh DWG preview if available
                    if (currentStage === 'dwg') {
                        showStage('dwg');
                    }
                }
            })
            .catch(error => {
                setStatus('Profile update failed: ' + error.message, 'error');
            });
        }
        
        function loadCadProfiles() {
            fetch('/cad_profiles')
            .then(response => response.json())
            .then(data => {
                const select = document.getElementById('cadProfile');
                select.value = data.current;
            })
            .catch(error => {
                console.log('Could not load CAD profiles:', error);
            });
        }
        
        function setMaskingMethod() {
            const methodSelect = document.getElementById('maskingMethod');
            const selectedMethod = methodSelect.value;
            
            setStatus('Updating masking method...');
            
            fetch('/set_masking_method', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({'method': selectedMethod})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    setStatus(data.error, 'error');
                } else {
                    setStatus('Masking method updated: ' + selectedMethod, 'success');
                    updateStats();
                    updateMaskQuality(data.quality);
                    // Refresh current view
                    if (['mask', 'processed', 'edges', 'contours', 'dwg'].includes(currentStage)) {
                        showStage(currentStage);
                    }
                }
            })
            .catch(error => {
                setStatus('Masking method update failed: ' + error.message, 'error');
            });
        }
        
        function loadMaskingMethods() {
            fetch('/masking_methods')
            .then(response => response.json())
            .then(data => {
                const select = document.getElementById('maskingMethod');
                select.value = data.current;
            })
            .catch(error => {
                console.log('Could not load masking methods:', error);
            });
        }
        
        function updateMaskQuality(quality) {
            const qualityDiv = document.getElementById('maskQuality');
            if (quality !== null && quality !== undefined) {
                let qualityClass = 'quality-poor';
                let qualityText = 'Poor';
                
                if (quality > 0.7) {
                    qualityClass = 'quality-excellent';
                    qualityText = 'Excellent';
                } else if (quality > 0.4) {
                    qualityClass = 'quality-good';
                    qualityText = 'Good';
                }
                
                qualityDiv.className = 'mask-quality ' + qualityClass;
                qualityDiv.textContent = `Quality: ${qualityText} (${(quality * 100).toFixed(0)}%)`;
            } else {
                qualityDiv.className = 'mask-quality';
                qualityDiv.textContent = '';
            }
        }
        
        function useDetailedMasking() {
            const methodSelect = document.getElementById('detailedMaskingMethod');
            const selectedMethod = methodSelect.value;
            
            setStatus('Applying detailed masking...');
            
            fetch('/use_detailed_masking', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({'method': selectedMethod})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    setStatus(data.error, 'error');
                } else {
                    setStatus('Detailed masking applied successfully!', 'success');
                    
                    // Update detailed mask result display
                    const resultDiv = document.getElementById('detailedMaskResult');
                    let qualityClass = 'quality-poor';
                    if (data.quality_score > 0.7) {
                        qualityClass = 'quality-excellent';
                    } else if (data.quality_score > 0.4) {
                        qualityClass = 'quality-good';
                    }
                    
                    resultDiv.className = 'mask-quality ' + qualityClass;
                    resultDiv.innerHTML = `
                        <div><strong>Method:</strong> ${data.method}</div>
                        <div><strong>Quality:</strong> ${(data.quality_score * 100).toFixed(0)}%</div>
                        <div><strong>Coverage:</strong> ${data.mask_coverage}</div>
                        ${data.improved ? '<div style="color: #28a745;">✓ Improved masking applied</div>' : ''}
                    `;
                    
                    updateStats();
                    // Refresh current view to show improved results
                    if (['mask', 'processed', 'edges', 'contours', 'dwg'].includes(currentStage)) {
                        showStage(currentStage);
                    }
                }
            })
            .catch(error => {
                setStatus('Detailed masking failed: ' + error.message, 'error');
            });
        }
        
        function saveDWG() {
            setStatus('Saving DWG file...');
            
            fetch('/save_dwg', {method: 'POST'})
            .then(response => {
                if (response.ok) {
                    return response.blob();
                } else {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Save failed');
                    });
                }
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'vectorized.dwg';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                setStatus('DWG file saved successfully', 'success');
            })
            .catch(error => {
                setStatus('Save failed: ' + error.message, 'error');
            });
        }
    </script>
</body>
</html>'''
    
    with open('templates/preview.html', 'w') as f:
        f.write(template_content)
    
    print("🚀 Starting Vectoriser Web Preview...")
    print("📱 Open your browser to: http://localhost:3000")
    print("🔄 The preview will update automatically as you adjust parameters")
    print("💡 Upload an image and start experimenting!")
    
    app.run(debug=True, host='127.0.0.1', port=3000)
