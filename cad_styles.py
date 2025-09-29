#!/usr/bin/env python3
"""
CAD Styling Standards for Vectoriser

Provides professional CAD layer standards, line weights, colors, and styling
profiles following industry best practices for architectural, mechanical, 
and electrical drawings.

Author: AI Assistant
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
from enum import Enum


class LineType(Enum):
    """Standard CAD line types"""
    CONTINUOUS = "CONTINUOUS"
    DASHED = "DASHED"
    CENTER = "CENTER"
    PHANTOM = "PHANTOM"
    HIDDEN = "HIDDEN"
    DOT = "DOT"


class LineClassification(Enum):
    """Line classification based on geometric properties"""
    OUTER_BOUNDARY = "outer_boundary"    # Main object outline
    INNER_BOUNDARY = "inner_boundary"    # Internal features
    DETAIL_LINE = "detail_line"          # Fine details
    CONSTRUCTION = "construction"        # Construction/reference lines
    DIMENSION = "dimension"              # Dimension lines
    TEXT = "text"                        # Text elements
    HATCH = "hatch"                      # Hatching/pattern lines


class CADProfile:
    """Base class for CAD styling profiles"""
    
    def __init__(self, name: str):
        self.name = name
        self.layers = {}
        self.line_weights = {}
        self.colors = {}
        self.line_types = {}
    
    def get_layer_style(self, classification: LineClassification) -> Dict[str, Any]:
        """Get complete style definition for a line classification"""
        return {
            'layer_name': self.layers.get(classification, 'DEFAULT'),
            'color': self.colors.get(classification, 7),  # White default
            'lineweight': self.line_weights.get(classification, 0.25),
            'linetype': self.line_types.get(classification, LineType.CONTINUOUS)
        }


class ArchitecturalProfile(CADProfile):
    """Architectural CAD standards (AIA guidelines)"""
    
    def __init__(self):
        super().__init__("Architectural")
        
        # Layer naming following AIA standards
        self.layers = {
            LineClassification.OUTER_BOUNDARY: "A-WALL-OTLN",
            LineClassification.INNER_BOUNDARY: "A-WALL-PART", 
            LineClassification.DETAIL_LINE: "A-DETL-MLIN",
            LineClassification.CONSTRUCTION: "A-GRID",
            LineClassification.DIMENSION: "A-ANNO-DIMS",
            LineClassification.TEXT: "A-ANNO-TEXT",
            LineClassification.HATCH: "A-WALL-PATT"
        }
        
        # Line weights in mm (following ISO 128)
        self.line_weights = {
            LineClassification.OUTER_BOUNDARY: 0.7,    # Heavy line
            LineClassification.INNER_BOUNDARY: 0.35,   # Medium line
            LineClassification.DETAIL_LINE: 0.18,      # Light line
            LineClassification.CONSTRUCTION: 0.13,     # Very light
            LineClassification.DIMENSION: 0.13,        # Very light
            LineClassification.TEXT: 0.18,             # Light line
            LineClassification.HATCH: 0.09             # Hairline
        }
        
        # AutoCAD Color Index (ACI)
        self.colors = {
            LineClassification.OUTER_BOUNDARY: 1,      # Red
            LineClassification.INNER_BOUNDARY: 3,      # Green
            LineClassification.DETAIL_LINE: 7,         # White/Black
            LineClassification.CONSTRUCTION: 8,        # Dark Gray
            LineClassification.DIMENSION: 4,           # Cyan
            LineClassification.TEXT: 7,               # White/Black
            LineClassification.HATCH: 8               # Dark Gray
        }
        
        # Line types
        self.line_types = {
            LineClassification.OUTER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.INNER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.DETAIL_LINE: LineType.CONTINUOUS,
            LineClassification.CONSTRUCTION: LineType.CENTER,
            LineClassification.DIMENSION: LineType.CONTINUOUS,
            LineClassification.TEXT: LineType.CONTINUOUS,
            LineClassification.HATCH: LineType.CONTINUOUS
        }


class MechanicalProfile(CADProfile):
    """Mechanical CAD standards (ASME Y14.2)"""
    
    def __init__(self):
        super().__init__("Mechanical")
        
        # Layer naming following ASME standards
        self.layers = {
            LineClassification.OUTER_BOUNDARY: "M-PART-OTLN",
            LineClassification.INNER_BOUNDARY: "M-PART-FEAT",
            LineClassification.DETAIL_LINE: "M-DETL-LINE",
            LineClassification.CONSTRUCTION: "M-CNTR-LINE", 
            LineClassification.DIMENSION: "M-DIMS",
            LineClassification.TEXT: "M-ANNO",
            LineClassification.HATCH: "M-SECT-PATT"
        }
        
        # Mechanical line weights (ASME standards)
        self.line_weights = {
            LineClassification.OUTER_BOUNDARY: 0.6,    # Thick
            LineClassification.INNER_BOUNDARY: 0.3,    # Medium
            LineClassification.DETAIL_LINE: 0.3,       # Medium
            LineClassification.CONSTRUCTION: 0.15,     # Thin
            LineClassification.DIMENSION: 0.15,        # Thin
            LineClassification.TEXT: 0.2,              # Medium-thin
            LineClassification.HATCH: 0.1              # Very thin
        }
        
        # Mechanical colors
        self.colors = {
            LineClassification.OUTER_BOUNDARY: 7,      # White/Black (visible)
            LineClassification.INNER_BOUNDARY: 7,      # White/Black
            LineClassification.DETAIL_LINE: 7,         # White/Black
            LineClassification.CONSTRUCTION: 4,        # Cyan (center lines)
            LineClassification.DIMENSION: 2,           # Yellow
            LineClassification.TEXT: 7,               # White/Black
            LineClassification.HATCH: 8               # Gray
        }
        
        # Mechanical line types
        self.line_types = {
            LineClassification.OUTER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.INNER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.DETAIL_LINE: LineType.CONTINUOUS,
            LineClassification.CONSTRUCTION: LineType.CENTER,
            LineClassification.DIMENSION: LineType.CONTINUOUS,
            LineClassification.TEXT: LineType.CONTINUOUS,
            LineClassification.HATCH: LineType.CONTINUOUS
        }


class ElectricalProfile(CADProfile):
    """Electrical CAD standards (IEEE/NFPA)"""
    
    def __init__(self):
        super().__init__("Electrical")
        
        # Electrical layer naming
        self.layers = {
            LineClassification.OUTER_BOUNDARY: "E-POWR-OTLN",
            LineClassification.INNER_BOUNDARY: "E-POWR-EQPM", 
            LineClassification.DETAIL_LINE: "E-LITE-CIRC",
            LineClassification.CONSTRUCTION: "E-GRID",
            LineClassification.DIMENSION: "E-ANNO-DIMS",
            LineClassification.TEXT: "E-ANNO-TEXT",
            LineClassification.HATCH: "E-POWR-PATT"
        }
        
        # Electrical line weights
        self.line_weights = {
            LineClassification.OUTER_BOUNDARY: 0.5,    # Medium-heavy
            LineClassification.INNER_BOUNDARY: 0.3,    # Medium
            LineClassification.DETAIL_LINE: 0.2,       # Light
            LineClassification.CONSTRUCTION: 0.13,     # Very light
            LineClassification.DIMENSION: 0.13,        # Very light
            LineClassification.TEXT: 0.18,             # Light
            LineClassification.HATCH: 0.1              # Very light
        }
        
        # Electrical colors
        self.colors = {
            LineClassification.OUTER_BOUNDARY: 1,      # Red (power)
            LineClassification.INNER_BOUNDARY: 3,      # Green (equipment)
            LineClassification.DETAIL_LINE: 6,         # Magenta (circuits)
            LineClassification.CONSTRUCTION: 8,        # Gray
            LineClassification.DIMENSION: 4,           # Cyan
            LineClassification.TEXT: 7,               # White/Black
            LineClassification.HATCH: 8               # Gray
        }
        
        # Electrical line types
        self.line_types = {
            LineClassification.OUTER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.INNER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.DETAIL_LINE: LineType.DASHED,
            LineClassification.CONSTRUCTION: LineType.CENTER,
            LineClassification.DIMENSION: LineType.CONTINUOUS,
            LineClassification.TEXT: LineType.CONTINUOUS,
            LineClassification.HATCH: LineType.CONTINUOUS
        }


class IllustrationProfile(CADProfile):
    """Clean line art illustration style - inspired by professional character illustrations"""
    
    def __init__(self):
        super().__init__("Illustration")
        
        # Simple, clean layer naming for illustrations
        self.layers = {
            LineClassification.OUTER_BOUNDARY: "IL-OUTLINE-MAIN",   # Main character/object outlines
            LineClassification.INNER_BOUNDARY: "IL-OUTLINE-FEAT",   # Facial features, clothing details
            LineClassification.DETAIL_LINE: "IL-DETAIL-FINE",      # Fine details, textures, small elements
            LineClassification.CONSTRUCTION: "IL-CONSTRUCT",        # Construction/guide lines
            LineClassification.DIMENSION: "IL-GUIDES",             # Reference guides
            LineClassification.TEXT: "IL-TEXT",                   # Text/annotations
            LineClassification.HATCH: "IL-PATTERN"                # Patterns/textures
        }
        
        # Line weights matching the illustration style - clean and consistent
        self.line_weights = {
            LineClassification.OUTER_BOUNDARY: 0.4,    # Medium weight for main outlines
            LineClassification.INNER_BOUNDARY: 0.25,   # Slightly thinner for internal features
            LineClassification.DETAIL_LINE: 0.15,      # Fine lines for details and textures
            LineClassification.CONSTRUCTION: 0.1,      # Very light construction lines
            LineClassification.DIMENSION: 0.1,         # Very light guides
            LineClassification.TEXT: 0.2,              # Medium-light for text
            LineClassification.HATCH: 0.12             # Light hatching/patterns
        }
        
        # Monochromatic black - clean illustration style
        self.colors = {
            LineClassification.OUTER_BOUNDARY: 7,      # Black
            LineClassification.INNER_BOUNDARY: 7,      # Black  
            LineClassification.DETAIL_LINE: 7,         # Black
            LineClassification.CONSTRUCTION: 8,        # Light gray (barely visible)
            LineClassification.DIMENSION: 8,           # Light gray
            LineClassification.TEXT: 7,               # Black
            LineClassification.HATCH: 7               # Black
        }
        
        # All continuous lines for clean illustration look
        self.line_types = {
            LineClassification.OUTER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.INNER_BOUNDARY: LineType.CONTINUOUS,
            LineClassification.DETAIL_LINE: LineType.CONTINUOUS,
            LineClassification.CONSTRUCTION: LineType.CONTINUOUS,
            LineClassification.DIMENSION: LineType.CONTINUOUS,
            LineClassification.TEXT: LineType.CONTINUOUS,
            LineClassification.HATCH: LineType.CONTINUOUS
        }


class LineClassifier:
    """Intelligent line classification based on geometric properties"""
    
    def __init__(self):
        self.classification_rules = self._init_classification_rules()
    
    def _init_classification_rules(self):
        """Initialize geometric classification rules"""
        return {
            'area_threshold_large': 0.4,    # >40% of total area = outer boundary (more sensitive for illustrations)
            'area_threshold_small': 0.02,   # <2% of total area = detail (catch finer details)
            'length_threshold_long': 0.7,   # >70% of max length = construction
            'complexity_threshold': 0.25,   # Lower threshold for better illustration handling
            'perimeter_ratio_high': 0.3,    # High perimeter-to-area ratio = detail lines
        }
    
    def classify_contour(self, contour: np.ndarray, all_contours: List[np.ndarray], 
                        image_bounds: Tuple[int, int]) -> LineClassification:
        """Classify a single contour based on geometric properties"""
        
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate relative metrics
        total_area = sum(cv2.contourArea(c) for c in all_contours)
        area_ratio = area / total_area if total_area > 0 else 0
        
        img_diagonal = np.sqrt(image_bounds[0]**2 + image_bounds[1]**2)
        length_ratio = max(w, h) / img_diagonal
        
        # Complexity metric (perimeter² / area)
        complexity = (perimeter**2) / area if area > 0 else float('inf')
        
        # Classification logic
        if area_ratio > self.classification_rules['area_threshold_large']:
            return LineClassification.OUTER_BOUNDARY
        elif area_ratio < self.classification_rules['area_threshold_small']:
            return LineClassification.DETAIL_LINE
        elif length_ratio > self.classification_rules['length_threshold_long']:
            return LineClassification.CONSTRUCTION
        elif complexity > self.classification_rules['complexity_threshold']:
            return LineClassification.HATCH
        else:
            return LineClassification.INNER_BOUNDARY
    
    def classify_contours(self, contours: List[np.ndarray], 
                         image_bounds: Tuple[int, int]) -> List[LineClassification]:
        """Classify all contours in a list"""
        return [self.classify_contour(contour, contours, image_bounds) 
                for contour in contours]


class CADStyleManager:
    """Manages CAD styling profiles and applies them to DWG generation"""
    
    def __init__(self):
        self.profiles = {
            'architectural': ArchitecturalProfile(),
            'mechanical': MechanicalProfile(), 
            'electrical': ElectricalProfile(),
            'illustration': IllustrationProfile()
        }
        self.current_profile = 'illustration'  # Default to illustration style
        self.classifier = LineClassifier()
    
    def set_profile(self, profile_name: str):
        """Set the active styling profile"""
        if profile_name in self.profiles:
            self.current_profile = profile_name
        else:
            raise ValueError(f"Unknown profile: {profile_name}")
    
    def get_profile(self) -> CADProfile:
        """Get the current active profile"""
        return self.profiles[self.current_profile]
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available profile names"""
        return list(self.profiles.keys())
    
    def classify_and_style_contours(self, contours: List[np.ndarray], 
                                  image_bounds: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Classify contours and apply styling"""
        classifications = self.classifier.classify_contours(contours, image_bounds)
        profile = self.get_profile()
        
        styled_contours = []
        for i, (contour, classification) in enumerate(zip(contours, classifications)):
            style = profile.get_layer_style(classification)
            styled_contours.append({
                'contour': contour,
                'classification': classification,
                'style': style,
                'index': i
            })
        
        return styled_contours
    
    def create_styled_dwg(self, contours: List[np.ndarray], output_path: str,
                         width: int, height: int) -> None:
        """Create a DWG file with professional styling applied"""
        import ezdxf
        
        # Classify and style contours
        styled_contours = self.classify_and_style_contours(contours, (width, height))
        
        # Create DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Create layers with proper styling
        layers_created = set()
        for styled_contour in styled_contours:
            style = styled_contour['style']
            layer_name = style['layer_name']
            
            if layer_name not in layers_created:
                layer = doc.layers.new(layer_name)
                layer.color = style['color']
                # Convert mm to 0.01mm units, but ensure reasonable range
                lineweight_units = max(1, min(200, int(style['lineweight'] * 100)))
                layer.lineweight = lineweight_units
                print(f"Created layer {layer_name}: color={style['color']}, lineweight={lineweight_units}")
                layers_created.add(layer_name)
        
        # Add contours to appropriate layers
        for styled_contour in styled_contours:
            contour = styled_contour['contour']
            style = styled_contour['style']
            
            # Convert contour points
            points = []
            for point in contour:
                x, y = point[0]
                # Convert from image coordinates to DXF coordinates
                dxf_x = x - width / 2
                dxf_y = (height - y) - height / 2
                points.append((dxf_x, dxf_y))
            
            # Close the polyline if needed
            if len(points) > 2 and points[0] != points[-1]:
                points.append(points[0])
            
            # Create polyline
            if len(points) >= 2:
                polyline = msp.add_lwpolyline(points)
                polyline.dxf.layer = style['layer_name']
        
        # Save the DWG file
        doc.saveas(output_path)
        
        return styled_contours  # Return for statistics/preview


# Color mapping for preview (AutoCAD Color Index to RGB)
ACI_COLORS = {
    1: (255, 0, 0),      # Red
    2: (255, 255, 0),    # Yellow  
    3: (0, 255, 0),      # Green
    4: (0, 255, 255),    # Cyan
    5: (0, 0, 255),      # Blue
    6: (255, 0, 255),    # Magenta
    7: (255, 255, 255),  # White
    8: (128, 128, 128),  # Gray
}

def aci_to_rgb(color_index: int) -> Tuple[int, int, int]:
    """Convert AutoCAD Color Index to RGB tuple"""
    return ACI_COLORS.get(color_index, (255, 255, 255))  # Default to white