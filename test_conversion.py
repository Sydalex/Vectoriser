#!/usr/bin/env python3
"""
Test script for the Image to DWG Converter

This script demonstrates how to use the ImageToDwgConverter class programmatically.
"""

import os
import sys
from image_to_dwg import ImageToDwgConverter

def test_conversion():
    """Test the conversion with a sample configuration."""
    
    # Check if we have a test image
    test_image = "test_image.png"
    if not os.path.exists(test_image):
        print(f"Test image '{test_image}' not found.")
        print("Please place a test image named 'test_image.png' in the current directory.")
        return False
    
    # Configuration for testing
    config = {
        'canny_low': 50,
        'canny_high': 150,
        'min_len': 25,
        'epsilon': 1.5
    }
    
    output_file = "test_output.dwg"
    
    try:
        print("Creating ImageToDwgConverter instance...")
        converter = ImageToDwgConverter()
        
        print("Starting conversion...")
        converter.convert(test_image, output_file, config)
        
        if os.path.exists(output_file):
            print(f"✅ Conversion successful! Output saved to: {output_file}")
            return True
        else:
            print("❌ Conversion failed - output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_conversion()
    sys.exit(0 if success else 1)
