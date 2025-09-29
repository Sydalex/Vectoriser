#!/usr/bin/env python3
"""
Simple test version of the web preview to debug issues
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Vectoriser Test Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-area { padding: 20px; border: 2px dashed #ccc; text-align: center; margin: 20px 0; }
        .preview { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; border: 1px solid #ccc; }
        input[type="file"] { margin: 10px; }
        button { padding: 10px 20px; margin: 10px; background: #007AFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056CC; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Vectoriser Test Preview</h1>
        <div class="upload-area">
            <h3>Upload an Image</h3>
            <input type="file" id="imageInput" accept="image/*">
            <button onclick="uploadImage()">Upload & Process</button>
        </div>
        <div class="preview" id="preview">
            <p>No image uploaded yet</p>
        </div>
    </div>

    <script>
        function uploadImage() {
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            document.getElementById('preview').innerHTML = '<p>Processing...</p>';

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('preview').innerHTML = '<p>Error: ' + data.error + '</p>';
                } else {
                    document.getElementById('preview').innerHTML = 
                        '<h3>Uploaded: ' + data.filename + '</h3>' +
                        '<p>Size: ' + data.size + '</p>' +
                        '<img src="' + data.preview + '" alt="Preview">';
                }
            })
            .catch(error => {
                document.getElementById('preview').innerHTML = '<p>Upload failed: ' + error.message + '</p>';
            });
        }
    </script>
</body>
</html>'''

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Convert to RGB for display
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64 for display
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        preview_data = f"data:image/png;base64,{encoded}"
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': file.filename,
            'size': f"{image.shape[1]}x{image.shape[0]}",
            'preview': preview_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🧪 Starting Test Preview Server...")
    print("📱 Open your browser to: http://localhost:3001")
    app.run(debug=True, host='127.0.0.1', port=3001)