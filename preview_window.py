#!/usr/bin/env python3
"""
Live Preview Window for Vectoriser Development

A GUI application that provides real-time preview of the image-to-DWG conversion process
with adjustable parameters and multi-stage visualization.

Author: AI Assistant
Dependencies: tkinter, opencv-python, numpy, PIL, matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import os
from pathlib import Path

# Import your existing converter
from image_to_dwg import ImageToDwgConverter


class VectoriserPreview:
    """
    A live preview window for testing and developing vectoriser features.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vectoriser Live Preview")
        self.root.geometry("1400x900")
        
        # Initialize converter
        self.converter = None
        self.current_image = None
        self.current_mask = None
        self.current_processed = None
        self.current_edges = None
        self.current_contours = None
        
        # Processing parameters
        self.params = {
            'canny_low': tk.IntVar(value=50),
            'canny_high': tk.IntVar(value=150),
            'min_len': tk.IntVar(value=25),
            'epsilon': tk.DoubleVar(value=1.5),
            'blur_kernel': tk.IntVar(value=15),
            'auto_update': tk.BooleanVar(value=True)
        }
        
        # Bind parameter changes to auto-update
        for param in self.params.values():
            if isinstance(param, (tk.IntVar, tk.DoubleVar)):
                param.trace('w', self.on_parameter_change)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control Panel (Left side)
        self.create_control_panel(main_frame)
        
        # Preview Area (Right side)
        self.create_preview_area(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to start")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.Frame(parent, padding="5")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save DWG", command=self.save_dwg).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Preview", command=self.export_preview).pack(fill=tk.X, pady=2)
        
        # Processing parameters
        params_frame = ttk.LabelFrame(control_frame, text="Processing Parameters", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Canny parameters
        ttk.Label(params_frame, text="Canny Low Threshold").pack(anchor=tk.W)
        canny_low_scale = ttk.Scale(params_frame, from_=10, to=200, variable=self.params['canny_low'], orient=tk.HORIZONTAL)
        canny_low_scale.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(params_frame, textvariable=self.params['canny_low']).pack(anchor=tk.E)
        
        ttk.Label(params_frame, text="Canny High Threshold").pack(anchor=tk.W)
        canny_high_scale = ttk.Scale(params_frame, from_=50, to=400, variable=self.params['canny_high'], orient=tk.HORIZONTAL)
        canny_high_scale.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(params_frame, textvariable=self.params['canny_high']).pack(anchor=tk.E)
        
        # Contour parameters
        ttk.Label(params_frame, text="Min Contour Length").pack(anchor=tk.W)
        min_len_scale = ttk.Scale(params_frame, from_=5, to=100, variable=self.params['min_len'], orient=tk.HORIZONTAL)
        min_len_scale.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(params_frame, textvariable=self.params['min_len']).pack(anchor=tk.E)
        
        ttk.Label(params_frame, text="Epsilon (Simplification)").pack(anchor=tk.W)
        epsilon_scale = ttk.Scale(params_frame, from_=0.1, to=10.0, variable=self.params['epsilon'], orient=tk.HORIZONTAL)
        epsilon_scale.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(params_frame, textvariable=self.params['epsilon']).pack(anchor=tk.E)
        
        # Preprocessing parameters
        ttk.Label(params_frame, text="Blur Kernel Size").pack(anchor=tk.W)
        blur_scale = ttk.Scale(params_frame, from_=3, to=31, variable=self.params['blur_kernel'], orient=tk.HORIZONTAL)
        blur_scale.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(params_frame, textvariable=self.params['blur_kernel']).pack(anchor=tk.E)
        
        # Auto-update checkbox
        ttk.Checkbutton(params_frame, text="Auto Update", variable=self.params['auto_update']).pack(anchor=tk.W, pady=5)
        
        # Manual update button
        ttk.Button(params_frame, text="Update Preview", command=self.update_preview).pack(fill=tk.X, pady=5)
        
        # Processing stages
        stages_frame = ttk.LabelFrame(control_frame, text="Processing Stages", padding="5")
        stages_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stage_buttons = {}
        stages = ["Original", "Mask", "Processed", "Edges", "Contours"]
        for stage in stages:
            btn = ttk.Button(stages_frame, text=stage, command=lambda s=stage: self.show_stage(s))
            btn.pack(fill=tk.X, pady=2)
            self.stage_buttons[stage] = btn
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
    def create_preview_area(self, parent):
        """Create the main preview area"""
        preview_frame = ttk.Frame(parent, padding="5")
        preview_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure for image display
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Load an image to start")
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, preview_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for zoom/pan
        toolbar_frame = ttk.Frame(preview_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def load_image(self):
        """Load an image for processing"""
        filetypes = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.gif'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title='Select an image',
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load image with OpenCV
                self.current_image = cv2.imread(filename)
                if self.current_image is None:
                    messagebox.showerror("Error", f"Could not load image: {filename}")
                    return
                
                # Initialize converter if needed
                if self.converter is None:
                    self.status_var.set("Loading AI model...")
                    self.root.update()
                    self.converter = ImageToDwgConverter()
                
                self.status_var.set(f"Loaded: {Path(filename).name}")
                self.show_stage("Original")
                self.update_preview()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def on_parameter_change(self, *args):
        """Called when parameters change"""
        if self.params['auto_update'].get() and self.current_image is not None:
            # Use a timer to avoid too frequent updates
            if hasattr(self, '_update_timer'):
                self.root.after_cancel(self._update_timer)
            self._update_timer = self.root.after(500, self.update_preview)
    
    def update_preview(self):
        """Update the preview with current parameters"""
        if self.current_image is None:
            return
        
        try:
            self.status_var.set("Processing...")
            self.root.update()
            
            # Create config from current parameters
            config = {
                'canny_low': self.params['canny_low'].get(),
                'canny_high': self.params['canny_high'].get(),
                'min_len': self.params['min_len'].get(),
                'epsilon': self.params['epsilon'].get()
            }
            
            # Process in stages
            self.current_mask = self.converter._get_object_mask(self.current_image)
            self.current_processed = self.converter._preprocess_and_mask(self.current_image, self.current_mask)
            self.current_edges = self.converter._detect_edges(self.current_processed, config)
            self.current_contours = self.converter._vectorize_contours(self.current_edges, config)
            
            # Update statistics
            self.update_statistics()
            
            # Refresh current view
            current_stage = getattr(self, 'current_stage', 'Original')
            self.show_stage(current_stage)
            
            self.status_var.set("Preview updated")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Processing Error", str(e))
    
    def show_stage(self, stage):
        """Display a specific processing stage"""
        self.current_stage = stage
        
        if self.current_image is None:
            return
        
        # Clear the plot
        self.ax.clear()
        self.ax.set_title(f"Stage: {stage}")
        self.ax.axis('off')
        
        if stage == "Original":
            # Show original image (convert BGR to RGB)
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.ax.imshow(rgb_image)
            
        elif stage == "Mask" and self.current_mask is not None:
            self.ax.imshow(self.current_mask, cmap='gray')
            
        elif stage == "Processed" and self.current_processed is not None:
            self.ax.imshow(self.current_processed, cmap='gray')
            
        elif stage == "Edges" and self.current_edges is not None:
            self.ax.imshow(self.current_edges, cmap='gray')
            
        elif stage == "Contours" and self.current_contours is not None:
            # Create a visualization of contours
            if self.current_image is not None:
                contour_image = self.current_image.copy()
                
                # Draw contours with different colors
                primary_contour = None
                max_area = 0
                
                for contour in self.current_contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        primary_contour = contour
                
                # Draw detail contours in white
                for contour in self.current_contours:
                    if not np.array_equal(contour, primary_contour):
                        cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), 2)
                
                # Draw primary contour in red
                if primary_contour is not None:
                    cv2.drawContours(contour_image, [primary_contour], -1, (0, 0, 255), 3)
                
                rgb_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
                self.ax.imshow(rgb_image)
        
        self.canvas.draw()
    
    def update_statistics(self):
        """Update the statistics panel"""
        if self.current_image is None:
            return
        
        stats = []
        h, w = self.current_image.shape[:2]
        stats.append(f"Image Size: {w}x{h}")
        
        if self.current_mask is not None:
            mask_area = np.sum(self.current_mask > 0)
            mask_percentage = (mask_area / (h * w)) * 100
            stats.append(f"Mask Area: {mask_percentage:.1f}%")
        
        if self.current_edges is not None:
            edge_pixels = np.sum(self.current_edges > 0)
            stats.append(f"Edge Pixels: {edge_pixels}")
        
        if self.current_contours is not None:
            stats.append(f"Contours Found: {len(self.current_contours)}")
            
            if self.current_contours:
                # Find primary contour (largest area)
                areas = [cv2.contourArea(c) for c in self.current_contours]
                max_area_idx = np.argmax(areas)
                stats.append(f"Primary Contour Area: {areas[max_area_idx]:.0f}")
                stats.append(f"Total Contour Points: {sum(len(c) for c in self.current_contours)}")
        
        # Add current parameters
        stats.append("\nCurrent Parameters:")
        stats.append(f"Canny Low: {self.params['canny_low'].get()}")
        stats.append(f"Canny High: {self.params['canny_high'].get()}")
        stats.append(f"Min Length: {self.params['min_len'].get()}")
        stats.append(f"Epsilon: {self.params['epsilon'].get():.2f}")
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, "\n".join(stats))
    
    def save_dwg(self):
        """Save the current result as DWG"""
        if self.current_contours is None:
            messagebox.showwarning("Warning", "No processed data to save. Process an image first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".dwg",
            filetypes=[('DWG files', '*.dwg'), ('All files', '*.*')]
        )
        
        if filename:
            try:
                h, w = self.current_image.shape[:2]
                self.converter._create_dwg(self.current_contours, filename, w, h)
                messagebox.showinfo("Success", f"DWG saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save DWG: {str(e)}")
    
    def export_preview(self):
        """Export the current preview as an image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[('PNG files', '*.png'), ('JPEG files', '*.jpg'), ('All files', '*.*')]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Preview exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export preview: {str(e)}")
    
    def run(self):
        """Start the preview window"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = VectoriserPreview()
    app.run()


if __name__ == "__main__":
    main()