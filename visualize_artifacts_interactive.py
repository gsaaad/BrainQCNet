#!/usr/bin/env python3
"""
Interactive visualization of BrainQCNet artifact predictions.
Shows all slices (artifact and clean) with bounding boxes and heatmaps.
Navigate using arrow keys or buttons.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image
import cv2

class ArtifactVisualizer:
    def __init__(self, output_dir):
        """
        Initialize the visualizer.

        output_dir: Path to out_brainqcnet/sub-XX/ses-XX/single_run/
        """
        self.output_dir = output_dir
        self.current_axis = 'x'
        self.current_index = 0
        self.axes_data = {}

        # Load predictions
        print("Loading predictions...")
        self.table = pd.read_csv(os.path.join(output_dir, "table.csv"))

        # Organize by axis
        for axis in ['x', 'y', 'z']:
            axis_data = self.table[self.table['axis'] == axis].sort_values('slice').reset_index(drop=True)
            self.axes_data[axis] = axis_data

        print(f"\nSlices per axis:")
        for axis in ['x', 'y', 'z']:
            artifact_count = (self.axes_data[axis]['pred'] == 1).sum()
            total = len(self.axes_data[axis])
            print(f"  {axis.upper()}: {artifact_count}/{total} artifacts ({100*artifact_count/total:.1f}%)")

        # Find PNG directory (where med2image saved slices)
        self.png_dir = self._find_png_directory()
        if not self.png_dir:
            print("\nWarning: PNG directory not found.")
            print("To generate PNG slices, run:")
            print("  python generate_pngs_from_nifti.py BIDS_data/sub-08/anat/sub-08_T1w.nii.gz out_brainqcnet/sub-08/pngs/")
        else:
            png_count = len(glob.glob(os.path.join(self.png_dir, "*.png")))
            print(f"\nPNG directory found: {self.png_dir}")
            print(f"  Total PNG files: {png_count}")

        # Setup figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'BrainQCNet Artifact Visualization - Sub-08', fontsize=16, fontweight='bold')

        # Axes for image and heatmap
        self.ax_img = plt.subplot(1, 2, 1)
        self.ax_heat = plt.subplot(1, 2, 2)

        # Navigation buttons
        ax_prev = plt.axes([0.35, 0.05, 0.08, 0.05])
        ax_next = plt.axes([0.57, 0.05, 0.08, 0.05])
        ax_axis_x = plt.axes([0.1, 0.05, 0.06, 0.05])
        ax_axis_y = plt.axes([0.17, 0.05, 0.06, 0.05])
        ax_axis_z = plt.axes([0.24, 0.05, 0.06, 0.05])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_axis_x = Button(ax_axis_x, 'X-axis', color='lightblue')
        self.btn_axis_y = Button(ax_axis_y, 'Y-axis', color='lightgreen')
        self.btn_axis_z = Button(ax_axis_z, 'Z-axis', color='lightyellow')

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        self.btn_axis_x.on_clicked(lambda e: self.change_axis('x'))
        self.btn_axis_y.on_clicked(lambda e: self.change_axis('y'))
        self.btn_axis_z.on_clicked(lambda e: self.change_axis('z'))

        # Connect keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()

    def _find_png_directory(self):
        """Find where PNGs are stored."""
        # Extract subject ID from output path (e.g., sub-08)
        parts = self.output_dir.split(os.sep)
        subject_id = [p for p in parts if p.startswith("sub-")]

        # Common locations
        possible_paths = [
            os.path.join(self.output_dir, "pngs"),
            os.path.join(self.output_dir, "converted_images"),
            os.path.join(os.path.dirname(self.output_dir), "pngs"),
            os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), "pngs"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.output_dir))), "pngs"),
        ]

        for path in possible_paths:
            if os.path.isdir(path):
                pngs = glob.glob(os.path.join(path, "*.png"))
                if pngs:
                    return path

        return None

    def _get_slice_image(self, axis, slice_num):
        """Load PNG for a given slice."""
        if not self.png_dir:
            return None

        # Try multiple naming conventions used by different tools
        patterns = [
            f"*_T1w_{axis}_{slice_num:03d}.png",  # Our generate_pngs script
            f"*_{axis}_{slice_num:03d}.png",
            f"*_{axis}_{slice_num}.png",
            f"*_{axis}{slice_num:03d}.png",
        ]

        for pattern in patterns:
            matches = glob.glob(os.path.join(self.png_dir, pattern))
            if matches:
                try:
                    img = Image.open(matches[0]).convert('RGB')
                    return np.array(img)
                except Exception as e:
                    print(f"Error loading {matches[0]}: {e}")

        return None

    def _create_heatmap_overlay(self, img, confidence):
        """Create a confidence heatmap overlay on the image."""
        if img is None:
            return None

        h, w = img.shape[:2]

        # Create a simple heatmap based on confidence
        # High confidence (0.8+) = red, medium = yellow, low = green
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)

        if confidence > 0.7:
            heatmap[:, :] = [255, 0, 0]  # Red for high artifact confidence
        elif confidence > 0.4:
            heatmap[:, :] = [255, 255, 0]  # Yellow for medium
        else:
            heatmap[:, :] = [0, 255, 0]  # Green for low

        # Blend with original image
        alpha = 0.3
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        return overlay

    def _draw_bounding_boxes(self, img, is_artifact):
        """Draw bounding box on artifact slices."""
        if img is None:
            return img

        img = img.copy()
        h, w = img.shape[:2]

        if is_artifact:
            # Draw red rectangle border
            thickness = 15
            color = (255, 0, 0)  # Red
            cv2.rectangle(img, (thickness, thickness), (w - thickness, h - thickness), color, thickness)

            # Draw "ARTIFACT" text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "ARTIFACT DETECTED", (50, 80), font, 2, (255, 0, 0), 3)
        else:
            # Draw green border for clean slices
            thickness = 10
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (thickness, thickness), (w - thickness, h - thickness), color, thickness)

            # Draw "CLEAN" text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "CLEAN", (50, 80), font, 2, (0, 255, 0), 3)

        return img

    def change_axis(self, axis):
        """Switch to a different axis."""
        self.current_axis = axis
        self.current_index = 0
        self.update_display()

    def on_prev(self, event):
        """Go to previous slice."""
        self.current_index = max(0, self.current_index - 1)
        self.update_display()

    def on_next(self, event):
        """Go to next slice."""
        max_idx = len(self.axes_data[self.current_axis]) - 1
        self.current_index = min(max_idx, self.current_index + 1)
        self.update_display()

    def on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'left':
            self.on_prev(None)
        elif event.key == 'right':
            self.on_next(None)
        elif event.key == 'x':
            self.change_axis('x')
        elif event.key == 'y':
            self.change_axis('y')
        elif event.key == 'z':
            self.change_axis('z')

    def update_display(self):
        """Update the visualization for current slice."""
        axis_data = self.axes_data[self.current_axis]
        row = axis_data.iloc[self.current_index]

        slice_num = int(row['slice'])
        prediction = int(row['pred'])
        is_artifact = prediction == 1

        # Load image
        img = self._get_slice_image(self.current_axis, slice_num)

        # Clear previous plots
        self.ax_img.clear()
        self.ax_heat.clear()

        # Left plot: Image with bounding box
        if img is not None:
            img_with_box = self._draw_bounding_boxes(img, is_artifact)
            self.ax_img.imshow(img_with_box)
            self.ax_img.set_title(
                f"{self.current_axis.upper()}-axis, Slice {slice_num}\n"
                f"Status: {'ARTIFACT' if is_artifact else 'CLEAN'}",
                fontsize=14, fontweight='bold',
                color='red' if is_artifact else 'green'
            )
        else:
            self.ax_img.text(0.5, 0.5, "PNG not found\n(med2image may not have run)",
                            ha='center', va='center', fontsize=12)
            self.ax_img.set_title(f"{self.current_axis.upper()}-axis, Slice {slice_num}")

        self.ax_img.axis('off')

        # Right plot: Confidence heatmap and statistics
        if img is not None:
            heatmap = self._create_heatmap_overlay(img, prediction)
            self.ax_heat.imshow(heatmap)

        # Display statistics
        total_slices = len(axis_data)
        artifact_count = (axis_data['pred'] == 1).sum()
        clean_count = total_slices - artifact_count

        stats_text = (
            f"Navigation: Arrow keys or buttons\n"
            f"Axis: X/Y/Z keys or buttons\n\n"
            f"{self.current_axis.upper()}-Axis Statistics:\n"
            f"  Total slices: {total_slices}\n"
            f"  Artifacts: {artifact_count} ({100*artifact_count/total_slices:.1f}%)\n"
            f"  Clean: {clean_count} ({100*clean_count/total_slices:.1f}%)\n\n"
            f"Current: Slice {self.current_index + 1} / {total_slices}\n"
            f"Prediction: {prediction}\n"
            f"Status: {'ARTIFACT ⚠️' if is_artifact else 'CLEAN ✓'}"
        )

        self.ax_heat.text(0.5, 0.5, stats_text, ha='center', va='center',
                         fontsize=11, family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax_heat.axis('off')
        self.ax_heat.set_title("Statistics & Heatmap", fontsize=12, fontweight='bold')

        plt.tight_layout()
        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive viewer."""
        print("\n" + "="*60)
        print("BRAINQCNET ARTIFACT VISUALIZER")
        print("="*60)
        print("\nControls:")
        print("  Arrow keys: Navigate slices (← previous, → next)")
        print("  X/Y/Z keys: Switch anatomical axis")
        print("  Buttons: Click to navigate or switch axis")
        print("\nColor coding:")
        print("  🔴 RED border = ARTIFACT detected")
        print("  🟢 GREEN border = CLEAN slice")
        print("="*60 + "\n")

        plt.show()


def main():
    if len(sys.argv) < 2:
        # Try default location
        output_dir = os.path.expanduser(
            "out_brainqcnet/sub-08/ses-single_session/single_run"
        )
        if not os.path.isdir(output_dir):
            print("Usage: python visualize_artifacts_interactive.py <output_dir>")
            print("\nExample:")
            print("  python visualize_artifacts_interactive.py out_brainqcnet/sub-08/ses-single_session/single_run")
            sys.exit(1)
    else:
        output_dir = sys.argv[1]

    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    if not os.path.isfile(os.path.join(output_dir, "table.csv")):
        print(f"Error: table.csv not found in {output_dir}")
        print("Make sure you've run the BrainQCNet pipeline first.")
        sys.exit(1)

    visualizer = ArtifactVisualizer(output_dir)
    visualizer.run()


if __name__ == '__main__':
    main()
