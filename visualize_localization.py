#!/usr/bin/env python3
"""
Interactive visualization of BrainQCNet artifact predictions with activation maps.
Shows slices with heatmap overlays and tight bounding boxes around artifact regions.
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

class ArtifactVisualizer:
    def __init__(self, output_dir, localization_dir):
        """
        output_dir: Path to out_brainqcnet/sub-XX/ses-XX/single_run/ (has table.csv)
        localization_dir: Path to directory with activation map visualizations
        """
        self.output_dir = output_dir
        self.localization_dir = localization_dir
        self.current_axis = 'x'
        self.current_index = 0
        self.axes_data = {}

        # Load predictions
        print("Loading predictions...")
        self.table = pd.read_csv(os.path.join(output_dir, "table.csv"))

        # Try to load localization results (with bbox coords)
        loc_csv = os.path.join(localization_dir, "localization_results.csv")
        if os.path.isfile(loc_csv):
            self.loc_df = pd.read_csv(loc_csv)
            print(f"Loaded localization results: {len(self.loc_df)} slices")
        else:
            self.loc_df = None
            print("Warning: localization_results.csv not found")

        # Organize by axis
        for axis in ['x', 'y', 'z']:
            axis_data = self.table[self.table['axis'] == axis].sort_values('slice').reset_index(drop=True)
            self.axes_data[axis] = axis_data

        print(f"\nSlices per axis:")
        for axis in ['x', 'y', 'z']:
            artifact_count = (self.axes_data[axis]['pred'] == 1).sum()
            total = len(self.axes_data[axis])
            print(f"  {axis.upper()}: {artifact_count}/{total} artifacts ({100*artifact_count/total:.1f}%)")

        # Setup figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('BrainQCNet Artifact Localization', fontsize=16, fontweight='bold')

        # Axes for image and heatmap
        self.ax_img = plt.subplot(1, 2, 1)
        self.ax_stats = plt.subplot(1, 2, 2)

        # Navigation buttons
        ax_prev = plt.axes([0.35, 0.05, 0.08, 0.05])
        ax_next = plt.axes([0.57, 0.05, 0.08, 0.05])
        ax_axis_x = plt.axes([0.1, 0.05, 0.06, 0.05])
        ax_axis_y = plt.axes([0.17, 0.05, 0.06, 0.05])
        ax_axis_z = plt.axes([0.24, 0.05, 0.06, 0.05])
        ax_next_artifact = plt.axes([0.75, 0.05, 0.15, 0.05])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_axis_x = Button(ax_axis_x, 'X-axis', color='lightblue')
        self.btn_axis_y = Button(ax_axis_y, 'Y-axis', color='lightgreen')
        self.btn_axis_z = Button(ax_axis_z, 'Z-axis', color='lightyellow')
        self.btn_next_artifact = Button(ax_next_artifact, 'Next Artifact →', color='lightcoral')

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        self.btn_axis_x.on_clicked(lambda e: self.change_axis('x'))
        self.btn_axis_y.on_clicked(lambda e: self.change_axis('y'))
        self.btn_axis_z.on_clicked(lambda e: self.change_axis('z'))
        self.btn_next_artifact.on_clicked(self.on_next_artifact)

        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()

    def _find_visualization(self, axis, slice_num):
        """Find the generated visualization image for a slice."""
        patterns = [
            f"{axis}_{slice_num:03d}_artifact.png",
            f"{axis}_{slice_num:03d}_clean.png",
        ]
        for pattern in patterns:
            path = os.path.join(self.localization_dir, pattern)
            if os.path.isfile(path):
                return path
        return None

    def _get_bbox(self, axis, slice_num):
        """Get bounding box coordinates from localization CSV."""
        if self.loc_df is None:
            return None
        row = self.loc_df[(self.loc_df['axis'] == axis) & (self.loc_df['slice'] == slice_num)]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        return (int(r['bbox_y1']), int(r['bbox_y2']), int(r['bbox_x1']), int(r['bbox_x2']))

    def change_axis(self, axis):
        self.current_axis = axis
        self.current_index = 0
        self.update_display()

    def on_prev(self, event):
        self.current_index = max(0, self.current_index - 1)
        self.update_display()

    def on_next(self, event):
        max_idx = len(self.axes_data[self.current_axis]) - 1
        self.current_index = min(max_idx, self.current_index + 1)
        self.update_display()

    def on_next_artifact(self, event):
        """Jump to the next artifact slice."""
        axis_data = self.axes_data[self.current_axis]
        for i in range(self.current_index + 1, len(axis_data)):
            if axis_data.iloc[i]['pred'] == 1:
                self.current_index = i
                self.update_display()
                return
        # Wrap to beginning
        for i in range(0, self.current_index):
            if axis_data.iloc[i]['pred'] == 1:
                self.current_index = i
                self.update_display()
                return

    def on_key(self, event):
        if event.key == 'left':
            self.on_prev(None)
        elif event.key == 'right':
            self.on_next(None)
        elif event.key == ' ':
            self.on_next_artifact(None)
        elif event.key in ['x', 'y', 'z']:
            self.change_axis(event.key)

    def update_display(self):
        axis_data = self.axes_data[self.current_axis]
        row = axis_data.iloc[self.current_index]

        slice_num = int(row['slice'])
        prediction = int(row['pred'])
        is_artifact = prediction == 1

        self.ax_img.clear()
        self.ax_stats.clear()

        # Load precomputed visualization
        viz_path = self._find_visualization(self.current_axis, slice_num)
        bbox = self._get_bbox(self.current_axis, slice_num)

        if viz_path and os.path.isfile(viz_path):
            img = np.array(Image.open(viz_path).convert('RGB'))
            self.ax_img.imshow(img)

            status = "ARTIFACT DETECTED" if is_artifact else "CLEAN"
            color = 'red' if is_artifact else 'green'

            title = f"{self.current_axis.upper()}-axis, Slice {slice_num}\n{status}"
            if is_artifact and bbox is not None:
                bbox_h = bbox[1] - bbox[0]
                bbox_w = bbox[3] - bbox[2]
                title += f"\nArtifact region: {bbox_w}x{bbox_h} px"

            self.ax_img.set_title(title, fontsize=13, fontweight='bold', color=color)
        else:
            self.ax_img.text(0.5, 0.5, "Visualization not found\nRun generate_activation_maps.py first",
                            ha='center', va='center', fontsize=12)
            self.ax_img.set_title(f"{self.current_axis.upper()}-axis, Slice {slice_num}")

        self.ax_img.axis('off')

        # Statistics panel
        total_slices = len(axis_data)
        artifact_count = (axis_data['pred'] == 1).sum()
        clean_count = total_slices - artifact_count

        # Overall stats
        total_all = len(self.table)
        artifact_all = (self.table['pred'] == 1).sum()

        stats_text = (
            f"NAVIGATION:\n"
            f"  ← → : Previous/Next slice\n"
            f"  Space: Jump to next artifact\n"
            f"  X/Y/Z: Switch axis\n"
            f"  Buttons: Click to navigate\n\n"
            f"CURRENT AXIS ({self.current_axis.upper()}):\n"
            f"  Total slices: {total_slices}\n"
            f"  Artifacts: {artifact_count} ({100*artifact_count/total_slices:.1f}%)\n"
            f"  Clean: {clean_count} ({100*clean_count/total_slices:.1f}%)\n\n"
            f"OVERALL ({total_all} slices):\n"
            f"  Artifacts: {artifact_all} ({100*artifact_all/total_all:.1f}%)\n\n"
            f"CURRENT SLICE:\n"
            f"  Position: {self.current_index + 1} / {total_slices}\n"
            f"  Slice #: {slice_num}\n"
            f"  Prediction: {prediction}\n"
            f"  Status: {'ARTIFACT' if is_artifact else 'CLEAN'}\n\n"
            f"LEGEND:\n"
            f"  Yellow box: Artifact region\n"
            f"  Heatmap: Model attention\n"
            f"    Red = High activation\n"
            f"    Blue = Low activation"
        )

        self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes,
                          ha='left', va='top', fontsize=10, family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax_stats.axis('off')
        self.ax_stats.set_title("Statistics & Info", fontsize=12, fontweight='bold')

        self.fig.canvas.draw_idle()

    def run(self):
        print("\n" + "="*60)
        print("BRAINQCNET ARTIFACT LOCALIZATION VIEWER")
        print("="*60)
        print("\nControls:")
        print("  ← →         Navigate slices")
        print("  Space       Jump to next artifact")
        print("  X/Y/Z       Switch anatomical axis")
        print("  Buttons     Click to navigate")
        print("\nVisualization:")
        print("  - Yellow bounding box: Actual artifact region")
        print("  - Heatmap overlay: Model's attention map")
        print("  - Red = High artifact confidence")
        print("  - Blue = Low artifact confidence")
        print("="*60 + "\n")
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_localization.py <output_dir> [localization_dir]")
        print("\nExample:")
        print("  python visualize_localization.py out_brainqcnet/sub-08/ses-single_session/single_run out_brainqcnet/sub-08/localization")
        sys.exit(1)

    output_dir = sys.argv[1]
    localization_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(output_dir, "..", "..", "..", "localization")
    localization_dir = os.path.abspath(localization_dir)

    if not os.path.isdir(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    if not os.path.isdir(localization_dir):
        print(f"Error: Localization directory not found: {localization_dir}")
        print("\nFirst, generate activation maps by running:")
        print("  python generate_activation_maps.py \\")
        print(f"    --modeldir saved_models/saved_models/resnet152/19112020/ \\")
        print(f"    --pngdir out_brainqcnet/sub-08/pngs \\")
        print(f"    --tablecsv {os.path.join(output_dir, 'table.csv')} \\")
        print(f"    --outdir {localization_dir}")
        sys.exit(1)

    viz = ArtifactVisualizer(output_dir, localization_dir)
    viz.run()


if __name__ == '__main__':
    main()
