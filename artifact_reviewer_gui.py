#!/usr/bin/env python3
"""
PyQt6 GUI for interactive BrainQCNet artifact review.

Features:
- Browse all slices across x/y/z axes
- Toggle activation heatmap overlay on/off
- Toggle bounding box on/off
- Adjustable heatmap opacity
- Jump to next/previous artifact
- Keyboard shortcuts
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QComboBox, QFileDialog,
    QGroupBox, QStatusBar, QMessageBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QFont


def numpy_to_qpixmap(img):
    """Convert a numpy RGB uint8 array to a QPixmap."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ArtifactReviewer(QMainWindow):
    def __init__(self, output_dir, png_dir, localization_dir):
        super().__init__()
        self.output_dir = output_dir
        self.png_dir = png_dir
        self.localization_dir = localization_dir
        self.current_axis = 'x'
        self.current_index = 0

        # Toggle state
        self.show_heatmap = True
        self.show_bbox = True
        self.heatmap_opacity = 0.4

        self._load_data()
        self._build_ui()
        self._install_shortcuts()
        self.update_display()

    # ---------- Data loading ---------- #

    def _load_data(self):
        table_csv = os.path.join(self.output_dir, "table.csv")
        self.table = pd.read_csv(table_csv)

        loc_csv = os.path.join(self.localization_dir, "localization_results.csv")
        if os.path.isfile(loc_csv):
            self.loc_df = pd.read_csv(loc_csv)
        else:
            self.loc_df = None

        self.axes_data = {}
        for axis in ['x', 'y', 'z']:
            self.axes_data[axis] = (
                self.table[self.table['axis'] == axis]
                .sort_values('slice')
                .reset_index(drop=True)
            )

    def _find_png(self, axis, slice_num):
        patterns = [
            f"*_T1w_{axis}_{slice_num:03d}.png",
            f"*_{axis}_{slice_num:03d}.png",
            f"*_{axis}_{slice_num}.png",
        ]
        for pattern in patterns:
            matches = glob.glob(os.path.join(self.png_dir, pattern))
            if matches:
                return matches[0]
        return None

    def _find_viz(self, axis, slice_num):
        for suffix in ['artifact', 'clean']:
            path = os.path.join(self.localization_dir, f"{axis}_{slice_num:03d}_{suffix}.png")
            if os.path.isfile(path):
                return path
        return None

    def _get_bbox(self, axis, slice_num):
        if self.loc_df is None:
            return None
        row = self.loc_df[(self.loc_df['axis'] == axis) & (self.loc_df['slice'] == slice_num)]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        return (int(r['bbox_y1']), int(r['bbox_y2']), int(r['bbox_x1']), int(r['bbox_x2']))

    # ---------- UI ---------- #

    def _build_ui(self):
        self.setWindowTitle("BrainQCNet Artifact Reviewer")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # --- Image panel (left) --- #
        image_panel = QVBoxLayout()
        self.image_label = QLabel("Loading...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setFrameStyle(QFrame.Shape.Box)
        self.image_label.setStyleSheet("background-color: #222;")
        image_panel.addWidget(self.image_label, stretch=1)

        # Slice slider
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self.on_slider_change)
        slider_row.addWidget(self.slice_slider, stretch=1)
        self.slice_label = QLabel("0/0")
        self.slice_label.setMinimumWidth(80)
        slider_row.addWidget(self.slice_label)
        image_panel.addLayout(slider_row)

        # Navigation buttons
        nav_row = QHBoxLayout()
        btn_prev = QPushButton("◀ Previous (←)")
        btn_prev.clicked.connect(self.on_prev)
        btn_next = QPushButton("Next (→) ▶")
        btn_next.clicked.connect(self.on_next)
        btn_prev_art = QPushButton("◀◀ Prev Artifact (Shift+←)")
        btn_prev_art.clicked.connect(self.on_prev_artifact)
        btn_next_art = QPushButton("Next Artifact (Space) ▶▶")
        btn_next_art.clicked.connect(self.on_next_artifact)
        nav_row.addWidget(btn_prev_art)
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        nav_row.addWidget(btn_next_art)
        image_panel.addLayout(nav_row)

        root_layout.addLayout(image_panel, stretch=3)

        # --- Controls panel (right) --- #
        controls_panel = QVBoxLayout()
        controls_panel.setSpacing(10)

        # Axis selector
        axis_group = QGroupBox("Axis")
        axis_layout = QHBoxLayout(axis_group)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["X (Sagittal)", "Y (Coronal)", "Z (Axial)"])
        self.axis_combo.currentIndexChanged.connect(self.on_axis_change)
        axis_layout.addWidget(self.axis_combo)
        controls_panel.addWidget(axis_group)

        # Overlay toggles
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout(overlay_group)

        self.chk_heatmap = QCheckBox("Show Activation Heatmap  (H)")
        self.chk_heatmap.setChecked(True)
        self.chk_heatmap.stateChanged.connect(self.on_toggle_heatmap)
        overlay_layout.addWidget(self.chk_heatmap)

        self.chk_bbox = QCheckBox("Show Bounding Box  (B)")
        self.chk_bbox.setChecked(True)
        self.chk_bbox.stateChanged.connect(self.on_toggle_bbox)
        overlay_layout.addWidget(self.chk_bbox)

        # Opacity slider
        overlay_layout.addWidget(QLabel("Heatmap Opacity:"))
        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.valueChanged.connect(self.on_opacity_change)
        opacity_row.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("40%")
        self.opacity_label.setMinimumWidth(40)
        opacity_row.addWidget(self.opacity_label)
        overlay_layout.addLayout(opacity_row)

        controls_panel.addWidget(overlay_group)

        # Slice info
        self.info_group = QGroupBox("Current Slice")
        self.info_layout = QVBoxLayout(self.info_group)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setFont(QFont("Courier", 10))
        self.info_layout.addWidget(self.info_label)
        controls_panel.addWidget(self.info_group)

        # Axis stats
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setFont(QFont("Courier", 10))
        stats_layout.addWidget(self.stats_label)
        controls_panel.addWidget(stats_group)

        # Legend
        legend_group = QGroupBox("Legend & Shortcuts")
        legend_layout = QVBoxLayout(legend_group)
        legend_text = (
            "Yellow box = Artifact region\n"
            "Heatmap: Red = high activation\n"
            "         Blue = low activation\n"
            "\n"
            "Shortcuts:\n"
            "  ← →        Prev/next slice\n"
            "  Shift+←→   Prev/next artifact\n"
            "  Space      Next artifact\n"
            "  H          Toggle heatmap\n"
            "  B          Toggle bounding box\n"
            "  1/2/3      Switch X/Y/Z axis"
        )
        legend_label = QLabel(legend_text)
        legend_label.setFont(QFont("Courier", 9))
        legend_layout.addWidget(legend_label)
        controls_panel.addWidget(legend_group)

        controls_panel.addStretch()

        controls_widget = QWidget()
        controls_widget.setLayout(controls_panel)
        controls_widget.setMaximumWidth(380)
        root_layout.addWidget(controls_widget, stretch=1)

        self.setStatusBar(QStatusBar())

    def _install_shortcuts(self):
        shortcuts = [
            (Qt.Key.Key_Left, self.on_prev),
            (Qt.Key.Key_Right, self.on_next),
            ("Shift+Left", self.on_prev_artifact),
            ("Shift+Right", self.on_next_artifact),
            (Qt.Key.Key_Space, self.on_next_artifact),
            (Qt.Key.Key_H, lambda: self.chk_heatmap.toggle()),
            (Qt.Key.Key_B, lambda: self.chk_bbox.toggle()),
            (Qt.Key.Key_1, lambda: self.axis_combo.setCurrentIndex(0)),
            (Qt.Key.Key_2, lambda: self.axis_combo.setCurrentIndex(1)),
            (Qt.Key.Key_3, lambda: self.axis_combo.setCurrentIndex(2)),
        ]
        for key, fn in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fn)

    # ---------- Rendering ---------- #

    def _render_image(self):
        """Compose the image: base PNG + optional heatmap + optional bbox."""
        axis_data = self.axes_data[self.current_axis]
        if len(axis_data) == 0:
            return None, None

        row = axis_data.iloc[self.current_index]
        slice_num = int(row['slice'])
        is_artifact = int(row['pred']) == 1

        # Start from base PNG (clean brain image)
        base_path = self._find_png(self.current_axis, slice_num)
        if base_path is None:
            return None, (slice_num, is_artifact)

        base = np.array(Image.open(base_path).convert('RGB'))

        # Overlay heatmap if requested: derive from precomputed viz
        viz_path = self._find_viz(self.current_axis, slice_num)
        if self.show_heatmap and viz_path is not None and is_artifact:
            viz = np.array(Image.open(viz_path).convert('RGB'))
            if viz.shape[:2] != base.shape[:2]:
                viz = cv2.resize(viz, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
            # Precomputed viz was base blended with heatmap; extract heatmap by subtracting base
            # But base size may differ. Re-resize base to viz for subtraction, then blend fresh.
            base_resized = cv2.resize(base, (viz.shape[1], viz.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            # Approximate heatmap: viz * (1/(1-alpha_pre)) - base * (alpha_pre/(1-alpha_pre)) with alpha_pre=0.4
            # Simpler: just blend base with viz weighted by opacity slider
            alpha = self.heatmap_opacity
            blended = (1 - alpha) * base_resized + alpha * viz.astype(np.float32)
            composed = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            composed = base.copy()

        # Draw bounding box
        if self.show_bbox and is_artifact:
            bbox = self._get_bbox(self.current_axis, slice_num)
            if bbox is not None:
                lower_y, upper_y, lower_x, upper_x = bbox
                # bbox coords are in model img_size space (usually 224). Scale to composed size.
                model_size = 224
                h, w = composed.shape[:2]
                sx = w / model_size
                sy = h / model_size
                pt1 = (int(lower_x * sx), int(lower_y * sy))
                pt2 = (int((upper_x - 1) * sx), int((upper_y - 1) * sy))
                cv2.rectangle(composed, pt1, pt2, (255, 255, 0), 2)

        return composed, (slice_num, is_artifact)

    def update_display(self):
        axis_data = self.axes_data[self.current_axis]
        total = len(axis_data)

        # Keep slider in sync
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, total - 1))
        self.slice_slider.setValue(self.current_index)
        self.slice_slider.blockSignals(False)
        self.slice_label.setText(f"{self.current_index + 1}/{total}")

        composed, meta = self._render_image()

        if composed is not None:
            pixmap = numpy_to_qpixmap(composed)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText("Slice image not found")

        # Info
        if meta is not None:
            slice_num, is_artifact = meta
            bbox = self._get_bbox(self.current_axis, slice_num)
            status = "ARTIFACT" if is_artifact else "CLEAN"
            color = "#d9534f" if is_artifact else "#5cb85c"
            bbox_txt = ""
            if is_artifact and bbox is not None:
                bbox_h = bbox[1] - bbox[0]
                bbox_w = bbox[3] - bbox[2]
                bbox_txt = f"Region: {bbox_w} × {bbox_h} px\nAt: ({bbox[2]},{bbox[0]}) → ({bbox[3]},{bbox[1]})\n"
            self.info_label.setText(
                f"Axis: {self.current_axis.upper()}\n"
                f"Slice #: {slice_num}\n"
                f"Status: {status}\n"
                f"{bbox_txt}"
            )
            self.info_group.setStyleSheet(f"QGroupBox {{ color: {color}; font-weight: bold; }}")

        # Stats
        artifact_count = (axis_data['pred'] == 1).sum()
        total_all = len(self.table)
        artifact_all = (self.table['pred'] == 1).sum()
        self.stats_label.setText(
            f"{self.current_axis.upper()}-axis:\n"
            f"  {artifact_count}/{total} artifacts ({100*artifact_count/total:.1f}%)\n"
            f"  {total - artifact_count}/{total} clean ({100*(total-artifact_count)/total:.1f}%)\n"
            f"\nAll axes:\n"
            f"  {artifact_all}/{total_all} artifacts ({100*artifact_all/total_all:.1f}%)"
        )

        self.statusBar().showMessage(
            f"Axis {self.current_axis.upper()} | Slice {self.current_index + 1}/{total}"
        )

    # ---------- Event handlers ---------- #

    def on_axis_change(self, idx):
        self.current_axis = ['x', 'y', 'z'][idx]
        self.current_index = 0
        self.update_display()

    def on_slider_change(self, value):
        self.current_index = value
        self.update_display()

    def on_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def on_next(self):
        axis_data = self.axes_data[self.current_axis]
        if self.current_index < len(axis_data) - 1:
            self.current_index += 1
            self.update_display()

    def _jump_to_artifact(self, direction):
        axis_data = self.axes_data[self.current_axis]
        n = len(axis_data)
        if direction == 'next':
            indices = list(range(self.current_index + 1, n)) + list(range(0, self.current_index))
        else:
            indices = list(range(self.current_index - 1, -1, -1)) + list(range(n - 1, self.current_index, -1))
        for i in indices:
            if int(axis_data.iloc[i]['pred']) == 1:
                self.current_index = i
                self.update_display()
                return

    def on_next_artifact(self):
        self._jump_to_artifact('next')

    def on_prev_artifact(self):
        self._jump_to_artifact('prev')

    def on_toggle_heatmap(self, state):
        self.show_heatmap = self.chk_heatmap.isChecked()
        self.update_display()

    def on_toggle_bbox(self, state):
        self.show_bbox = self.chk_bbox.isChecked()
        self.update_display()

    def on_opacity_change(self, value):
        self.heatmap_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='BrainQCNet PyQt6 Artifact Reviewer')
    parser.add_argument('--output_dir', default='out_brainqcnet/sub-08/ses-single_session/single_run',
                        help='Directory containing table.csv')
    parser.add_argument('--png_dir', default='out_brainqcnet/sub-08/pngs',
                        help='Directory containing base PNG slices')
    parser.add_argument('--localization_dir', default='out_brainqcnet/sub-08/localization',
                        help='Directory with activation maps and localization_results.csv')
    args = parser.parse_args()

    for path, name in [
        (args.output_dir, 'output_dir'),
        (args.png_dir, 'png_dir'),
        (args.localization_dir, 'localization_dir'),
    ]:
        if not os.path.isdir(path):
            print(f"Error: {name} does not exist: {path}")
            sys.exit(1)

    app = QApplication(sys.argv)
    win = ArtifactReviewer(args.output_dir, args.png_dir, args.localization_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
