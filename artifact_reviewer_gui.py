#!/usr/bin/env python3
"""
PyQt6 GUI for interactive BrainQCNet artifact review.

Features:
- Browse all slices across x/y/z axes
- Toggle activation heatmap overlay on/off          (H)
- Toggle bounding box on/off                        (B)
- Adjustable heatmap opacity
- Live bbox confidence threshold slider (re-crops from cached activation)
- Jump to next/previous artifact                    (Space / Shift+←→)
- Human reviewer labels: Confirmed / False Positive / Uncertain  (C / F / U)
  · Auto-saved to review_labels.csv, persists across sessions
  · Color badge rendered on image; auto-advances after labeling
- 3-panel orthographic view (Sagittal/Coronal/Axial) with crosshair
  · Requires --nifti_path argument
- Side-by-side skull-strip comparison               (S)
  · Requires pre-generated skull-strip PNGs in <png_dir>_skull/
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import nibabel as nib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QComboBox,
    QGroupBox, QStatusBar, QFrame, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QFont


# ---------- Module-level helpers ---------- #

def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def find_high_activation_crop(activation_map: np.ndarray, percentile: int = 95):
    """Return (lower_y, upper_y, lower_x, upper_x) tight crop at given percentile."""
    threshold = np.percentile(activation_map, percentile)
    mask = (activation_map >= threshold).astype(np.uint8)
    lower_y = upper_y = lower_x = upper_x = 0
    for i in range(mask.shape[0]):
        if mask[i].max() > 0:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if mask[i].max() > 0:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if mask[:, j].max() > 0:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if mask[:, j].max() > 0:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def norm_to_uint8(data: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    clipped = np.clip(data, lo, hi)
    if hi - lo < 1e-6:
        return np.zeros_like(clipped, dtype=np.uint8)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def make_crosshair(img_gray: np.ndarray, cy: int, cx: int,
                  color=(0, 200, 255), thickness: int = 1) -> np.ndarray:
    """Draw crosshair on grayscale slice, return RGB array."""
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.uint8)
    cv2.line(rgb, (cx, 0), (cx, rgb.shape[0] - 1), color, thickness)
    cv2.line(rgb, (0, cy), (rgb.shape[1] - 1, cy), color, thickness)
    return rgb


LABEL_COLORS = {
    'confirmed':     '#d9534f',
    'false_positive': '#5cb85c',
    'uncertain':     '#f0ad4e',
    'unlabeled':     '#888888',
}
LABEL_DISPLAY = {
    'confirmed':     'Confirmed Artifact',
    'false_positive': 'False Positive',
    'uncertain':     'Uncertain',
    'unlabeled':     'Unlabeled',
}


class ArtifactReviewer(QMainWindow):
    def __init__(self, output_dir, png_dir, localization_dir,
                 nifti_path=None, skull_strip_path=None):
        super().__init__()
        self.output_dir = output_dir
        self.png_dir = png_dir
        self.localization_dir = localization_dir
        self.nifti_path = nifti_path
        self.skull_strip_path = skull_strip_path

        self.current_axis = 'x'
        self.current_index = 0

        # Overlay state
        self.show_heatmap = True
        self.show_bbox = True
        self.heatmap_opacity = 0.4
        self.bbox_percentile = 95
        self.show_skull_strip = False

        # Volume cache (loaded lazily on first ortho render)
        self._vol_raw = None
        self._vol_skull = None

        self._load_data()
        self._load_activation_cache()
        self._build_ui()
        self._install_shortcuts()
        self.update_display()

    # ---------- Data loading ---------- #

    def _load_data(self):
        self.table = pd.read_csv(os.path.join(self.output_dir, 'table.csv'))

        loc_csv = os.path.join(self.localization_dir, 'localization_results.csv')
        self.loc_df = pd.read_csv(loc_csv) if os.path.isfile(loc_csv) else None

        # Reviewer labels — load existing or bootstrap from table
        self.labels_csv = os.path.join(self.output_dir, 'review_labels.csv')
        if os.path.isfile(self.labels_csv):
            self.labels_df = pd.read_csv(self.labels_csv)
            # Ensure all slices present (handles new data added later)
            existing = set(zip(self.labels_df['axis'], self.labels_df['slice']))
            new_rows = []
            for _, row in self.table.iterrows():
                if (row['axis'], row['slice']) not in existing:
                    new_rows.append({'axis': row['axis'], 'slice': row['slice'],
                                     'pred': row['pred'], 'label': 'unlabeled', 'notes': ''})
            if new_rows:
                self.labels_df = pd.concat(
                    [self.labels_df, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            self.labels_df = self.table[['axis', 'slice', 'pred']].copy()
            self.labels_df['label'] = 'unlabeled'
            self.labels_df['notes'] = ''

        self.axes_data = {}
        for axis in ['x', 'y', 'z']:
            self.axes_data[axis] = (
                self.table[self.table['axis'] == axis]
                .sort_values('slice')
                .reset_index(drop=True)
            )

    def _load_activation_cache(self):
        """Load raw activation NPZ for live bbox threshold recompute (optional)."""
        npz = os.path.join(self.localization_dir, 'activation_maps.npz')
        self._act_cache = dict(np.load(npz, allow_pickle=False)) if os.path.isfile(npz) else {}

    # ---------- Slice / volume helpers ---------- #

    def _current_slice_num(self) -> int:
        return int(self.axes_data[self.current_axis].iloc[self.current_index]['slice'])

    def _current_pred(self) -> int:
        return int(self.axes_data[self.current_axis].iloc[self.current_index]['pred'])

    def _find_png(self, axis, slice_num):
        for pat in [f'*_T1w_{axis}_{slice_num:03d}.png',
                    f'*_{axis}_{slice_num:03d}.png',
                    f'*_{axis}_{slice_num}.png']:
            m = glob.glob(os.path.join(self.png_dir, pat))
            if m:
                return m[0]
        return None

    def _get_slice_from_vol(self, vol: np.ndarray, axis: str, slice_num: int) -> np.ndarray:
        """Extract one slice from a volume and apply the same rot90 as
        generate_pngs_from_nifti.py so the slice matches the base PNGs."""
        if axis == 'x':
            s = vol[min(slice_num, vol.shape[0] - 1), :, :]
        elif axis == 'y':
            s = vol[:, min(slice_num, vol.shape[1] - 1), :]
        else:
            s = vol[:, :, min(slice_num, vol.shape[2] - 1)]
        return np.rot90(s)

    def _find_viz(self, axis, slice_num):
        for suffix in ['artifact', 'clean']:
            p = os.path.join(self.localization_dir, f'{axis}_{slice_num:03d}_{suffix}.png')
            if os.path.isfile(p):
                return p
        return None

    def _get_stored_bbox(self, axis, slice_num):
        if self.loc_df is None:
            return None
        row = self.loc_df[(self.loc_df['axis'] == axis) & (self.loc_df['slice'] == slice_num)]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        return (int(r['bbox_y1']), int(r['bbox_y2']), int(r['bbox_x1']), int(r['bbox_x2']))

    def _get_loc_pred(self, axis, slice_num) -> int:
        """Pred stored by generate_activation_maps.py (may differ from table.csv).
        Returns -1 if not found. Used to guard heatmap display: the viz PNG only
        contains colormap content when the localization run also said artifact.
        """
        if self.loc_df is None:
            return -1
        row = self.loc_df[(self.loc_df['axis'] == axis) & (self.loc_df['slice'] == slice_num)]
        return int(row.iloc[0]['pred']) if len(row) > 0 else -1

    def _get_bbox(self, axis, slice_num):
        """Return bbox; recomputes live from cached activation if NPZ is present."""
        key = f'{axis}_{slice_num:03d}'
        if key in self._act_cache:
            return find_high_activation_crop(self._act_cache[key],
                                             percentile=self.bbox_percentile)
        return self._get_stored_bbox(axis, slice_num)

    def _get_volume(self):
        """Lazily load raw NIfTI volume as uint8 array (x, y, z)."""
        if self._vol_raw is None and self.nifti_path and os.path.isfile(self.nifti_path):
            img = nib.load(self.nifti_path)
            self._vol_raw = norm_to_uint8(img.get_fdata())
        return self._vol_raw

    def _get_skull_volume(self):
        """Lazily load skull-stripped NIfTI volume as uint8 array (x, y, z)."""
        if self._vol_skull is None and self.skull_strip_path and os.path.isfile(self.skull_strip_path):
            img = nib.load(self.skull_strip_path)
            self._vol_skull = norm_to_uint8(img.get_fdata())
        return self._vol_skull

    # ---------- Label helpers ---------- #

    def _get_label(self, axis, slice_num) -> str:
        row = self.labels_df[
            (self.labels_df['axis'] == axis) & (self.labels_df['slice'] == slice_num)]
        return str(row.iloc[0]['label']) if len(row) > 0 else 'unlabeled'

    def _set_label(self, axis, slice_num, label: str):
        mask = (self.labels_df['axis'] == axis) & (self.labels_df['slice'] == slice_num)
        if mask.any():
            self.labels_df.loc[mask, 'label'] = label
        else:
            new_row = pd.DataFrame([{'axis': axis, 'slice': slice_num,
                                      'pred': self._current_pred(),
                                      'label': label, 'notes': ''}])
            self.labels_df = pd.concat([self.labels_df, new_row], ignore_index=True)
        self.labels_df.to_csv(self.labels_csv, index=False)

    def _label_summary(self) -> str:
        artifact_slices = self.labels_df[self.labels_df['pred'] == 1]
        total_art = len(artifact_slices)
        labeled = (artifact_slices['label'] != 'unlabeled').sum()
        lines = [f'Reviewed: {labeled} / {total_art} artifact slices']
        counts = artifact_slices[artifact_slices['label'] != 'unlabeled']['label'].value_counts()
        for k, v in counts.items():
            lines.append(f'  {LABEL_DISPLAY.get(k, k)}: {v}')
        return '\n'.join(lines)

    # ---------- UI ---------- #

    def _build_ui(self):
        self.setWindowTitle('BrainQCNet Artifact Reviewer')
        self.resize(1500, 960)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(6)

        # ---- Left column: main image + ortho strip + controls ---- #
        left_col = QVBoxLayout()

        self.image_label = QLabel('Loading...')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 480)
        self.image_label.setFrameStyle(QFrame.Shape.Box)
        self.image_label.setStyleSheet('background-color: #1a1a1a;')
        left_col.addWidget(self.image_label, stretch=4)

        # Orthographic strip (sagittal / coronal / axial)
        self.ortho_label = QLabel('3D position view — pass --nifti_path to enable')
        self.ortho_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ortho_label.setFixedHeight(230)
        self.ortho_label.setFrameStyle(QFrame.Shape.Box)
        self.ortho_label.setStyleSheet('background-color: #111; color: #666;')
        left_col.addWidget(self.ortho_label)

        # Slice slider
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel('Slice:'))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self.on_slider_change)
        slider_row.addWidget(self.slice_slider, stretch=1)
        self.slice_label = QLabel('0/0')
        self.slice_label.setMinimumWidth(60)
        slider_row.addWidget(self.slice_label)
        left_col.addLayout(slider_row)

        # Navigation buttons
        nav_row = QHBoxLayout()
        for label, fn in [('◀◀ Prev Artifact', self.on_prev_artifact),
                          ('◀ Prev', self.on_prev),
                          ('Next ▶', self.on_next),
                          ('Next Artifact ▶▶', self.on_next_artifact)]:
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            nav_row.addWidget(btn)
        left_col.addLayout(nav_row)

        root.addLayout(left_col, stretch=3)

        # ---- Right column: tabbed panels ---- #
        tabs = QTabWidget()
        tabs.setMaximumWidth(400)
        tabs.addTab(self._build_controls_tab(), 'Controls')
        tabs.addTab(self._build_labels_tab(),   'Labels')
        tabs.addTab(self._build_stats_tab(),    'Stats')
        root.addWidget(tabs, stretch=1)

        self.setStatusBar(QStatusBar())

    def _build_controls_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        # Axis
        axis_group = QGroupBox('Anatomical Axis')
        axis_layout = QHBoxLayout(axis_group)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(['X (Sagittal)', 'Y (Coronal)', 'Z (Axial)'])
        self.axis_combo.currentIndexChanged.connect(self.on_axis_change)
        axis_layout.addWidget(self.axis_combo)
        layout.addWidget(axis_group)

        # Overlays
        ov_group = QGroupBox('Overlays')
        ov_layout = QVBoxLayout(ov_group)

        self.chk_heatmap = QCheckBox('Show Activation Heatmap  (H)')
        self.chk_heatmap.setChecked(True)
        self.chk_heatmap.stateChanged.connect(self.on_toggle_heatmap)
        ov_layout.addWidget(self.chk_heatmap)

        self.chk_bbox = QCheckBox('Show Bounding Box  (B)')
        self.chk_bbox.setChecked(True)
        self.chk_bbox.stateChanged.connect(self.on_toggle_bbox)
        ov_layout.addWidget(self.chk_bbox)

        ov_layout.addWidget(QLabel('Heatmap Opacity:'))
        op_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.valueChanged.connect(self.on_opacity_change)
        op_row.addWidget(self.opacity_slider)
        self.opacity_label = QLabel('40%')
        self.opacity_label.setMinimumWidth(40)
        op_row.addWidget(self.opacity_label)
        ov_layout.addLayout(op_row)

        ov_layout.addWidget(QLabel('BBox Confidence Threshold:'))
        th_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 99)
        self.threshold_slider.setValue(95)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        th_row.addWidget(self.threshold_slider)
        self.threshold_label = QLabel('95th %ile')
        self.threshold_label.setMinimumWidth(70)
        th_row.addWidget(self.threshold_label)
        ov_layout.addLayout(th_row)

        layout.addWidget(ov_group)

        # Comparison
        cmp_group = QGroupBox('Comparison')
        cmp_layout = QVBoxLayout(cmp_group)
        self.chk_skull = QCheckBox('Side-by-side skull strip  (S)')
        self.chk_skull.setChecked(False)
        self.chk_skull.stateChanged.connect(self.on_toggle_skull)
        cmp_layout.addWidget(self.chk_skull)
        skull_hint = QLabel('Requires --skull_strip_path <skull.nii.gz>\n'
                            'Slices are extracted on-the-fly from the volume.')
        skull_hint.setStyleSheet('color: #888; font-size: 9pt;')
        cmp_layout.addWidget(skull_hint)
        layout.addWidget(cmp_group)

        # Shortcuts legend
        leg_group = QGroupBox('Keyboard Shortcuts')
        leg_layout = QVBoxLayout(leg_group)
        leg_label = QLabel(
            '← →       Prev / next slice\n'
            'Shift+←→  Prev / next artifact\n'
            'Space     Next artifact\n'
            'H         Toggle heatmap\n'
            'B         Toggle bbox\n'
            'S         Toggle skull strip\n'
            'C         Confirm artifact\n'
            'F         Mark false positive\n'
            'U         Mark uncertain\n'
            '1/2/3     Switch X/Y/Z axis'
        )
        leg_label.setFont(QFont('Courier', 9))
        leg_layout.addWidget(leg_label)
        layout.addWidget(leg_group)

        layout.addStretch()
        return w

    def _build_labels_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        # Current slice label display
        self.label_info = QLabel('')
        self.label_info.setWordWrap(True)
        self.label_info.setFont(QFont('Courier', 11))
        self.label_info.setStyleSheet('padding: 8px; border-radius: 4px;')
        layout.addWidget(self.label_info)

        # Label buttons
        btn_group = QGroupBox('Reviewer Decision')
        btn_layout = QVBoxLayout(btn_group)
        for key, text, color in [
            ('confirmed',     'Confirmed Artifact  (C)', '#c0392b'),
            ('false_positive', 'False Positive  (F)',    '#27ae60'),
            ('uncertain',     'Uncertain  (U)',          '#d68910'),
            ('unlabeled',     'Clear Label',             '#555555'),
        ]:
            btn = QPushButton(text)
            btn.setStyleSheet(
                f'background-color: {color}; color: white; '
                f'font-weight: bold; padding: 8px; border-radius: 3px;')
            btn.clicked.connect(lambda checked, k=key: self._apply_label(k))
            btn_layout.addWidget(btn)
        layout.addWidget(btn_group)

        # Running summary
        self.review_summary = QLabel('')
        self.review_summary.setWordWrap(True)
        self.review_summary.setFont(QFont('Courier', 10))
        layout.addWidget(self.review_summary)

        save_btn = QPushButton('Save Labels to CSV')
        save_btn.clicked.connect(self._save_labels)
        layout.addWidget(save_btn)

        layout.addStretch()
        return w

    def _build_stats_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        self.stats_label = QLabel('')
        self.stats_label.setWordWrap(True)
        self.stats_label.setFont(QFont('Courier', 10))
        layout.addWidget(self.stats_label)

        self.slice_info_label = QLabel('')
        self.slice_info_label.setWordWrap(True)
        self.slice_info_label.setFont(QFont('Courier', 10))
        self.slice_info_label.setStyleSheet(
            'padding: 8px; background: #1e1e1e; color: #ddd; border-radius: 4px;')
        layout.addWidget(self.slice_info_label)

        layout.addStretch()
        return w

    def _install_shortcuts(self):
        shortcuts = [
            (Qt.Key.Key_Left,  self.on_prev),
            (Qt.Key.Key_Right, self.on_next),
            ('Shift+Left',     self.on_prev_artifact),
            ('Shift+Right',    self.on_next_artifact),
            (Qt.Key.Key_Space, self.on_next_artifact),
            (Qt.Key.Key_H, lambda: self.chk_heatmap.toggle()),
            (Qt.Key.Key_B, lambda: self.chk_bbox.toggle()),
            (Qt.Key.Key_S, lambda: self.chk_skull.toggle()),
            (Qt.Key.Key_C, lambda: self._apply_label('confirmed')),
            (Qt.Key.Key_F, lambda: self._apply_label('false_positive')),
            (Qt.Key.Key_U, lambda: self._apply_label('uncertain')),
            (Qt.Key.Key_1, lambda: self.axis_combo.setCurrentIndex(0)),
            (Qt.Key.Key_2, lambda: self.axis_combo.setCurrentIndex(1)),
            (Qt.Key.Key_3, lambda: self.axis_combo.setCurrentIndex(2)),
        ]
        for key, fn in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fn)

    # ---------- Rendering ---------- #

    def _render_main_image(self):
        """Compose main image: base PNG + optional skull-strip split
        + optional heatmap + optional bbox + label badge."""
        axis = self.current_axis
        slice_num = self._current_slice_num()
        is_artifact = self._current_pred() == 1

        base_path = self._find_png(axis, slice_num)
        if base_path is None:
            return None

        base = np.array(Image.open(base_path).convert('RGB'))
        orig_w = base.shape[1]   # track before any concatenation

        # Side-by-side skull-strip comparison.
        # Slices are extracted on-the-fly from the skull-stripped NIfTI
        # (no pre-generated PNGs needed; requires --skull_strip_path).
        skull_added = False
        if self.show_skull_strip:
            skull_vol = self._get_skull_volume()
            if skull_vol is not None:
                skull_slice = self._get_slice_from_vol(skull_vol, axis, slice_num)
                skull_rgb = np.stack([skull_slice, skull_slice, skull_slice], axis=-1)
                skull_rgb = cv2.resize(skull_rgb, (orig_w, base.shape[0]),
                                       interpolation=cv2.INTER_AREA)
                divider = np.full((base.shape[0], 2, 3), 80, dtype=np.uint8)
                base = np.concatenate([base, divider, skull_rgb], axis=1)
                skull_added = True

        composed = base.copy()
        # draw_w is the width of the left (raw) panel only
        draw_w = orig_w  # same whether skull panel is present or not

        # Heatmap overlay (left half only when skull-strip is shown).
        # The viz PNG only contains actual colormap content when generate_activation_maps.py
        # also classified this slice as artifact (loc_pred == 1). For clean-classified
        # slices the saved PNG is just the original image, so skip blending.
        viz_path = self._find_viz(axis, slice_num)
        loc_pred = self._get_loc_pred(axis, slice_num)
        if self.show_heatmap and viz_path and is_artifact and loc_pred == 1:
            viz = np.array(Image.open(viz_path).convert('RGB'))
            viz = cv2.resize(viz, (draw_w, composed.shape[0]), interpolation=cv2.INTER_CUBIC)
            alpha = self.heatmap_opacity
            left = composed[:, :draw_w, :].astype(np.float32)
            blended = np.clip((1 - alpha) * left + alpha * viz.astype(np.float32),
                              0, 255).astype(np.uint8)
            composed[:, :draw_w, :] = blended

        # Bounding box (left half only)
        if self.show_bbox and is_artifact:
            bbox = self._get_bbox(axis, slice_num)
            if bbox is not None:
                model_size = 224
                h = composed.shape[0]
                sx = draw_w / model_size
                sy = h / model_size
                lower_y, upper_y, lower_x, upper_x = bbox
                pt1 = (int(lower_x * sx), int(lower_y * sy))
                pt2 = (int((upper_x - 1) * sx), int((upper_y - 1) * sy))
                cv2.rectangle(composed, pt1, pt2, (255, 255, 0), 2)

        # Label badge at top of image
        label = self._get_label(axis, slice_num)
        if label != 'unlabeled':
            badge_colors = {
                'confirmed':      (192, 57,  43),
                'false_positive': (39,  174, 96),
                'uncertain':      (214, 137, 16),
            }
            color = badge_colors.get(label, (100, 100, 100))
            cv2.rectangle(composed, (0, 0), (composed.shape[1], 34), color, -1)
            cv2.putText(composed, LABEL_DISPLAY.get(label, label).upper(),
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return composed

    def _render_ortho_panel(self):
        """Render a 3-panel ortho strip (Sagittal/Coronal/Axial) with crosshair.

        All three panels use _get_slice_from_vol so the orientation exactly
        matches the main base PNGs (same rot90 applied).
        Crosshair coordinates are computed in POST-rotation space.
        """
        vol = self._get_volume()
        if vol is None:
            return None

        axis = self.current_axis
        idx = self._current_slice_num()
        size = 200
        sx, sy, sz = vol.shape  # X(sagittal), Y(coronal), Z(axial)

        # Each entry: (slice_array_pre_rot, crosshair_row_pre_rot, crosshair_col_pre_rot, title)
        # Crosshairs are in PRE-rotation space; we convert to post-rotation below.
        if axis == 'x':
            # Current sagittal at idx
            raw_sag = vol[min(idx, sx-1), :, :]          # (Y, Z)
            raw_cor = vol[:, sy//2, :]                    # (X, Z)
            raw_axi = vol[:, :, sz//2]                    # (X, Y)
            # Crosshair positions in pre-rot space (row, col)
            sag_cr, sag_cc = sy//2, sz//2                 # center of sagittal
            cor_cr, cor_cc = sz//2, idx                   # mid-Z row, cur-X col
            axi_cr, axi_cc = sy//2, idx                   # mid-Y row, cur-X col
        elif axis == 'y':
            raw_sag = vol[sx//2, :, :]                    # (Y, Z)
            raw_cor = vol[:, min(idx, sy-1), :]           # (X, Z)
            raw_axi = vol[:, :, sz//2]                    # (X, Y)
            sag_cr, sag_cc = idx, sz//2                   # cur-Y row, mid-Z col
            cor_cr, cor_cc = sz//2, sx//2                 # mid-Z row, mid-X col
            axi_cr, axi_cc = idx, sx//2                   # cur-Y row, mid-X col
        else:  # z
            raw_sag = vol[sx//2, :, :]                    # (Y, Z)
            raw_cor = vol[:, sy//2, :]                    # (X, Z)
            raw_axi = vol[:, :, min(idx, sz-1)]           # (X, Y)
            sag_cr, sag_cc = sy//2, idx                   # mid-Y row, cur-Z col
            cor_cr, cor_cc = idx, sx//2                   # cur-Z row, mid-X col
            axi_cr, axi_cc = sy//2, sx//2                 # center of axial

        specs = [
            (raw_sag, sag_cr, sag_cc, 'Sagittal'),
            (raw_cor, cor_cr, cor_cc, 'Coronal'),
            (raw_axi, axi_cr, axi_cc, 'Axial'),
        ]

        panels = []
        for raw_plane, pre_r, pre_c, title in specs:
            H, W = raw_plane.shape          # pre-rotation dimensions
            plane = np.rot90(raw_plane)     # (W, H) post-rotation

            # Convert pre-rotation (pre_r, pre_c) → post-rotation (post_r, post_c)
            # np.rot90 CCW: new[i,j] = old[j, H-1-i]  →  old[pre_r, pre_c] → new[H-1-pre_c, pre_r]
            post_r = H - 1 - pre_c
            post_c = pre_r

            resized = cv2.resize(plane, (size, size), interpolation=cv2.INTER_AREA)
            norm_cy = int(post_r * size / plane.shape[0])
            norm_cx = int(post_c * size / plane.shape[1])
            norm_cy = max(0, min(size - 1, norm_cy))
            norm_cx = max(0, min(size - 1, norm_cx))

            panel_rgb = make_crosshair(resized, norm_cy, norm_cx)
            title_bar = np.zeros((22, size, 3), dtype=np.uint8)
            label_txt = f'{title} [{axis.upper()}={idx}]' if title.lower() == {
                'x': 'sagittal', 'y': 'coronal', 'z': 'axial'}[axis] else title
            cv2.putText(title_bar, label_txt, (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            panels.append(np.vstack([title_bar, panel_rgb]))

        return np.hstack(panels)

    def update_display(self):
        axis = self.current_axis
        axis_data = self.axes_data[axis]
        total = len(axis_data)
        slice_num = self._current_slice_num()
        is_artifact = self._current_pred() == 1
        label = self._get_label(axis, slice_num)

        # Slider sync
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, total - 1))
        self.slice_slider.setValue(self.current_index)
        self.slice_slider.blockSignals(False)
        self.slice_label.setText(f'{self.current_index + 1}/{total}')

        # Main image
        composed = self._render_main_image()
        if composed is not None:
            pix = numpy_to_qpixmap(composed)
            self.image_label.setPixmap(
                pix.scaled(self.image_label.size(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setText('Slice PNG not found')

        # Ortho strip
        ortho = self._render_ortho_panel()
        if ortho is not None:
            pix_o = numpy_to_qpixmap(ortho)
            self.ortho_label.setPixmap(
                pix_o.scaled(self.ortho_label.size(),
                             Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation))

        # Stats tab
        artifact_count = (axis_data['pred'] == 1).sum()
        total_all = len(self.table)
        artifact_all = (self.table['pred'] == 1).sum()
        self.stats_label.setText(
            f'{axis.upper()}-axis:  {artifact_count}/{total} artifacts'
            f'  ({100*artifact_count/total:.1f}%)\n'
            f'All axes:  {artifact_all}/{total_all} artifacts'
            f'  ({100*artifact_all/total_all:.1f}%)\n'
        )
        bbox = self._get_bbox(axis, slice_num)
        bbox_txt = ''
        if is_artifact and bbox is not None:
            bbox_txt = (f'\nRegion: {bbox[3]-bbox[2]} × {bbox[1]-bbox[0]} px'
                        f'\n  @ ({bbox[2]},{bbox[0]}) → ({bbox[3]},{bbox[1]})')
        self.slice_info_label.setText(
            f'Axis: {axis.upper()}  |  Slice: {slice_num}\n'
            f'Model: {"ARTIFACT" if is_artifact else "CLEAN"}{bbox_txt}\n'
            f'Label: {LABEL_DISPLAY.get(label, label)}'
        )

        # Labels tab
        lc = LABEL_COLORS.get(label, '#888')
        self.label_info.setStyleSheet(
            f'padding: 8px; background: {lc}; color: white; '
            f'font-weight: bold; border-radius: 4px;')
        self.label_info.setText(
            f'Axis {axis.upper()} · Slice {slice_num}\n'
            f'Model: {"ARTIFACT" if is_artifact else "CLEAN"}\n'
            f'Your label: {LABEL_DISPLAY.get(label, label)}'
        )
        self.review_summary.setText(self._label_summary())

        self.statusBar().showMessage(
            f'Axis {axis.upper()} | Slice {self.current_index + 1}/{total} | '
            f'{"ARTIFACT" if is_artifact else "CLEAN"} | '
            f'{LABEL_DISPLAY.get(label, label)}'
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
        if self.current_index < len(self.axes_data[self.current_axis]) - 1:
            self.current_index += 1
            self.update_display()

    def _jump_to_artifact(self, direction):
        data = self.axes_data[self.current_axis]
        n = len(data)
        if direction == 'next':
            indices = list(range(self.current_index + 1, n)) + list(range(0, self.current_index))
        else:
            indices = (list(range(self.current_index - 1, -1, -1))
                       + list(range(n - 1, self.current_index, -1)))
        for i in indices:
            if int(data.iloc[i]['pred']) == 1:
                self.current_index = i
                self.update_display()
                return

    def on_next_artifact(self):
        self._jump_to_artifact('next')

    def on_prev_artifact(self):
        self._jump_to_artifact('prev')

    def on_toggle_heatmap(self):
        self.show_heatmap = self.chk_heatmap.isChecked()
        self.update_display()

    def on_toggle_bbox(self):
        self.show_bbox = self.chk_bbox.isChecked()
        self.update_display()

    def on_toggle_skull(self):
        self.show_skull_strip = self.chk_skull.isChecked()
        self.update_display()

    def on_opacity_change(self, value):
        self.heatmap_opacity = value / 100.0
        self.opacity_label.setText(f'{value}%')
        self.update_display()

    def on_threshold_change(self, value):
        self.bbox_percentile = value
        self.threshold_label.setText(f'{value}th %ile')
        self.update_display()

    def _apply_label(self, label: str):
        axis = self.current_axis
        slice_num = self._current_slice_num()
        self._set_label(axis, slice_num, label)
        self.update_display()
        # Auto-advance to next artifact after labeling (skip for 'clear')
        if label != 'unlabeled':
            self._jump_to_artifact('next')

    def _save_labels(self):
        self.labels_df.to_csv(self.labels_csv, index=False)
        self.statusBar().showMessage(f'Saved to {self.labels_csv}', 3000)

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
    parser.add_argument('--nifti_path', default=None,
                        help='Path to raw T1w .nii.gz — enables 3D orthographic crosshair view')
    parser.add_argument('--skull_strip_path', default=None,
                        help='Path to skull-stripped T1w .nii.gz (PNGs must be pre-generated '
                             'in <png_dir>_skull/ for side-by-side comparison)')
    args = parser.parse_args()

    for path, name in [
        (args.output_dir, 'output_dir'),
        (args.png_dir, 'png_dir'),
        (args.localization_dir, 'localization_dir'),
    ]:
        if not os.path.isdir(path):
            print(f'Error: {name} does not exist: {path}')
            sys.exit(1)

    app = QApplication(sys.argv)
    win = ArtifactReviewer(
        args.output_dir, args.png_dir, args.localization_dir,
        nifti_path=args.nifti_path,
        skull_strip_path=args.skull_strip_path,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
