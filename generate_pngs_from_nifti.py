#!/usr/bin/env python3
"""
Generate PNG slices from NIfTI file using nibabel and matplotlib.
Simpler alternative to med2image.
"""

import os
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

def generate_pngs(nifti_path, output_dir):
    """
    Convert NIfTI to PNG slices along all three axes.

    nifti_path: Path to NIfTI file (.nii.gz)
    output_dir: Directory to save PNG slices
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading NIfTI: {nifti_path}")
    img = nib.load(nifti_path)
    data = img.get_fdata()

    print(f"Shape: {data.shape}")
    print(f"Data range: {data.min():.1f} to {data.max():.1f}")

    # Normalize data to 0-255
    data_min = np.percentile(data, 1)  # Use 1st percentile to avoid outliers
    data_max = np.percentile(data, 99)
    data_norm = np.clip((data - data_min) / (data_max - data_min + 1e-6) * 255, 0, 255).astype(np.uint8)

    # Generate slices for each axis
    axes_info = {
        'x': (0, data.shape[0], "X-axis (Sagittal)"),
        'y': (1, data.shape[1], "Y-axis (Coronal)"),
        'z': (2, data.shape[2], "Z-axis (Axial)"),
    }

    for axis_name, (axis_idx, num_slices, axis_label) in axes_info.items():
        print(f"\nGenerating {axis_label} ({num_slices} slices)...")

        for i in range(num_slices):
            # Extract slice
            if axis_idx == 0:
                slice_data = data_norm[i, :, :]
            elif axis_idx == 1:
                slice_data = data_norm[:, i, :]
            else:  # axis_idx == 2
                slice_data = data_norm[:, :, i]

            # Rotate for display (medical convention)
            slice_data = np.rot90(slice_data)

            # Convert to RGB
            slice_rgb = np.stack([slice_data, slice_data, slice_data], axis=-1)

            # Save as PNG
            filename = os.path.join(output_dir, f"sub-08_T1w_{axis_name}_{i:03d}.png")
            Image.fromarray(slice_rgb).save(filename)

            if (i + 1) % 50 == 0 or i == num_slices - 1:
                print(f"  Saved {i + 1}/{num_slices} slices")

    print(f"\n✓ PNG generation complete!")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Total PNG files: {len(os.listdir(output_dir))}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_pngs_from_nifti.py <nifti_file> <output_dir>")
        print("\nExample:")
        print("  python generate_pngs_from_nifti.py BIDS_data/sub-08/anat/sub-08_T1w.nii.gz out_brainqcnet/sub-08/pngs/")
        sys.exit(1)

    nifti_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isfile(nifti_path):
        print(f"Error: NIfTI file not found: {nifti_path}")
        sys.exit(1)

    generate_pngs(nifti_path, output_dir)

if __name__ == '__main__':
    main()
