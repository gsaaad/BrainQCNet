#!/usr/bin/env python3
"""
Generate per-slice activation maps and bounding boxes around actual artifact regions.
Uses the trained PPNet model to compute spatial activation patterns.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from helpers import find_high_activation_crop
from preprocess import mean, std


def load_model(model_path):
    """Load the trained PPNet model."""
    print(f"Loading model: {model_path}")
    ppnet = torch.load(model_path, weights_only=False)
    ppnet = ppnet.cuda()
    ppnet.eval()
    return ppnet


def compute_artifact_localization(ppnet, img_path, class_of_interest=1, percentile=95):
    """
    For a single slice, compute:
      - prediction (0 = clean, 1 = artifact)
      - activation heatmap (spatial, upsampled to image size)
      - bounding box around the high-activation region

    Returns: dict with keys 'pred', 'heatmap', 'bbox', 'original_img'
    """
    img_size = ppnet.img_size

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img_pil).unsqueeze(0).cuda()

    with torch.no_grad():
        logits, min_distances = ppnet(img_tensor)
        conv_output, distances = ppnet.push_forward(img_tensor)
        prototype_activations = ppnet.distance_2_similarity(min_distances)
        prototype_activation_patterns = ppnet.distance_2_similarity(distances)

    pred = torch.argmax(logits, dim=1)[0].item()

    # Find prototypes that belong to the class of interest (artifact class)
    # Get the prototypes with highest activation for this class
    # We use prototype-to-class mapping from last_layer weights
    last_layer_weights = ppnet.last_layer.weight.detach().cpu().numpy()  # [num_classes, num_prototypes]
    class_prototype_weights = last_layer_weights[class_of_interest]  # Weights for artifact class

    # Get activation patterns from artifact-class prototypes only (positive weights)
    prototype_mask = class_prototype_weights > 0

    # Weighted sum of activation patterns for artifact class
    activation_patterns_np = prototype_activation_patterns[0].detach().cpu().numpy()  # [n_prototypes, h, w]

    # Weight each prototype by its connection to the artifact class
    weighted_activations = np.zeros_like(activation_patterns_np[0])
    for p_idx in range(len(class_prototype_weights)):
        if prototype_mask[p_idx]:
            weight = class_prototype_weights[p_idx]
            weighted_activations += weight * activation_patterns_np[p_idx]

    # Upsample to original image size
    upsampled_activation = cv2.resize(
        weighted_activations,
        dsize=(img_size, img_size),
        interpolation=cv2.INTER_CUBIC
    )

    # Find high-activation bounding box
    bbox = find_high_activation_crop(upsampled_activation, percentile=percentile)
    # bbox = (lower_y, upper_y, lower_x, upper_x)

    # Also prepare the original image at model's img_size
    img_resized = img_pil.resize((img_size, img_size))
    original_img_np = np.array(img_resized).astype(np.float32) / 255.0

    # Normalize heatmap
    heat_min = upsampled_activation.min()
    heat_max = upsampled_activation.max()
    if heat_max - heat_min > 1e-6:
        heatmap_norm = (upsampled_activation - heat_min) / (heat_max - heat_min)
    else:
        heatmap_norm = np.zeros_like(upsampled_activation)

    return {
        'pred': pred,
        'heatmap': heatmap_norm,
        'bbox': bbox,
        'original_img': original_img_np,
        'img_size': img_size,
    }


def create_overlay_image(original_img, heatmap, bbox, is_artifact, alpha=0.4):
    """
    Create visualization image with heatmap overlay and bounding box.

    Returns RGB image (uint8).
    """
    # Apply colormap to heatmap (JET: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_float = heatmap_colored.astype(np.float32) / 255.0

    # Blend with original (original must be in [0,1])
    if original_img.max() > 1.0:
        original_float = original_img.astype(np.float32) / 255.0
    else:
        original_float = original_img.astype(np.float32)

    if is_artifact:
        overlayed = (1 - alpha) * original_float + alpha * heatmap_float
    else:
        # For clean slices, minimal heatmap overlay
        overlayed = original_float.copy()

    overlayed = np.clip(overlayed, 0, 1)
    overlayed_uint8 = (overlayed * 255).astype(np.uint8)

    # Draw bounding box only for artifacts
    if is_artifact:
        lower_y, upper_y, lower_x, upper_x = bbox
        color = (255, 255, 0)  # Yellow bounding box
        thickness = 2
        cv2.rectangle(
            overlayed_uint8,
            (lower_x, lower_y),
            (upper_x - 1, upper_y - 1),
            color,
            thickness
        )

    return overlayed_uint8


def process_all_slices(model_path, png_dir, output_dir, table_csv, percentile=95):
    """
    Process all PNG slices and generate activation maps + bounding boxes.
    """
    os.makedirs(output_dir, exist_ok=True)

    ppnet = load_model(model_path)

    table = pd.read_csv(table_csv)
    print(f"\nTotal slices to process: {len(table)}")

    results = []

    for idx, row in table.iterrows():
        axis = row['axis']
        slice_num = int(row['slice'])

        # Find matching PNG
        patterns = [
            f"*_T1w_{axis}_{slice_num:03d}.png",
            f"*_{axis}_{slice_num:03d}.png",
        ]
        png_file = None
        for pattern in patterns:
            matches = glob.glob(os.path.join(png_dir, pattern))
            if matches:
                png_file = matches[0]
                break

        if png_file is None:
            continue

        try:
            result = compute_artifact_localization(ppnet, png_file, class_of_interest=1, percentile=percentile)
            is_artifact = result['pred'] == 1

            # Save visualization
            overlay = create_overlay_image(
                result['original_img'],
                result['heatmap'],
                result['bbox'],
                is_artifact
            )

            output_filename = f"{axis}_{slice_num:03d}_{'artifact' if is_artifact else 'clean'}.png"
            output_path = os.path.join(output_dir, output_filename)
            Image.fromarray(overlay).save(output_path)

            results.append({
                'axis': axis,
                'slice': slice_num,
                'pred': result['pred'],
                'bbox_y1': result['bbox'][0],
                'bbox_y2': result['bbox'][1],
                'bbox_x1': result['bbox'][2],
                'bbox_x2': result['bbox'][3],
                'heatmap_max': float(result['heatmap'].max()),
                'visualization_path': output_path,
            })

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(table)}")

        except Exception as e:
            print(f"  Error processing {png_file}: {e}")
            continue

    # Save results table
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "localization_results.csv")
    results_df.to_csv(results_csv, index=False)

    print(f"\n✓ Processed {len(results)} slices")
    print(f"✓ Visualizations saved to: {output_dir}")
    print(f"✓ Results table: {results_csv}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Generate artifact localization maps.')
    parser.add_argument('--modeldir', required=True, help='Path to model directory')
    parser.add_argument('--model', default='10push0.8167.pth', help='Model filename')
    parser.add_argument('--pngdir', required=True, help='Directory containing PNG slices')
    parser.add_argument('--tablecsv', required=True, help='Path to table.csv with predictions')
    parser.add_argument('--outdir', required=True, help='Output directory for visualizations')
    parser.add_argument('--percentile', type=int, default=95, help='Activation percentile for bbox (default: 95)')
    args = parser.parse_args()

    model_path = os.path.join(args.modeldir, args.model)

    process_all_slices(
        model_path=model_path,
        png_dir=args.pngdir,
        output_dir=args.outdir,
        table_csv=args.tablecsv,
        percentile=args.percentile,
    )


if __name__ == '__main__':
    main()
