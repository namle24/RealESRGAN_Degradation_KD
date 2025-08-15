import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from RealisticDegradationGenerator_RealESRGAN import RealisticDegradationGenerator
from OriginalRealESRGANDegradation import OriginalRealESRGANDegradation
import argparse


def visualize_comparison(img_path, output_dir, scale_factor=0.25):
    """Generate degraded images using both methods and visualize the comparison"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh: {img_path}")
        return

    # Convert BGR to RGB for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize both degradation methods
    your_degradation = RealisticDegradationGenerator()
    original_degradation = OriginalRealESRGANDegradation()

    # Apply degradations
    degraded_yours = your_degradation.apply_degradation(img, scale_factor)
    degraded_original = original_degradation.apply_degradation(img, scale_factor)

    # Convert to RGB for visualization
    degraded_yours_rgb = cv2.cvtColor(degraded_yours, cv2.COLOR_BGR2RGB)
    degraded_original_rgb = cv2.cvtColor(degraded_original, cv2.COLOR_BGR2RGB)

    # Calculate height to maintain aspect ratio for visualization
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # Create figure
    plt.figure(figsize=(15, 8))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image (GT)')
    plt.axis('off')

    # Original Real-ESRGAN degradation result
    plt.subplot(1, 3, 3)
    plt.imshow(degraded_original_rgb)
    plt.title('Original Degradation')
    plt.axis('off')

    # Your degradation result
    plt.subplot(1, 3, 2)
    plt.imshow(degraded_yours_rgb)
    plt.title('Our Generated Degradation')
    plt.axis('off')

    output_path = os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + '_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Comparison saved to {output_path}")

    # Save individual degraded images for further inspection
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + '_yours.png'), degraded_yours)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + '_original.png'), degraded_original)


def batch_comparison(input_dir, output_dir, scale_factor=0.25, max_images=5):
    """Process multiple images for comparison"""
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []

    # Collect all image files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    # Limit the number of images to process
    if max_images > 0 and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]

    print(f"Processing {len(image_paths)} images...")
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        visualize_comparison(img_path, output_dir, scale_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare degradation methods')
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./comparison_results', help='Output directory')
    parser.add_argument('--scale', type=float, default=0.25, help='Scale factor (e.g., 0.25 for x4)')
    parser.add_argument('--max_images', type=int, default=20, help='Max number of images to process in batch mode')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_comparison(args.input, args.output, args.scale, args.max_images)
    else:
        visualize_comparison(args.input, args.output, args.scale)