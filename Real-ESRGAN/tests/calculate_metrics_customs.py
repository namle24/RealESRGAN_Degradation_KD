import cv2
import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch
from basicsr.metrics import calculate_niqe


def calculate_metrics(sr_dir, gt_dir, crop_border=4):
    sr_files = sorted(glob.glob(os.path.join(sr_dir, '*.png')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

    if not sr_files:
        print(f"No SR images found in {sr_dir}")
    if not gt_files:
        print(f"No GT images found in {gt_dir}")

    metrics_list = []

    # Load LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    lpips_model.eval()

    for sr_path, gt_path in zip(sr_files, gt_files):
        sr_img = cv2.imread(sr_path)
        gt_img = cv2.imread(gt_path)

        if sr_img is None:
            print(f"Failed to read SR image: {sr_path}")
            continue
        if gt_img is None:
            print(f"Failed to read GT image: {gt_path}")
            continue

        if crop_border > 0:
            sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border]
            gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(gt_img, sr_img)

        # Calculate SSIM
        ssim = structural_similarity(gt_img, sr_img, channel_axis=2)

        # Prepare tensors (CPU)
        sr_tensor = torch.from_numpy(sr_img.transpose(2, 0, 1)).float() / 255.0
        gt_tensor = torch.from_numpy(gt_img.transpose(2, 0, 1)).float() / 255.0

        # LPIPS (GPU, no_grad to save memory)
        with torch.no_grad():
            lpips_val = lpips_model(
                sr_tensor.unsqueeze(0).cuda(),
                gt_tensor.unsqueeze(0).cuda()
            ).item()
        torch.cuda.empty_cache()  # Dọn RAM GPU

        # NIQE
        niqe_val = calculate_niqe(sr_img, crop_border)

        metrics_list.append({
            'image': os.path.basename(sr_path),
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_val,
            'niqe': niqe_val
        })

    if not metrics_list:
        print("No metrics calculated. Check input images.")

    return metrics_list



def save_metrics(metrics_list, save_dir, dataset_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{dataset_name}_metrics.txt")

    with open(file_path, "w") as f:
        for metrics in metrics_list:
            f.write(
                f"{metrics['image']}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, "
                f"LPIPS={metrics['lpips']:.4f}, NIQE={metrics['niqe']:.2f}\n"
            )
    print(f"Metrics saved to {file_path}")


# Define your test sets
test_sets = [
    (
        "RealSR(V2)_original",
        "/storage/student12/Real-ESRGAN/Real-ESRGAN/inputs/metric_calculate/SR_orig",
        "/storage/student12/Real-ESRGAN/Real-ESRGAN/inputs/metric_calculate/GT",
    ),
    (
        "RealSR(V2)_our",
        "/storage/student12/Real-ESRGAN/Real-ESRGAN/inputs/metric_calculate/SR_our",
        "/storage/student12/Real-ESRGAN/Real-ESRGAN/inputs/metric_calculate/GT",
    ),
]

save_directory = "/storage/student12/Real-ESRGAN/Real-ESRGAN/inputs"

# Iterate through test sets and calculate metrics
for name, sr_dir, gt_dir in test_sets:
    metrics_list = calculate_metrics(sr_dir, gt_dir, crop_border=4)

    # Print metrics for each image
    for metrics in metrics_list:
        print(
            f"{name} - {metrics['image']}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, "
            f"LPIPS={metrics['lpips']:.4f}, NIQE={metrics['niqe']:.2f}"
        )

    # Save the results to the specified directory
    if metrics_list:
        save_metrics(metrics_list, save_directory, name)
    else:
        print(f"No metrics to save for {name}.")
