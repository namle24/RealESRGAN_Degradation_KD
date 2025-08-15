import os
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import piq
from torchvision import transforms

gt_dir = r"D:\Download\archive\RealSR(V3)\Canon\HR"
lq_dir = r"D:\Download\archive\RealSR(V3)\Canon\LR"

def load_image(path, target_shape=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if target_shape and img.shape != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_niqe_piq(img_np):
    from PIL import Image
    from torchvision import transforms
    import torch
    import piq

    image_pil = Image.fromarray(img_np)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    img_tensor = transform(image_pil).unsqueeze(0).to(torch.float32)  # (1, 1, H, W)
    niqe_score = piq.niqe(img_tensor, data_range=1.0).item()
    return niqe_score

def evaluate_pair(lq_path, gt_path):
    gt = load_image(gt_path)
    if gt is None:
        return None
    lq = load_image(lq_path, target_shape=gt.shape)
    if lq is None:
        return None

    psnr = compare_psnr(gt, lq, data_range=255)
    ssim = compare_ssim(gt, lq, data_range=255, channel_axis=-1, win_size=7)
    niqe = calculate_niqe_piq(lq)
    return psnr, ssim, niqe

def evaluate_all(lq_dir, gt_dir):
    results = []

    for filename in sorted(os.listdir(gt_dir)):
        if not filename.lower().endswith(".png"):
            continue

        gt_path = os.path.join(gt_dir, filename)
        lq_path = os.path.join(lq_dir, filename)

        if not os.path.exists(lq_path):
            print(f"Không tìm thấy ảnh LQ: {filename}")
            continue

        metrics = evaluate_pair(lq_path, gt_path)
        if metrics:
            psnr, ssim, niqe = metrics
            results.append((filename, psnr, ssim, niqe))

    # Top 5 ảnh theo PSNR
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n== Top 5 ảnh LQ có PSNR cao nhất so với GT ==")
    for i, (fname, psnr, ssim, niqe) in enumerate(results[:5], 1):
        print(f"{i}. {fname} → PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, NIQE: {niqe:.3f}")

    # Trung bình
    avg_psnr = sum(r[1] for r in results) / len(results)
    avg_ssim = sum(r[2] for r in results) / len(results)
    avg_niqe = sum(r[3] for r in results) / len(results)

    print(f"\n== Trung bình toàn bộ ảnh ==")
    print(f"PSNR trung bình: {avg_psnr:.2f}")
    print(f"SSIM trung bình: {avg_ssim:.4f}")
    print(f"NIQE trung bình: {avg_niqe:.3f}")

evaluate_all(lq_dir, gt_dir)
