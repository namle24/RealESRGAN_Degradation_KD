import os
import glob
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from basicsr.metrics import calculate_niqe


def calculate_metrics(lq_dir, hq_dir, crop_border=4):
    lq_files = sorted(glob.glob(os.path.join(lq_dir, '*.png')))
    hq_files = sorted(glob.glob(os.path.join(hq_dir, '*.png')))

    assert len(lq_files) == len(hq_files), "Số lượng ảnh không khớp giữa hai thư mục!"

    total_psnr, total_ssim, total_niqe = 0, 0, 0
    count = 0

    for lq_path, hq_path in zip(lq_files, hq_files):
        img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        img_hq = cv2.imread(hq_path, cv2.IMREAD_COLOR)

        if crop_border > 0:
            img_lq = img_lq[crop_border:-crop_border, crop_border:-crop_border, :]
            img_hq = img_hq[crop_border:-crop_border, crop_border:-crop_border, :]

        img_lq_y = cv2.cvtColor(img_lq, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        img_hq_y = cv2.cvtColor(img_hq, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        psnr = peak_signal_noise_ratio(img_hq_y, img_lq_y, data_range=255)
        ssim = structural_similarity(img_hq_y, img_lq_y, data_range=255)

        niqe = calculate_niqe(img_lq, input_order='HWC', convert_to='y')

        total_psnr += psnr
        total_ssim += ssim
        total_niqe += niqe
        count += 1

        print(f"{os.path.basename(lq_path)}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.2f}")

    print(f"\n===> Trung bình trên {count} ảnh:")
    print(f"PSNR trung bình: {total_psnr / count:.2f}")
    print(f"SSIM trung bình: {total_ssim / count:.4f}")
    print(f"NIQE trung bình: {total_niqe / count:.2f}")


# === Đường dẫn thư mục ===
lq_dir = r"D:\EnhanceVideo_ImageDLM\data\dataset\train\HR"
hq_dir = r"D:\EnhanceVideo_ImageDLM\data\dataset\train\LR_light"

calculate_metrics(lq_dir, hq_dir)
