import cv2
import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch
from basicsr.metrics import calculate_niqe
from basicsr.utils import scandir

def calculate_metrics(sr_dir, gt_dir, crop_border=4):
    sr_files = sorted(glob.glob(os.path.join(sr_dir, '*.png')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    psnr_list, ssim_list, lpips_list, niqe_list = [], [], [], []
    
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    
    for sr_path, gt_path in zip(sr_files, gt_files):
        sr_img = cv2.imread(sr_path)
        gt_img = cv2.imread(gt_path)
        
        # Crop border
        if crop_border > 0:
            sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border]
            gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]
        
        # PSNR
        psnr = peak_signal_noise_ratio(gt_img, sr_img)
        psnr_list.append(psnr)
        
        # SSIM
        ssim = structural_similarity(gt_img, sr_img, multichannel=True)
        ssim_list.append(ssim)
        
        # LPIPS
        sr_tensor = torch.from_numpy(sr_img.transpose(2, 0, 1)).float().cuda() / 255.0
        gt_tensor = torch.from_numpy(gt_img.transpose(2, 0, 1)).float().cuda() / 255.0
        lpips_val = lpips_model(sr_tensor, gt_tensor).item()
        lpips_list.append(lpips_val)
        
        # NIQE (hoặc PIQE)
        niqe_val, _, _ = piqe(sr_img)
        niqe_list.append(niqe_val)
    
    return {
        'psnr': np.mean(psnr_list),
        'ssim': np.mean(ssim_list),
        'lpips': np.mean(lpips_list),
        'niqe': np.mean(niqe_list)
    }

test_sets = [
    ("DIV2K_val_bicubic", "/storage/student12/Real-ESRGAN/data/DIV2K/DIV2K_valid_LR_bicubic/X4", "/storage/student12/Real-ESRGAN/data/DIV2K/DIV2K_valid_HR/"),
    ("Set5", "/storage/student12/Real-ESRGAN/data/Set5/Set5/GTmod12", "/storage/student12/Real-ESRGAN/data/Set5/Set5/LRbicx4"),
    ("Set14", "/storage/student12/Real-ESRGAN/data/Set14/Set14/GTmod12/", "/storage/student12/Real-ESRGAN/data/Set14/Set14/LRbicx4/")
]

for name, sr_dir, gt_dir in test_sets:
    metrics = calculate_metrics(sr_dir, gt_dir, crop_border=4)
    print(f"{name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}, NIQE={metrics['niqe']:.2f}")