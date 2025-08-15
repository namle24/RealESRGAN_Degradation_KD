import cv2
import numpy as np
import torch
import torch.nn.functional as F
from basicsr.utils import DiffJPEG, USMSharp, filter2D
import os
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

# Cấu hình logging để ghi log vào file và hiển thị trên console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("degradation_log.txt"),
        logging.StreamHandler()
    ]
)

# Đường dẫn thư mục chứa ảnh HR và thư mục lưu ảnh LR
hr_dir = "D:\\EnhanceVideo_ImageDLM\\DIV2K+_scale_4x\\train\\HR"
output_dir = "D:\\EnhanceVideo_ImageDLM\\DIV2K_original_Degradation\\train\\LR"

# Thông số degradation (dựa trên giá trị mặc định của Real-ESRGAN)
scale = 4  # Scale x4
resize_prob = [0.2, 0.7, 0.1]  # [up, down, keep] for first degradation
resize_range = [0.4, 1.2]
gaussian_noise_prob = 0.5
noise_range = [1, 30]
poisson_scale_range = [0.5, 3.0]
gray_noise_prob = 0.4
jpeg_range = [60, 100]
second_blur_prob = 0.5
resize_prob2 = [0.3, 0.4, 0.3]  # [up, down, keep] for second degradation
resize_range2 = [0.3, 1.2]
gaussian_noise_prob2 = 0.5
noise_range2 = [1, 25]
poisson_scale_range2 = [0.5, 3.0]
gray_noise_prob2 = 0.4
jpeg_range2 = [50, 95]
gt_size = 256  # Patch size cho HR

# Hàm thêm Gaussian noise
def random_add_gaussian_noise_pt(img, sigma_range, clip=True, rounds=False, gray_prob=0.0, device='cpu'):
    noise_sigma = torch.FloatTensor([np.random.uniform(sigma_range[0], sigma_range[1])]).to(device)
    noise = torch.randn_like(img) * (noise_sigma / 255.0)
    if np.random.uniform() < gray_prob:
        noise = noise.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    img = img + noise
    if clip:
        img = torch.clamp(img, 0, 1)
    if rounds:
        img = torch.round(img * 255) / 255
    return img

# Hàm thêm Poisson noise
def random_add_poisson_noise_pt(img, scale_range, clip=True, rounds=False, gray_prob=0.0, device='cpu'):
    scale = torch.FloatTensor([np.random.uniform(scale_range[0], scale_range[1])]).to(device)
    noise = torch.poisson(img * 255.0 * scale) / 255.0 / scale - img
    if np.random.uniform() < gray_prob:
        noise = noise.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    img = img + noise
    if clip:
        img = torch.clamp(img, 0, 1)
    if rounds:
        img = torch.round(img * 255) / 255
    return img

# Hàm tạo sinc kernel (mô phỏng cảm biến camera)
def create_sinc_kernel(device='cpu'):
    kernel_size = 21
    omega_c = np.pi * 0.9
    kernel = torch.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            r = np.sqrt((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2)
            if r == 0:
                kernel[i, j] = omega_c / np.pi
            else:
                kernel[i, j] = np.sin(omega_c * r) / (np.pi * r)
    kernel = kernel / kernel.sum()  # Chuẩn hóa
    return kernel.unsqueeze(0).unsqueeze(0).to(device)

# Hàm tạo Gaussian kernel
def create_gaussian_kernel(sigma, kernel_size=21, device='cpu'):
    x = torch.arange(kernel_size).float().to(device) - kernel_size // 2
    y = x.view(-1, 1)
    gaussian_kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel.unsqueeze(0).unsqueeze(0)

def degrade_image(filename, hr_dir, output_dir, device='cpu'):
    try:
        # Đọc ảnh HR
        hr_path = os.path.join(hr_dir, filename)
        hr_img = cv2.imread(hr_path)
        if hr_img is None:
            logging.error(f"Không thể đọc ảnh: {hr_path}")
            return
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = hr_img / 255.0
        hr_img = torch.from_numpy(hr_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Khởi tạo các công cụ
        jpeger = DiffJPEG(differentiable=False).to(device)
        usm_sharpener = USMSharp().to(device)
        sinc_kernel = create_sinc_kernel(device)

        # USM Sharpening
        hr_img_usm = usm_sharpener(hr_img)

        # Lấy kích thước gốc
        _, _, ori_h, ori_w = hr_img.size()

        # First Degradation Process
        sigma1 = np.random.uniform(0.5, 2.0)
        kernel1 = create_gaussian_kernel(sigma1, device=device)
        out = filter2D(hr_img_usm, kernel1)

        # Random Resize
        updown_type = np.random.choice(['up', 'down', 'keep'], p=resize_prob)
        if updown_type == 'up':
            scale = np.random.uniform(1, resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(resize_range[0], 1)
        else:
            scale = 1
        mode = np.random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Add Noise
        if np.random.uniform() < gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob, device=device)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=poisson_scale_range, clip=True, rounds=False, gray_prob=gray_noise_prob, device=device)

        # JPEG Compression
        jpeg_p = torch.FloatTensor([np.random.uniform(jpeg_range[0], jpeg_range[1])]).to(device)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)

        # Second Degradation Process
        if np.random.uniform() < second_blur_prob:
            sigma2 = np.random.uniform(0.5, 2.0)
            kernel2 = create_gaussian_kernel(sigma2, device=device)
            out = filter2D(out, kernel2)

        # Random Resize
        updown_type = np.random.choice(['up', 'down', 'keep'], p=resize_prob2)
        if updown_type == 'up':
            scale = np.random.uniform(1, resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(resize_range2[0], 1)
        else:
            scale = 1
        mode = np.random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(int(ori_h / scale), int(ori_w / scale)), mode=mode)

        # Add Noise
        if np.random.uniform() < gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(out, sigma_range=noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob2, device=device)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=poisson_scale_range2, clip=True, rounds=False, gray_prob=gray_noise_prob2, device=device)

        # Final Steps (JPEG + Sinc Filter)
        if np.random.uniform() < 0.5:
            # Resize + Sinc Filter → JPEG
            mode = np.random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
            out = filter2D(out, sinc_kernel)
            jpeg_p = torch.FloatTensor([np.random.uniform(jpeg_range2[0], jpeg_range2[1])]).to(device)
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG → Resize + Sinc Filter
            jpeg_p = torch.FloatTensor([np.random.uniform(jpeg_range2[0], jpeg_range2[1])]).to(device)
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            mode = np.random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
            out = filter2D(out, sinc_kernel)

        # Clamp và Round
        lq_img = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # Chuyển tensor về numpy để lưu
        lq_img = lq_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        lq_img = (lq_img * 255).round().astype(np.uint8)
        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_RGB2BGR)

        # Lưu ảnh LR
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, lq_img)
        logging.info(f"Đã lưu ảnh LR: {output_path}")
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {filename}: {e}")

def main(args):
    # Kiểm tra thiết bị (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logging.info(f"Thiết bị được sử dụng: {device}")

    # Tạo thư mục đầu ra
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh HR
    image_files = [f for f in os.listdir(args.hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    logging.info(f"Tìm thấy {len(image_files)} ảnh HR trong {args.hr_dir}")

    # Quét qua tất cả ảnh HR và tạo ảnh LR
    for filename in tqdm(image_files, desc="Generating LR images"):
        degrade_image(filename, args.hr_dir, args.output_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LR images using Real-ESRGAN degradation pipeline")
    parser.add_argument("--hr_dir", type=str, default="D:\\EnhanceVideo_ImageDLM\\DIV2K+_scale_4x\\train\\HR",
                        help="Directory containing HR images")
    parser.add_argument("--output_dir", type=str, default="D:\\EnhanceVideo_ImageDLM\\DIV2K_original_Degradation\\train\\LR",
                        help="Directory to save LR images")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    main(args)