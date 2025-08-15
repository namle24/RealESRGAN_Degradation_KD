import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random


def add_noise(img, sigma_range=[1, 20]):
    noise_type = random.choice(['gaussian', 'sp', 'poisson', 'mixed'])

    if noise_type == 'gaussian':
        mean = 0
        sigma = random.uniform(*sigma_range)
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy
    elif noise_type == 'sp':
        prob = random.uniform(0.001, 0.01)
        output = np.copy(img)
        salt_mask = np.random.random(img.shape[:2]) < prob / 2
        output[salt_mask] = 255
        pepper_mask = np.random.random(img.shape[:2]) < prob / 2
        output[pepper_mask] = 0
        return output
    elif noise_type == 'poisson':
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        output = np.random.poisson(img * vals) / float(vals)
        return np.clip(output, 0, 255).astype(np.uint8)
    else:  # mixed
        temp = add_noise(img, sigma_range)
        return add_noise(temp, sigma_range)


def crop_image(img, x, y, width, height):
    # Đảm bảo vùng cắt nằm trong ảnh
    h, w = img.shape[:2]
    x = max(0, min(x, w - width))
    y = max(0, min(y, h - height))
    return img[y:y + height, x:x + width]


def create_noise_comparison(input_path, crop_x=50, crop_y=50, crop_width=100, crop_height=100, sigma_range=[1, 20]):
    # Kiểm tra file có tồn tại không
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {input_path}. Vui lòng kiểm tra đường dẫn.")

    # Đọc ảnh bằng OpenCV (RGB)
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Không thể đọc file ảnh tại: {input_path}. Kiểm tra định dạng hoặc quyền truy cập.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Cắt vùng ảnh
    cropped_img = crop_image(img, crop_x, crop_y, crop_width, crop_height)

    # Thêm các loại nhiễu vào vùng đã cắt
    gaussian_noisy = cropped_img.copy()
    mean = 0
    sigma = random.uniform(*sigma_range)
    gauss = np.random.normal(mean, sigma, gaussian_noisy.shape).astype(np.float32)
    gaussian_noisy = np.clip(gaussian_noisy.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    sp_noisy = cropped_img.copy()
    prob = random.uniform(0.001, 0.01)
    salt_mask = np.random.random(sp_noisy.shape[:2]) < prob / 2
    sp_noisy[salt_mask] = 255
    pepper_mask = np.random.random(sp_noisy.shape[:2]) < prob / 2
    sp_noisy[pepper_mask] = 0

    poisson_noisy = cropped_img.copy()
    vals = len(np.unique(poisson_noisy))
    vals = 2 ** np.ceil(np.log2(vals))
    poisson_noisy = np.random.poisson(poisson_noisy * vals) / float(vals)
    poisson_noisy = np.clip(poisson_noisy, 0, 255).astype(np.uint8)

    mixed_noisy = add_noise(cropped_img.copy(), sigma_range)
    mixed_noisy = add_noise(mixed_noisy, sigma_range)

    # Tạo lưới 1x5
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    titles = ['Original (Cropped)', 'Gaussian', 'Salt & Pepper', 'Poisson', 'Mixed']
    images = [cropped_img, gaussian_noisy, sp_noisy, poisson_noisy, mixed_noisy]

    # Hiển thị ảnh
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig('noise_comparison_cropped.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    input_image = r'D:\EnhanceVideo_ImageDLM\data\RealSR(V3)\Canon\Train\2\Canon_144_HR.png'
    create_noise_comparison(input_image, crop_x=100, crop_y=100, crop_width=512, crop_height=512)