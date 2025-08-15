import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random


def crop_image(img, x, y, width, height):
    h, w = img.shape[:2]
    x = max(0, min(x, w - width))
    y = max(0, min(y, h - height))
    return img[y:y + height, x:x + width]


def create_aniso_kernel(sigma_x, sigma_y, angle, kernel_size):
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x, y = np.meshgrid(x, y)
    x_rot = x * np.cos(np.radians(angle)) + y * np.sin(np.radians(angle))
    y_rot = -x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle))
    kernel = np.exp(-0.5 * ((x_rot ** 2 / sigma_x ** 2) + (y_rot ** 2 / sigma_y ** 2)))
    return kernel / kernel.sum()


def add_blur(img, blur_kernel_size=21):
    blur_type = random.choices(['gaussian', 'aniso', 'generalized', 'motion', 'defocus', 'sinc'], k=1)[0]

    if blur_type == 'gaussian':
        sigma = random.uniform(1.0, 5.0)  # Tăng sigma để tạo hiệu ứng mờ mạnh hơn
        return cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), sigma)
    elif blur_type == 'aniso':
        sigma_x = random.uniform(1.0, 6.0)  # Tăng sigma_x để tạo mờ không đồng đều
        sigma_y = random.uniform(1.0, 6.0)  # Tăng sigma_y để tạo mờ không đồng đều
        angle = random.uniform(0, 180)
        kernel = create_aniso_kernel(sigma_x, sigma_y, angle, blur_kernel_size)
        blurred = cv2.filter2D(img, -1, kernel)
        return np.clip(blurred, 0, 255).astype(np.uint8)
    elif blur_type == 'generalized':
        sigma = random.uniform(1.0, 5.0)  # Tăng sigma
        beta = random.uniform(1.5, 5.0)  # Tăng beta để tạo hiệu ứng mờ phức tạp
        x = np.arange(-blur_kernel_size // 2 + 1, blur_kernel_size // 2 + 1)
        kernel = np.exp(-np.abs(x / sigma) ** beta)
        kernel /= kernel.sum()
        kernel_2d = np.outer(kernel, kernel)
        blurred = cv2.filter2D(img, -1, kernel_2d)
        return np.clip(blurred, 0, 255).astype(np.uint8)
    elif blur_type == 'motion':
        degree = random.randint(10, 40)  # Tăng độ dài kernel để mô phỏng chuyển động mạnh
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.hamming(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / motion_blur_kernel.sum()
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)
        if random.random() < 0.3:  # Tăng xác suất blur lần nữa lên 30%
            blurred = add_blur(blurred, blur_kernel_size)
        return np.clip(blurred, 0, 255).astype(np.uint8)
    elif blur_type == 'defocus':
        radius = random.randint(3, 15)  # Tăng bán kính để tạo mờ mạnh hơn
        return cv2.GaussianBlur(img, (radius * 2 + 1, radius * 2 + 1), 0)
    else:  # sinc
        kernel_size = random.randint(9, blur_kernel_size)  # Tăng kernel size tối thiểu
        kernel = np.sinc(np.linspace(-4, 4, kernel_size)).astype(np.float32)  # Mở rộng phạm vi để tạo mờ lan rộng
        kernel /= kernel.sum()
        kernel_2d = np.outer(kernel, kernel)
        blurred = cv2.filter2D(img, -1, kernel_2d)
        return np.clip(blurred, 0, 255).astype(np.uint8)


def create_blur_comparison(input_path, crop_x=50, crop_y=50, crop_width=100, crop_height=100, blur_kernel_size=21):
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

    # Áp dụng các loại blur vào vùng đã cắt
    gaussian_blur = add_blur(cropped_img.copy(), blur_kernel_size)
    aniso_blur = add_blur(cropped_img.copy(), blur_kernel_size)
    generalized_blur = add_blur(cropped_img.copy(), blur_kernel_size)
    motion_blur = add_blur(cropped_img.copy(), blur_kernel_size)
    defocus_blur = add_blur(cropped_img.copy(), blur_kernel_size)
    sinc_blur = add_blur(cropped_img.copy(), blur_kernel_size)

    # Tạo lưới 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Original (Cropped)', 'Gaussian', 'Aniso', 'Generalized', 'Motion', 'Defocus', 'Sinc']
    images = [cropped_img, gaussian_blur, aniso_blur, generalized_blur, motion_blur, defocus_blur, sinc_blur]

    # Hiển thị ảnh
    for ax, title, image in zip(axes.flat, titles, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig('blur_comparison_cropped.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    input_image = r'D:\EnhanceVideo_ImageDLM\data\RealSR(V3)\Canon\Train\2\Canon_116_HR.png'
    create_blur_comparison(input_image, crop_x=100, crop_y=100, crop_width=512, crop_height=512)