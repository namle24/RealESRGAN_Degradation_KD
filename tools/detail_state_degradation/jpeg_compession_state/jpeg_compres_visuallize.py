import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def crop_image(img, x, y, width, height):
    h, w = img.shape[:2]
    x = max(0, min(x, w - width))
    y = max(0, min(y, h - height))
    return img[y:y + height, x:x + width]


def apply_jpeg_compression(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def create_jpeg_comparison(input_path, crop_x=50, crop_y=50, crop_width=100, crop_height=100):
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

    # Áp dụng JPEG compression với các mức chất lượng khác nhau
    quality_levels = [50, 60, 70, 80, 95]  # Các mức chất lượng để mô phỏng ảnh thực tế
    jpeg_images = [apply_jpeg_compression(cropped_img, q) for q in quality_levels]

    # Tạo lưới 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Original (Cropped)', 'JPEG Q=50', 'JPEG Q=60', 'JPEG Q=70', 'JPEG Q=80', 'JPEG Q=95']
    images = [cropped_img] + jpeg_images

    # Hiển thị ảnh
    for ax, title, image in zip(axes.flat, titles, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig('jpeg_comparison_cropped.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    input_image = r'D:\EnhanceVideo_ImageDLM\data\RealSR(V3)\Canon\Train\2\Canon_120_HR.png'
    create_jpeg_comparison(input_image, crop_x=100, crop_y=100, crop_width=120, crop_height=120)