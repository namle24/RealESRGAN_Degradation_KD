import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def crop_image(img, x, y, width, height):
    h, w = img.shape[:2]
    x = max(0, min(x, w - width))
    y = max(0, min(y, h - height))
    return img[y:y + height, x:x + width]


def add_channel_shift(img):
    b, g, r = cv2.split(img)
    shift = random.randint(1, 5)

    if random.random() < 0.5:
        # Dịch ngang
        r = np.pad(r, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
        b = np.pad(b, ((0, 0), (0, shift)), mode='edge')[:, shift:]
    else:
        # Dịch dọc
        r = np.pad(r, ((shift, 0), (0, 0)), mode='edge')[:-shift, :]
        b = np.pad(b, ((0, shift), (0, 0)), mode='edge')[shift:, :]

    # Đảm bảo cùng kích thước
    r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)

    return cv2.merge([b, g, r])


def add_pattern_noise(img, sigma_range=(2, 5)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_camera_noise(img, apply_shift=True, apply_pattern=True):
    result = img.copy()
    if apply_shift:
        result = add_channel_shift(result)
    if apply_pattern:
        result = add_pattern_noise(result)
    return result


def create_camera_noise_comparison(input_path, crop_x=100, crop_y=100, crop_width=120, crop_height=120):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh tại: {input_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cropped_img = crop_image(img, crop_x, crop_y, crop_width, crop_height)

    # Tạo các biến thể nhiễu
    channel_shift = add_camera_noise(cropped_img, apply_shift=True, apply_pattern=False)
    pattern_noise = add_camera_noise(cropped_img, apply_shift=False, apply_pattern=True)
    both = add_camera_noise(cropped_img, apply_shift=True, apply_pattern=True)

    # Hiển thị ảnh
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = ['Original (Cropped)', 'Channel Shift Only', 'Pattern Noise Only', 'Both Noises']
    images = [cropped_img, channel_shift, pattern_noise, both]

    for ax, title, image in zip(axes.flat, titles, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('camera_noise_comparison.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    input_image = r'D:\EnhanceVideo_ImageDLM\data\RealSR(V3)\Canon\Test\4\Canon_020_HR.png'
    create_camera_noise_comparison(input_image, crop_x=100, crop_y=100, crop_width=120, crop_height=120)
