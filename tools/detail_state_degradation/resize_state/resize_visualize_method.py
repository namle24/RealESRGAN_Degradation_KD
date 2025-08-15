import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Load ảnh gốc
img = cv2.imread('D:\\EnhanceVideo_ImageDLM\\data\\RealSR(V3)\\Canon\\Train\\2\\Canon_044_HR.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Cấu hình
crop_size = 256
scale = 0.25
interpolations = {
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}
down_names = list(interpolations.keys())
up_names = list(interpolations.keys())

# Cắt vùng trung tâm
center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
half_crop = crop_size // 2

def crop_center(image):
    return image[
        center_y - half_crop:center_y + half_crop,
        center_x - half_crop:center_x + half_crop
    ]

# Tạo Figure và GridSpec với không gian cho nhãn
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(5, 5, width_ratios=[0.5, 1, 1, 1, 1], height_ratios=[0.5, 1, 1, 1, 1])
gs.update(wspace=0.05, hspace=0.05)

# Nhãn "Downsample" trên cùng
ax_title = plt.subplot(gs[0, 1:])
ax_title.axis('off')
ax_title.text(0.5, 0.3, 'Downsample ↓', ha='center', va='center', fontsize=12)

# Nhãn "Upsample" bên trái
ax_ylab = plt.subplot(gs[1:, 0])
ax_ylab.axis('off')
ax_ylab.text(0.2, 0.5, 'Upsample ↑', ha='center', va='center', fontsize=12, rotation=90)

# Lưới hình ảnh
for row, up_name in enumerate(up_names):
    for col, down_name in enumerate(down_names):
        ax = plt.subplot(gs[row + 1, col + 1])

        # Resize
        down_method = interpolations[down_name]
        up_method = interpolations[up_name]
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=down_method)
        restored = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=up_method)
        cropped = crop_center(restored)

        ax.imshow(cropped)
        ax.axis('off')

        # Tên cột (chỉ hàng đầu)
        if row == 0:
            ax.set_title(down_name, fontsize=10)

        # Tên hàng (chỉ cột đầu)
        if col == 0:
            ax.text(-0.1, 0.5, up_name, va='center', ha='right',
                    fontsize=10, transform=ax.transAxes)

plt.show()