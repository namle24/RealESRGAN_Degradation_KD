import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def crop_image_to_patches(image_path, crop_size=120, step=60, thresh_size=0, save_folder='D:\\EnhanceVideo_ImageDLM\\results\\sub\\LR_x4_sub\\img3', show_preview=True):
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    h, w = img.shape[:2]
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    os.makedirs(save_folder, exist_ok=True)

    # Tính các vùng cần crop
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    patches = []
    idx = 1
    for x in h_space:
        for y in w_space:
            patch = img[x:x+crop_size, y:y+crop_size]
            patch_filename = os.path.join(save_folder, f"{base_name}_s{idx:03d}{ext}")
            cv2.imwrite(patch_filename, patch, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))  # để hiển thị
            idx += 1

    print(f"✅ Đã crop và lưu {idx-1} patch vào thư mục: {save_folder}")

    # Hiển thị các patch nếu muốn
    if show_preview:
        cols = min(6, len(patches))
        rows = (len(patches) + cols - 1) // cols
        plt.figure(figsize=(15, 3 * rows))
        for i, patch in enumerate(patches):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(patch)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Ví dụ sử dụng:
crop_image_to_patches('D:\\EnhanceVideo_ImageDLM\\DIV2K+_dataset\\train\\LR_x4\\3.jpg', crop_size=120, step=60)
