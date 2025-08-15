import cv2
import numpy as np

def add_label_below(image, label, font_scale=0.6, thickness=1):
    """Thêm label bên dưới ảnh. In đậm nếu là 'RealDegra-ESRGAN'."""
    h, w = image.shape[:2]
    space = 30
    labeled_img = np.full((h + space, w, 3), 255, dtype=np.uint8)
    labeled_img[0:h, 0:w] = image

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h + (space + text_size[1]) // 2

    # In đậm nếu là RealDegra-ESRGAN
    if label == "Real-ESRGAN(OUR)":
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cv2.putText(labeled_img, label, (text_x + dx, text_y + dy),
                            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    else:
        cv2.putText(labeled_img, label, (text_x, text_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return labeled_img

def add_padding(img, pad_h, pad_w):
    """Thêm padding xung quanh ảnh."""
    return cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def resize_image(image, target_height):
    """Resize ảnh để có chiều cao tối đa là target_height, giữ tỷ lệ khung hình."""
    h, w = image.shape[:2]
    if h > target_height:
        scale = target_height / h
        new_size = (int(w * scale), target_height)
        return cv2.resize(image, new_size)
    return image

def compare_full_images_resized_grid(
    gt_path, lq_path, realesrgan_path, realdegra_our_path,
    pad_h=10, pad_w=10,
    save_path=None
):
    # Load ảnh
    paths = [gt_path, lq_path, realesrgan_path, realdegra_our_path]
    labels = ["Ground Truth", "Low Quality", "Real-ESRGAN", "Real-ESRGAN(OUR)"]
    imgs = [cv2.imread(p) for p in paths]

    # Resize tất cả ảnh về chiều cao tối đa 360
    imgs_resized = [resize_image(img, 360) for img in imgs]

    # Resize lq_image về kích thước của gt_image
    img_lq_resized = cv2.resize(imgs_resized[1], imgs_resized[0].shape[1::-1])

    # Cập nhật danh sách ảnh đã resize
    imgs_resized = [imgs_resized[0], img_lq_resized, imgs_resized[2], imgs_resized[3]]

    # Gán label + padding
    labeled_imgs = [
        add_padding(add_label_below(img, label), pad_h, pad_w)
        for img, label in zip(imgs_resized, labels)
    ]

    # Ghép lưới 2x2
    row1 = np.hstack(labeled_imgs[:2])
    row2 = np.hstack(labeled_imgs[2:])
    grid = np.vstack([row1, row2])

    # Hiển thị
    cv2.imshow("Comparison Grid (Resized)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, grid)
        print(f"Saved to {save_path}")


# 👉 Gọi hàm: khoảng cách giữa ảnh = 10px
compare_full_images_resized_grid(
    r"D:\Download\test\RealSR(V3)\Canon\Test\4\LRx4\Canon_014.png",
    r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_student_degradation_x4_RealSR_V3\visualization\RealSR_V3\Canon_014_RealESRGAN_student_degradation_x4_RealSR_V3.png",
    r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_Orgianal_x4_V2\visualization\Set14\bridge_RealESRGAN_Orgianal_x4.png",
    r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_x4_RealSR(V2)_Nikon\visualization\Set14\bridge_RealESRGAN_x4.png",
    pad_h=10, pad_w=10,
    save_path="comparison_grid_padded_2x2_no_crop_7.png"
)