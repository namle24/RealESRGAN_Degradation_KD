import cv2
import numpy as np

def add_label_below(image, label, font_scale=0.6, thickness=1):
    """Thêm label bên dưới ảnh. In đậm nếu là 'Real-ESRGAN teacher(OUR)' hoặc 'Real-ESRGAN student(OUR)'."""
    h, w = image.shape[:2]
    space = 30  # khoảng không dưới ảnh để ghi chữ
    labeled_img = np.full((h + space, w, 3), 255, dtype=np.uint8)
    labeled_img[0:h, 0:w] = image

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h + (space + text_size[1]) // 2

    # In đậm nếu là teacher hoặc student
    if label in ["Real-ESRGAN teacher(OUR)", "Real-ESRGAN student(OUR)"]:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cv2.putText(labeled_img, label, (text_x + dx, text_y + dy),
                            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    else:
        cv2.putText(labeled_img, label, (text_x, text_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return labeled_img

def add_padding(img, pad_h, pad_w):
    """Thêm padding trắng xung quanh ảnh."""
    return cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def crop_and_compare_grid_padded(
    gt_path, lq_path, realesrgan_path, realesrgan_teacher_our_path, realesrgan_student_our_path,
    crop_x, crop_y, crop_w, crop_h,
    pad_h=10, pad_w=10,
    save_path=None
):
    # Load ảnh
    img_gt = cv2.imread(gt_path)
    img_lq = cv2.imread(lq_path)
    img_realesrgan = cv2.imread(realesrgan_path)
    img_realesrgan_teacher_our = cv2.imread(realesrgan_teacher_our_path)
    img_realesrgan_student_our = cv2.imread(realesrgan_student_our_path)

    if img_gt is None or img_lq is None or img_realesrgan is None or img_realesrgan_teacher_our is None or img_realesrgan_student_our is None:
        print("Error loading images. Please check the file paths.")
        return

    # Resize về cùng kích thước gốc
    h, w = img_gt.shape[:2]
    imgs = [img_lq, img_realesrgan, img_realesrgan_teacher_our, img_realesrgan_student_our]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]
    img_list = [img_gt] + imgs

    labels = ["Ground Truth", "Low Quality", "Real-ESRGAN", "Real-ESRGAN teacher(OUR)", "Real-ESRGAN student(OUR)"]

    labeled_crops = []
    for img, label in zip(img_list, labels):
        crop = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        labeled_crop = add_label_below(crop, label)
        padded_crop = add_padding(labeled_crop, pad_h, pad_w)
        labeled_crops.append(padded_crop)

    # Resize tất cả về cùng chiều cao, giữ tỷ lệ gốc
    min_height = min(crop.shape[0] for crop in labeled_crops)
    labeled_crops = [
        cv2.resize(crop, (int(crop.shape[1] * (min_height / crop.shape[0])), min_height))
        for crop in labeled_crops
    ]

    # Ghép tất cả ảnh thành 1 hàng ngang
    grid = np.hstack(labeled_crops)

    # Hiển thị
    cv2.imshow("Comparison Grid (Padded)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh nếu cần
    if save_path:
        cv2.imwrite(save_path, grid)
        print(f"Saved to {save_path}")

# Gọi hàm
crop_and_compare_grid_padded(
    r"D:\Download\test\RealSR(V3)\Nikon\Test\4\HR\Nikon_041.png",
    r"D:\Download\test\RealSR(V3)\Nikon\Test\4\LRx4\Nikon_041.png",
    r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_Orgianal_x4_V2\visualization\RealSR(V2)_Nikon\Nikon_041_RealESRGAN_Orgianal_x4.png",
    r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_x4_RealSR(V2)_Nikon\visualization\RealSR(V2)_Nikon\Nikon_041_RealESRGAN_x4.png",
    r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_student_x4_RealSR_V2\visualization\RealSR_V2\Nikon_041_RealESRGAN_student_x4_RealSR_V2.png",
    crop_x=300, crop_y=300,
    crop_w=1080, crop_h=720,
    pad_h=10, pad_w=10,
    save_path="comparison_grid_padded_student_2.png"
)
