import cv2
import matplotlib.pyplot as plt

def crop_and_compare_two_images(img1_path, img2_path, box_img1, scale_factor=4, zoom_scale=3,
                                 save_annotated1=None, save_annotated2=None,
                                 save_crop1=None, save_crop2=None):
    """
    So sánh vùng crop tương ứng trên 2 ảnh có tỉ lệ khác nhau.

    Parameters:
    - img1_path: ảnh gốc (low-res).
    - img2_path: ảnh đã phóng đại (high-res).
    - box_img1: (x, y, w, h) trên ảnh gốc.
    - scale_factor: tỉ lệ kích thước giữa ảnh 2 và ảnh 1.
    - zoom_scale: tỉ lệ phóng to crop để hiển thị rõ.
    """

    # Đọc ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Không đọc được ảnh đầu vào")

    # Chuyển sang RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Vùng crop trên ảnh 1
    x1, y1, w1, h1 = box_img1

    # Vùng tương ứng trên ảnh 2 (scale gấp lên)
    x2, y2, w2, h2 = [int(v * scale_factor) for v in box_img1]

    # Crop & resize
    crop1 = img1_rgb[y1:y1+h1, x1:x1+w1]
    crop2 = img2_rgb[y2:y2+h2, x2:x2+w2]
    zoom1 = cv2.resize(crop1, (w1 * zoom_scale, h1 * zoom_scale), interpolation=cv2.INTER_NEAREST)
    zoom2 = cv2.resize(crop2, (w2 * zoom_scale, h2 * zoom_scale), interpolation=cv2.INTER_NEAREST)

    # Vẽ ô đỏ
    annotated1 = img1_rgb.copy()
    annotated2 = img2_rgb.copy()
    red_rgb = (255, 0, 0)
    cv2.rectangle(annotated1, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), thickness=1)
    cv2.rectangle(annotated2, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), thickness=1 * scale_factor)

    # Hiển thị
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(annotated1)
    plt.title("Ảnh gốc + ô crop")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(annotated2)
    plt.title("Ảnh phóng đại + ô crop tương ứng")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(zoom1)
    plt.title("Crop từ ảnh gốc")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(zoom2)
    plt.title("Crop từ ảnh phóng đại")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Lưu nếu cần
    if save_annotated1:
        cv2.imwrite(save_annotated1, cv2.cvtColor(annotated1, cv2.COLOR_RGB2BGR))
    if save_annotated2:
        cv2.imwrite(save_annotated2, cv2.cvtColor(annotated2, cv2.COLOR_RGB2BGR))
    if save_crop1:
        cv2.imwrite(save_crop1, cv2.cvtColor(zoom1, cv2.COLOR_RGB2BGR))
    if save_crop2:
        cv2.imwrite(save_crop2, cv2.cvtColor(zoom2, cv2.COLOR_RGB2BGR))
img1_path = r"D:\Download\test\RealSR(V3)\Canon\Test\4\LRx4\Canon_038.png"
img2_path = r"D:\EnhanceVideo_ImageDLM\BasicSR\results\RealESRGAN_student_degradation_x4_RealSR_V3\visualization\RealSR_V3\Canon_038_RealESRGAN_student_degradation_x4_RealSR_V3.png"
box = (150, 50, 120, 120)

crop_and_compare_two_images(
    img1_path, img2_path,
    box_img1=box,
    scale_factor=4,
    zoom_scale=4,
    save_annotated1='annotated_input.png',
    save_annotated2='annotated_output.png',
    save_crop1='crop_input.png',
    save_crop2='crop_output.png'
)
