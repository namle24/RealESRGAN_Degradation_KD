import os
import cv2

# Đường dẫn gốc và đích
input_dir = 'D:\\EnhanceVideo_ImageDLM\\dataset'           # Thư mục chứa dữ liệu đã gộp
output_dir = 'D:\\EnhanceVideo_ImageDLM\\dataset_resized'  # Thư mục lưu dữ liệu đã resize

# Kích thước mục tiêu
HR_SIZE = (1024, 1024)
LR_SIZE = (512, 512)

# Tạo thư mục đích
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'HR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'LR'), exist_ok=True)

# Hàm resize ảnh
def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(input_folder):
        print(f"Thư mục không tồn tại: {input_folder}")
        return
    files = os.listdir(input_folder)
    print(f"Đang xử lý {len(files)} file trong {input_folder}")
    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        # Đọc ảnh
        img = cv2.imread(input_path)
        if img is None:
            print(f"Không thể đọc file: {input_path}")
            continue
        # Resize
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        # Lưu ảnh
        cv2.imwrite(output_path, resized_img)
        print(f"Đã resize và lưu: {output_path}")

# Resize từng tập
for split in ['train', 'val', 'test']:
    # Resize HR
    print(f"\nResize HR trong {split}:")
    resize_images(
        os.path.join(input_dir, split, 'HR'),
        os.path.join(output_dir, split, 'HR'),
        HR_SIZE
    )
    # Resize LR
    print(f"\nResize LR trong {split}:")
    resize_images(
        os.path.join(input_dir, split, 'LR'),
        os.path.join(output_dir, split, 'LR'),
        LR_SIZE
    )


# Kiểm tra kết quả
for split in ['train', 'val', 'test']:
    hr_count = len(os.listdir(os.path.join(output_dir, split, 'HR')))
    lr_count = len(os.listdir(os.path.join(output_dir, split, 'LR')))
    print(f"{split} - HR: {hr_count} ảnh, LR: {lr_count} ảnh")

print("Đã resize toàn bộ dữ liệu thành công!")