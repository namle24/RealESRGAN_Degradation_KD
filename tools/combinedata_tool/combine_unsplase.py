import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn Unsplash và đích
unsplash_dir = 'D:\\EnhanceVideo_ImageDLM\data\\Image Super Resolution - Unsplash'
unsplash_hr_dir = os.path.join(unsplash_dir, 'high res')
unsplash_lr_dir = os.path.join(unsplash_dir, 'low res')
output_dir = 'D:\\EnhanceVideo_ImageDLM\\dataset'

# Kiểm tra thư mục
if not os.path.exists(unsplash_hr_dir):
    print(f"Thư mục HR không tồn tại: {unsplash_hr_dir}")
if not os.path.exists(unsplash_lr_dir):
    print(f"Thư mục LR không tồn tại: {unsplash_lr_dir}")

# Tạo thư mục đích
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'HR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'LR'), exist_ok=True)

# Thu thập cặp HR-LR từ Unsplash (dùng cả 3 scale)
unsplash_pairs = []
hr_files = os.listdir(unsplash_hr_dir)
lr_files = os.listdir(unsplash_lr_dir)
print(f"Tìm thấy {len(hr_files)} file trong high res: {hr_files[:5]}")
print(f"Tìm thấy {len(lr_files)} file trong low res: {lr_files[:5]}")

for hr_file in hr_files:
    if hr_file.lower().endswith(('.png', '.jpg')):
        hr_base = hr_file.split('.')[0]  # Ví dụ: 1
        hr_path = os.path.join(unsplash_hr_dir, hr_file)

        # Tạo cặp cho từng scale (2, 4, 6)
        for scale in ['2', '4', '6']:
            lr_file = f"{hr_base}_{scale}.jpg"  # Ví dụ: 1_2.jpg
            lr_path = os.path.join(unsplash_lr_dir, lr_file)

            if os.path.exists(lr_path):
                hr_new_name = f"Unsplash_{hr_base}_{scale}_HR.jpg"  # Unsplash_1_2_HR.jpg
                lr_new_name = f"Unsplash_{hr_base}_{scale}_LR.jpg"  # Unsplash_1_2_LR.jpg
                unsplash_pairs.append((hr_path, lr_path, hr_new_name, lr_new_name))
            else:
                print(f"Không tìm thấy LR x{scale} cho {hr_file} tại {lr_path}")

print(f"Tìm thấy {len(unsplash_pairs)} cặp từ Unsplash (tất cả scale)")

# Kiểm tra nếu rỗng
if not unsplash_pairs:
    raise ValueError("Không tìm thấy cặp HR-LR nào. Kiểm tra thư mục Unsplash.")

# Chia dữ liệu: 80% train, 10% val, 10% test
train_pairs, temp_pairs = train_test_split(unsplash_pairs, test_size=0.2, random_state=42)
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)


# Hàm di chuyển file
def move_pairs(pairs, dest_hr_dir, dest_lr_dir):
    for hr_path, lr_path, hr_name, lr_name in pairs:
        shutil.copy(hr_path, os.path.join(dest_hr_dir, hr_name))
        shutil.copy(lr_path, os.path.join(dest_lr_dir, lr_name))


# Di chuyển file
move_pairs(train_pairs, os.path.join(output_dir, 'train', 'HR'), os.path.join(output_dir, 'train', 'LR'))
move_pairs(val_pairs, os.path.join(output_dir, 'val', 'HR'), os.path.join(output_dir, 'val', 'LR'))
move_pairs(test_pairs, os.path.join(output_dir, 'test', 'HR'), os.path.join(output_dir, 'test', 'LR'))

# Kiểm tra số lượng
for split in ['train', 'val', 'test']:
    hr_count = len(os.listdir(os.path.join(output_dir, split, 'HR')))
    lr_count = len(os.listdir(os.path.join(output_dir, split, 'LR')))
    print(f"{split} - HR: {hr_count}, LR: {lr_count}")

print("Đã chia Unsplash thành train, val, test thành công!")