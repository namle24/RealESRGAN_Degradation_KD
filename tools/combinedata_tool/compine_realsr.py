import os
import shutil
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Đường dẫn gốc và đích
realsr_iccv_dir = 'D:\\EnhanceVideo_ImageDLM\\data\\RealSR (ICCV2019)'
realsr_v2_dir = 'D:\\EnhanceVideo_ImageDLM\\data\\RealSR(V2)'
realsr_v3_dir = 'D:\\EnhanceVideo_ImageDLM\\data\\RealSR(V3)'
output_dir = 'D:\\EnhanceVideo_ImageDLM\\RealSR_scale_4x'

# Tạo thư mục nếu chưa tồn tại
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'HR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'LR'), exist_ok=True)


# Hàm thu thập cặp HR-LR từ một bộ RealSR, tách riêng train và test
def collect_pairs(base_dir, dataset_prefix=''):
    train_pairs = []
    test_pairs = []

    for brand in ['Canon', 'Nikon']:
        for split in ['Train', 'Test']:
            for scale in ['3', '4']:
                folder_path = os.path.join(base_dir, brand, split, scale)
                if os.path.exists(folder_path):
                    logger.info(f"Đang kiểm tra thư mục: {folder_path}")
                    files = os.listdir(folder_path)
                    if files:
                        logger.info(f"Tìm thấy {len(files)} file: {files[:5]}...")  # In 5 file đầu để kiểm tra
                    else:
                        logger.warning(f"Thư mục rỗng: {folder_path}")
                        continue

                    file_dict = {f: os.path.join(folder_path, f) for f in files}
                    hr_files = [f for f in files if 'HR' in f.upper()]

                    for hr_file in hr_files:
                        hr_base = hr_file.split('_HR')[0]  # Lấy phần tên trước '_HR'
                        lr_file = f"{hr_base}_LR{scale}.png"  # Tạo tên LR dựa trên scale

                        if lr_file in file_dict:
                            hr_path = file_dict[hr_file]
                            lr_path = file_dict[lr_file]
                            # Tạo tên HR mới với scale (HR2, HR3, HR4)
                            hr_new_name = f"{dataset_prefix}{brand}_{split}_{scale}_{hr_base}_HR{scale}.png"
                            lr_new_name = f"{dataset_prefix}{brand}_{split}_{scale}_{lr_file}"

                            # Giữ nguyên phân loại train/test từ dataset gốc
                            if split == 'Train':
                                train_pairs.append((hr_path, lr_path, hr_new_name, lr_new_name))
                            else:  # 'Test'
                                test_pairs.append((hr_path, lr_path, hr_new_name, lr_new_name))

                            logger.debug(f"Đã tìm thấy cặp ({split}): {hr_new_name} - {lr_new_name}")
                        else:
                            logger.warning(f"Không tìm thấy LR cho {hr_file} ở scale {scale}: {lr_file}")
                else:
                    logger.warning(f"Thư mục không tồn tại: {folder_path}")

    logger.info(
        f"Tổng số cặp từ {base_dir}: {len(train_pairs) + len(test_pairs)} (Train: {len(train_pairs)}, Test: {len(test_pairs)})")
    return train_pairs, test_pairs


logger.info("Xử lý RealSR ICCV 2019:")
iccv_train_pairs, iccv_test_pairs = collect_pairs(realsr_iccv_dir, dataset_prefix='ICCV_')

logger.info("\nXử lý RealSR V3:")
v3_train_pairs, v3_test_pairs = collect_pairs(realsr_v3_dir, dataset_prefix='V3_')

logger.info("\nXử lý RealSR V2:")
v2_train_pairs, v2_test_pairs = collect_pairs(realsr_v2_dir, dataset_prefix='V2_')

# Gộp từ hai dataset
all_train_pairs = iccv_train_pairs + v3_train_pairs + v2_train_pairs
all_test_pairs = iccv_test_pairs + v3_test_pairs + v2_test_pairs

logger.info(f"Tổng số cặp HR-LR từ RealSR: {len(all_train_pairs) + len(all_test_pairs)}")
logger.info(f"- Train pairs: {len(all_train_pairs)}")
logger.info(f"- Test pairs: {len(all_test_pairs)}")

# Kiểm tra nếu rỗng
if not all_train_pairs and not all_test_pairs:
    raise ValueError("Không tìm thấy cặp HR-LR nào từ RealSR. Kiểm tra đường dẫn hoặc tên file.")

# Chia tập từ Train thành train (90%), val (10%)
# Test của dataset gốc sẽ được giữ làm test
if all_train_pairs:
    train_pairs, val_pairs = train_test_split(all_train_pairs, test_size=0.1, random_state=42)
    test_pairs = all_test_pairs
else:
    # Nếu không có train, chia test thành các phần
    train_pairs, temp = train_test_split(all_test_pairs, test_size=0.2, random_state=42)
    val_pairs, test_pairs = train_test_split(temp, test_size=0.5, random_state=42)
    logger.warning("Không có dữ liệu train từ dataset gốc, đã chia dữ liệu test thành train/val/test")

logger.info(f"Phân chia cuối cùng: Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")


# Hàm di chuyển file, không ghi đè dữ liệu cũ
def move_pairs(pairs, dest_hr_dir, dest_lr_dir):
    skipped = 0
    added = 0

    # Kiểm tra từng cặp
    for hr_path, lr_path, hr_name, lr_name in pairs:
        hr_dest = os.path.join(dest_hr_dir, hr_name)
        lr_dest = os.path.join(dest_lr_dir, lr_name)

        # Kiểm tra xem file gốc có tồn tại không
        if not os.path.exists(hr_path):
            logger.error(f"File nguồn HR không tồn tại: {hr_path}")
            skipped += 1
            continue

        if not os.path.exists(lr_path):
            logger.error(f"File nguồn LR không tồn tại: {lr_path}")
            skipped += 1
            continue

        # Kiểm tra xem đã tồn tại chưa
        if os.path.exists(hr_dest) or os.path.exists(lr_dest):
            logger.info(f"Bỏ qua cặp đã tồn tại: {hr_name} - {lr_name}")
            skipped += 1
        else:
            try:
                shutil.copy(hr_path, hr_dest)
                shutil.copy(lr_path, lr_dest)
                added += 1
            except Exception as e:
                logger.error(f"Lỗi khi sao chép {hr_path} -> {hr_dest} hoặc {lr_path} -> {lr_dest}: {str(e)}")
                skipped += 1

    logger.info(f"Đã thêm: {added} cặp, Bỏ qua: {skipped} cặp")
    return added, skipped


# Gộp dữ liệu
logger.info("Gộp vào train...")
train_added, train_skipped = move_pairs(train_pairs, os.path.join(output_dir, 'train', 'HR'),
                                        os.path.join(output_dir, 'train', 'LR'))

logger.info("Gộp vào val...")
val_added, val_skipped = move_pairs(val_pairs, os.path.join(output_dir, 'val', 'HR'),
                                    os.path.join(output_dir, 'val', 'LR'))

logger.info("Gộp vào test...")
test_added, test_skipped = move_pairs(test_pairs, os.path.join(output_dir, 'test', 'HR'),
                                      os.path.join(output_dir, 'test', 'LR'))

# Kiểm tra tên file trong các thư mục cuối cùng
for split in ['train', 'val', 'test']:
    hr_files = os.listdir(os.path.join(output_dir, split, 'HR'))
    lr_files = os.listdir(os.path.join(output_dir, split, 'LR'))

    # Đếm số file chứa 'Train' và 'Test' trong tên
    train_in_hr = sum(1 for f in hr_files if '_Train_' in f)
    test_in_hr = sum(1 for f in hr_files if '_Test_' in f)

    # Thống kê
    logger.info(f"{split.upper()} - Tổng số file: HR: {len(hr_files)}, LR: {len(lr_files)}")
    logger.info(f"{split.upper()} - File có '_Train_' trong tên: {train_in_hr}/{len(hr_files)}")
    logger.info(f"{split.upper()} - File có '_Test_' trong tên: {test_in_hr}/{len(hr_files)}")

    # Cảnh báo nếu có lỗi phân loại
    if split == 'train' and test_in_hr > 0:
        logger.warning(f"Phát hiện {test_in_hr} file có '_Test_' trong thư mục train/HR!")

    if split == 'test' and train_in_hr > 0:
        logger.warning(f"Phát hiện {train_in_hr} file có '_Train_' trong thư mục test/HR!")

logger.info("Đã hoàn thành việc gộp RealSR (ICCV 2019), RealSR(V2) và RealSR (V3)!")