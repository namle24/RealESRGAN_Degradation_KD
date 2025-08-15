import os

# Đường dẫn thư mục chứa dữ liệu đã gộp
dataset_dir = 'D:\\EnhanceVideo_ImageDLM\\dataset_resized'


# Hàm đồng bộ cặp HR-LR trong một tập
def sync_pairs(hr_dir, lr_dir):
    hr_files = set(os.listdir(hr_dir))
    lr_files = set(os.listdir(lr_dir))

    # Tạo từ điển để so sánh cặp dựa trên tên gốc (bỏ hậu tố _HR, _LR2/3/4)
    hr_dict = {f: f.split('_HR')[0] for f in hr_files}  # Phần trước _HR
    lr_dict = {f: f.split('_LR')[0] for f in lr_files}  # Phần trước _LR

    # Tìm cặp khớp nhau
    synced_pairs = []
    unmatched_hr = []
    unmatched_lr = list(lr_files)  # Bắt đầu với tất cả LR

    for hr_file in hr_files:
        hr_base = hr_dict[hr_file]
        lr_match = None
        for lr_file in lr_files:
            lr_base = lr_dict[lr_file]
            if hr_base == lr_base:
                lr_match = lr_file
                break
        if lr_match:
            synced_pairs.append((hr_file, lr_match))
            if lr_match in unmatched_lr:
                unmatched_lr.remove(lr_match)  # Loại LR đã khớp khỏi danh sách thừa
        else:
            unmatched_hr.append(hr_file)

    return synced_pairs, unmatched_hr, unmatched_lr


# Đồng bộ từng tập
for split in ['train', 'val', 'test']:
    hr_dir = os.path.join(dataset_dir, split, 'HR')
    lr_dir = os.path.join(dataset_dir, split, 'LR')

    print(f"\nĐồng bộ {split}:")
    synced_pairs, unmatched_hr, unmatched_lr = sync_pairs(hr_dir, lr_dir)

    # In kết quả
    print(f"Số cặp HR-LR khớp: {len(synced_pairs)}")
    print(f"HR không có LR: {len(unmatched_hr)} - {unmatched_hr[:5]}")
    print(f"LR không có HR: {len(unmatched_lr)} - {unmatched_lr[:5]}")

    # Xóa file LR không có HR
    for hr_file in unmatched_hr:
        hr_path = os.path.join(hr_dir, hr_file)
        if os.path.exists(hr_path):
            os.remove(hr_path)
            print(f"Đã xóa {hr_file} từ LR")
        else:
            print(f"Không tìm thấy file để xóa: {hr_path}")

# Kiểm tra lại số lượng
for split in ['train', 'val', 'test']:
    hr_count = len(os.listdir(os.path.join(dataset_dir, split, 'HR')))
    lr_count = len(os.listdir(os.path.join(dataset_dir, split, 'LR')))
    print(f"{split} - HR: {hr_count}, LR: {lr_count}")

print("Đã đồng bộ cặp HR-LR thành công!")