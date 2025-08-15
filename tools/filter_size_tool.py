import os
import glob
from PIL import Image


def filter_image_pairs_inplace(base_dir, min_size=720):
    """
    Kiểm tra các cặp ảnh LR/HR trong bộ dataset và xóa những cặp có ảnh HR < min_size x min_size

    Args:
        base_dir: Thư mục gốc chứa cấu trúc dataset (Train/Val/Test)
        min_size: Kích thước tối thiểu (cả chiều rộng và chiều cao)
    """
    # Danh sách các tập dữ liệu (Train, Val, Test)
    datasets = ['Train', 'Val', 'Test']

    # Thống kê tổng quan
    total_stats = {'total': 0, 'kept': 0, 'removed': 0}

    for dataset in datasets:
        print(f"\nXử lý tập dữ liệu {dataset}...")

        # Đường dẫn thư mục
        hr_dir = os.path.join(base_dir, dataset, 'HR')
        lr_dir = os.path.join(base_dir, dataset, 'LR')

        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
            print(f"Bỏ qua {dataset}: Không tìm thấy thư mục HR hoặc LR")
            continue

        # Thống kê cho dataset hiện tại
        dataset_stats = {'total': 0, 'kept': 0, 'removed': 0}

        # Lấy danh sách file HR
        hr_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            hr_files.extend(glob.glob(os.path.join(hr_dir, ext)))

        print(f"Tìm thấy {len(hr_files)} ảnh HR trong {dataset}")

        # Tạo danh sách các file cần xóa
        hr_to_remove = []
        lr_to_remove = []

        # Xử lý từng file HR
        for hr_path in hr_files:
            hr_filename = os.path.basename(hr_path)

            # Tìm file LR tương ứng (giả sử cùng tên file)
            lr_path = os.path.join(lr_dir, hr_filename)

            # Kiểm tra xem file LR có tồn tại không
            if os.path.exists(lr_path):
                dataset_stats['total'] += 1
                total_stats['total'] += 1

                # Đọc kích thước ảnh HR
                try:
                    with Image.open(hr_path) as img:
                        width, height = img.size

                    # Kiểm tra kích thước
                    if width < min_size or height < min_size:
                        # Thêm vào danh sách cần xóa
                        hr_to_remove.append(hr_path)
                        lr_to_remove.append(lr_path)
                        dataset_stats['removed'] += 1
                        total_stats['removed'] += 1
                    else:
                        dataset_stats['kept'] += 1
                        total_stats['kept'] += 1
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {hr_path}: {e}")
                    # Nếu có lỗi, chúng ta nên làm gì? Tùy vào yêu cầu, có thể xóa hoặc giữ lại cặp ảnh này
                    # Ở đây tôi quyết định không xóa để an toàn
                    print(f"Bỏ qua và giữ lại cặp ảnh này")

        # Xóa các file không đạt yêu cầu
        for file_path in hr_to_remove:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Không thể xóa file {file_path}: {e}")

        for file_path in lr_to_remove:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Không thể xóa file {file_path}: {e}")

        # Báo cáo thống kê cho dataset hiện tại
        print(f"Kết quả lọc {dataset}:")
        print(f"  - Tổng số cặp ảnh: {dataset_stats['total']}")
        print(f"  - Số cặp giữ lại (HR >= {min_size}x{min_size}): {dataset_stats['kept']}")
        print(f"  - Số cặp đã xóa: {dataset_stats['removed']}")

    # Báo cáo thống kê tổng thể
    print("\nThống kê tổng thể:")
    print(f"  - Tổng số cặp ảnh: {total_stats['total']}")
    print(f"  - Số cặp giữ lại (HR >= {min_size}x{min_size}): {total_stats['kept']}")
    print(f"  - Số cặp đã xóa: {total_stats['removed']}")
    print(f"  - Tỷ lệ loại bỏ: {total_stats['removed'] / total_stats['total'] * 100:.2f}% nếu có ảnh")


if __name__ == "__main__":
    # Thay đổi đường dẫn này theo nhu cầu của bạn
    base_directory = "D:\EnhanceVideo_ImageDLM\DIV2K+_scale_4x"  # Thư mục chứa cấu trúc Train/Val/Test

    # Thêm xác nhận để tránh xóa nhầm
    confirm = input(f"Bạn có chắc chắn muốn lọc và XÓA các cặp ảnh không đạt yêu cầu trong {base_directory}? (y/n): ")
    if confirm.lower() == 'y':
        filter_image_pairs_inplace(
            base_directory,
            min_size=480
        )
    else:
        print("Đã hủy thao tác.")