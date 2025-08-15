import os
import glob
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re


def analyze_realsr_resolutions(root_dir):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    # Duyệt qua cấu trúc thư mục
    for camera_dir in os.listdir(root_dir):
        camera_path = os.path.join(root_dir, camera_dir)
        if not os.path.isdir(camera_path):
            continue

        print(f"Đang phân tích camera: {camera_dir}")

        # Duyệt qua Train/Test
        for set_dir in os.listdir(camera_path):
            set_path = os.path.join(camera_path, set_dir)
            if not os.path.isdir(set_path) or set_dir not in ['Train', 'Test']:
                continue

            # Duyệt qua các tỉ lệ (2, 3, 4)
            for scale_dir in os.listdir(set_path):
                scale_path = os.path.join(set_path, scale_dir)
                if not os.path.isdir(scale_path) or not scale_dir.isdigit():
                    continue

                print(f"  Đang phân tích {camera_dir}/{set_dir}/{scale_dir}")

                # Tìm tất cả các file ảnh
                image_files = []
                for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg", "*.tif", "*.tiff"]:
                    image_files.extend(glob.glob(os.path.join(scale_path, "**", ext), recursive=True))

                if not image_files:
                    print(f"    Không tìm thấy ảnh trong {scale_path}")
                    continue

                # Phân loại HR và LR dựa vào tên file
                hr_files = [f for f in image_files if is_hr_file(f)]
                lr_files = [f for f in image_files if is_lr_file(f)]

                # Nếu không thể phân loại bằng tên file, thử phân loại bằng thư mục
                if not hr_files and not lr_files:
                    print(f"    Không thể phân loại HR/LR bằng tên file, thử phân loại bằng thư mục...")
                    for img_path in image_files:
                        if "HR" in img_path.upper():
                            hr_files.append(img_path)
                        elif "LR" in img_path.upper():
                            lr_files.append(img_path)

                # Nếu vẫn không thể phân loại, giả sử phân chia đều
                if not hr_files and not lr_files:
                    print(f"    Không thể phân loại bằng thư mục, giả sử phân chia đều...")
                    image_files.sort()  # Sắp xếp để đảm bảo tính nhất quán
                    half = len(image_files) // 2
                    hr_files = image_files[:half]
                    lr_files = image_files[half:]

                # Phân tích độ phân giải cho HR
                hr_results = analyze_images(hr_files)
                if hr_results:
                    results[camera_dir][set_dir][scale_dir]["HR"] = hr_results

                # Phân tích độ phân giải cho LR
                lr_results = analyze_images(lr_files)
                if lr_results:
                    results[camera_dir][set_dir][scale_dir]["LR"] = lr_results

    return results


def is_hr_file(filepath):
    """Kiểm tra xem file có phải là HR dựa vào tên file"""
    filename = os.path.basename(filepath).lower()
    return "hr" in filename or "_h" in filename or "high" in filename


def is_lr_file(filepath):
    """Kiểm tra xem file có phải là LR dựa vào tên file"""
    filename = os.path.basename(filepath).lower()
    return "lr" in filename or "_l" in filename or "low" in filename


def analyze_images(image_files):
    """Phân tích độ phân giải của danh sách ảnh"""
    if not image_files:
        return None

    resolutions = []
    areas = []
    widths = []
    heights = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                resolutions.append((width, height))
                areas.append(width * height)
                widths.append(width)
                heights.append(height)
        except Exception as e:
            print(f"    Lỗi khi mở ảnh {img_path}: {e}")

    if not resolutions:
        return None

    return {
        "total_images": len(resolutions),
        "min_resolution": min(resolutions, key=lambda x: x[0] * x[1]),
        "max_resolution": max(resolutions, key=lambda x: x[0] * x[1]),
        "most_common": Counter(resolutions).most_common(1)[0],
        "all_resolutions": Counter(resolutions),
        "pixel_areas": areas,
        "widths": widths,
        "heights": heights
    }


def format_resolution(res):
    """Format độ phân giải để in ra dạng đẹp hơn"""
    return f"{res[0]}×{res[1]}"


def print_realsr_results(results):
    """In kết quả phân tích"""
    for camera, camera_data in results.items():
        print(f"\n{'=' * 60}")
        print(f"KẾT QUẢ CHO {camera}")
        print(f"{'=' * 60}")

        for set_type, set_data in camera_data.items():
            print(f"\n{'-' * 50}")
            print(f"Loại dữ liệu: {set_type}")
            print(f"{'-' * 50}")

            for scale, scale_data in set_data.items():
                print(f"\nTỉ lệ phóng đại: {scale}")

                # Thống kê số lượng ảnh HR và LR
                hr_count = scale_data.get("HR", {}).get("total_images", 0)
                lr_count = scale_data.get("LR", {}).get("total_images", 0)
                print(f"Số lượng ảnh: HR={hr_count}, LR={lr_count}, Tổng={hr_count + lr_count}")

                for img_type, stats in scale_data.items():
                    print(f"\n  * {img_type} ({stats['total_images']} ảnh)")

                    min_res = stats['min_resolution']
                    max_res = stats['max_resolution']
                    most_common = stats['most_common']

                    print(
                        f"    Độ phân giải nhỏ nhất: {format_resolution(min_res)} ({min_res[0] * min_res[1]:,} pixels)")
                    print(
                        f"    Độ phân giải lớn nhất: {format_resolution(max_res)} ({max_res[0] * max_res[1]:,} pixels)")
                    print(f"    Độ phân giải phổ biến nhất: {format_resolution(most_common[0])} ({most_common[1]} ảnh)")

                    # Top 3 độ phân giải phổ biến
                    print("    Top 3 độ phân giải phổ biến:")
                    for i, (res, count) in enumerate(stats['all_resolutions'].most_common(3), 1):
                        print(f"      {i}. {format_resolution(res)} ({count} ảnh)")

                    # Tỉ lệ thực tế giữa HR và LR (nếu có cả hai)
                    if img_type == "HR" and "LR" in scale_data:
                        hr_common_res = most_common[0]
                        lr_common_res = scale_data["LR"]["most_common"][0]
                        w_ratio = hr_common_res[0] / lr_common_res[0]
                        h_ratio = hr_common_res[1] / lr_common_res[1]
                        print(f"    Tỉ lệ thực tế HR/LR: {w_ratio:.2f}x (rộng), {h_ratio:.2f}x (cao)")


def plot_realsr_histograms(results, output_dir="realsr_charts"):
    """Vẽ biểu đồ cho phân bố độ phân giải của RealSR"""
    os.makedirs(output_dir, exist_ok=True)

    for camera, camera_data in results.items():
        camera_dir = os.path.join(output_dir, camera)
        os.makedirs(camera_dir, exist_ok=True)

        for set_type, set_data in camera_data.items():
            set_dir = os.path.join(camera_dir, set_type)
            os.makedirs(set_dir, exist_ok=True)

            for scale, scale_data in set_data.items():
                # 1. Biểu đồ so sánh độ phân giải HR và LR
                if "HR" in scale_data and "LR" in scale_data:
                    plt.figure(figsize=(12, 6))

                    hr_areas = scale_data["HR"]["pixel_areas"]
                    lr_areas = scale_data["LR"]["pixel_areas"]

                    plt.hist([hr_areas, lr_areas], bins=20, label=['HR', 'LR'], alpha=0.7,
                             color=['skyblue', 'salmon'], edgecolor='black')

                    plt.title(f'So sánh diện tích pixel - {camera} {set_type} (Scale {scale})', fontsize=14)
                    plt.xlabel('Diện tích (pixels)', fontsize=12)
                    plt.ylabel('Số lượng ảnh', fontsize=12)
                    plt.legend()
                    plt.ticklabel_format(style='plain', axis='x')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(set_dir, f'scale{scale}_comparison.png'), dpi=300)
                    plt.close()

                # 2. Biểu đồ scatter cho từng loại
                for img_type, stats in scale_data.items():
                    # Scatter plot chiều rộng vs chiều cao
                    plt.figure(figsize=(10, 10))

                    widths = stats["widths"]
                    heights = stats["heights"]

                    plt.scatter(widths, heights, alpha=0.7, s=50)
                    plt.title(f'Phân bố độ phân giải - {camera} {set_type} {img_type} (Scale {scale})', fontsize=14)
                    plt.xlabel('Chiều rộng (pixels)', fontsize=12)
                    plt.ylabel('Chiều cao (pixels)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(set_dir, f'scale{scale}_{img_type}_scatter.png'), dpi=300)
                    plt.close()

                    # Histogram phân bố diện tích
                    plt.figure(figsize=(12, 6))
                    areas = stats["pixel_areas"]
                    plt.hist(areas, bins=20, color='lightblue', edgecolor='black')
                    plt.title(f'Phân bố diện tích pixel - {camera} {set_type} {img_type} (Scale {scale})', fontsize=14)
                    plt.xlabel('Diện tích (pixels)', fontsize=12)
                    plt.ylabel('Số lượng ảnh', fontsize=12)
                    plt.ticklabel_format(style='plain', axis='x')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(set_dir, f'scale{scale}_{img_type}_area_hist.png'), dpi=300)
                    plt.close()

                    # Top độ phân giải phổ biến
                    plt.figure(figsize=(14, 6))
                    top_n = min(10, len(stats['all_resolutions']))
                    top_resolutions = stats['all_resolutions'].most_common(top_n)

                    resolution_labels = [f"{w}×{h}" for (w, h), _ in top_resolutions]
                    counts = [count for _, count in top_resolutions]

                    plt.bar(range(len(counts)), counts, color='lightgreen', edgecolor='black')
                    plt.xticks(range(len(counts)), resolution_labels, rotation=45, ha='right', fontsize=10)
                    plt.title(f'Top {top_n} độ phân giải - {camera} {set_type} {img_type} (Scale {scale})', fontsize=14)
                    plt.xlabel('Độ phân giải', fontsize=12)
                    plt.ylabel('Số lượng ảnh', fontsize=12)
                    plt.grid(axis='y', alpha=0.3)

                    # Thêm giá trị số lượng trên mỗi cột
                    for i, count in enumerate(counts):
                        plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)

                    plt.tight_layout()
                    plt.savefig(os.path.join(set_dir, f'scale{scale}_{img_type}_top_res.png'), dpi=300)
                    plt.close()


def generate_realsr_summary(results, output_file="realsr_analysis_summary.txt"):
    """Tạo báo cáo tổng hợp cho RealSR"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO PHÂN TÍCH ĐỘ PHÂN GIẢI ẢNH REALSR\n")
        f.write("=" * 60 + "\n\n")

        # Tổng hợp số lượng ảnh
        f.write("TỔNG HỢP SỐ LƯỢNG ẢNH\n")
        f.write("-" * 60 + "\n")

        total_hr = 0
        total_lr = 0

        # Bảng thống kê
        f.write(f"{'Camera':<10} | {'Set':<8} | {'Scale':<5} | {'HR':<6} | {'LR':<6} | {'Tổng':<6}\n")
        f.write("-" * 60 + "\n")

        for camera, camera_data in results.items():
            camera_hr = 0
            camera_lr = 0

            for set_type, set_data in camera_data.items():
                set_hr = 0
                set_lr = 0

                for scale, scale_data in set_data.items():
                    hr_count = scale_data.get("HR", {}).get("total_images", 0)
                    lr_count = scale_data.get("LR", {}).get("total_images", 0)

                    set_hr += hr_count
                    set_lr += lr_count

                    f.write(
                        f"{camera:<10} | {set_type:<8} | {scale:<5} | {hr_count:<6} | {lr_count:<6} | {hr_count + lr_count:<6}\n")

                camera_hr += set_hr
                camera_lr += set_lr
                f.write(
                    f"{camera:<10} | {set_type:<8} | {'Tổng':<5} | {set_hr:<6} | {set_lr:<6} | {set_hr + set_lr:<6}\n")
                f.write("-" * 60 + "\n")

            total_hr += camera_hr
            total_lr += camera_lr

        f.write(f"{'TỔNG CỘNG':<26} | {total_hr:<6} | {total_lr:<6} | {total_hr + total_lr:<6}\n\n")

        # Chi tiết từng dataset
        for camera, camera_data in results.items():
            f.write(f"\n{camera.upper()}\n")
            f.write("=" * 60 + "\n")

            for set_type, set_data in camera_data.items():
                f.write(f"\n{set_type}\n")
                f.write("-" * 40 + "\n")

                for scale, scale_data in set_data.items():
                    f.write(f"\nTỉ lệ phóng đại: {scale}\n")

                    for img_type, stats in scale_data.items():
                        f.write(f"\n  * {img_type} ({stats['total_images']} ảnh)\n")

                        min_res = stats['min_resolution']
                        max_res = stats['max_resolution']
                        most_common = stats['most_common']

                        f.write(
                            f"    Độ phân giải nhỏ nhất: {format_resolution(min_res)} ({min_res[0] * min_res[1]:,} pixels)\n")
                        f.write(
                            f"    Độ phân giải lớn nhất: {format_resolution(max_res)} ({max_res[0] * max_res[1]:,} pixels)\n")
                        f.write(
                            f"    Độ phân giải phổ biến nhất: {format_resolution(most_common[0])} ({most_common[1]} ảnh)\n")

                        # Top 5 độ phân giải phổ biến
                        f.write("    Top 5 độ phân giải phổ biến:\n")
                        for i, (res, count) in enumerate(stats['all_resolutions'].most_common(5), 1):
                            f.write(
                                f"      {i}. {format_resolution(res):15} - {count:4} ảnh ({res[0] * res[1]:,} pixels)\n")

                    # Tỉ lệ thực tế giữa HR và LR (nếu có cả hai)
                    if "HR" in scale_data and "LR" in scale_data:
                        hr_data = scale_data["HR"]
                        lr_data = scale_data["LR"]

                        hr_common_res = hr_data["most_common"][0]
                        lr_common_res = lr_data["most_common"][0]

                        w_ratio = hr_common_res[0] / lr_common_res[0]
                        h_ratio = hr_common_res[1] / lr_common_res[1]

                        f.write(f"\n  * Tỉ lệ thực tế HR/LR:\n")
                        f.write(f"    Chiều rộng: {w_ratio:.2f}x\n")
                        f.write(f"    Chiều cao: {h_ratio:.2f}x\n")
                        f.write(f"    Tỉ lệ theo tên thư mục: {scale}x\n\n")


# Sử dụng hàm
if __name__ == "__main__":
    root_dir = "D:\\EnhanceVideo_ImageDLM\\data\\RealSR (ICCV2019)"  # Thay đổi đường dẫn này tới thư mục gốc RealSR
    results = analyze_realsr_resolutions(root_dir)

    # In kết quả ra màn hình
    print_realsr_results(results)

    # Tạo biểu đồ phân tích
    plot_realsr_histograms(results)

    # Tạo báo cáo tổng hợp
    generate_realsr_summary(results)

    print("\nĐã hoàn thành phân tích. Các biểu đồ được lưu trong thư mục 'realsr_charts'")
    print("Báo cáo tổng hợp được lưu trong file 'realsr_analysis_summary.txt'")