import os
import glob
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def analyze_sr_dataset(root_dir):
    """Phân tích độ phân giải của ảnh trong bộ dữ liệu SR có cấu trúc Train/Val/Test với HR/LR"""
    results = {}

    # Phân tích qua các tập Train, Val, Test
    for dataset_split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, dataset_split)
        if not os.path.exists(split_dir):
            print(f"Thư mục {split_dir} không tồn tại, đang bỏ qua...")
            continue

        print(f"\nĐang phân tích tập {dataset_split}...")
        split_results = {}

        # Phân tích HR
        hr_dir = os.path.join(split_dir, "HR")
        if os.path.exists(hr_dir):
            print(f"  Đang phân tích thư mục HR...")
            hr_stats = analyze_directory(hr_dir)
            if hr_stats:
                split_results["HR"] = hr_stats

        # Phân tích LR
        lr_dir = os.path.join(split_dir, "LR")
        if os.path.exists(lr_dir):
            print(f"  Đang phân tích thư mục LR...")
            lr_stats = analyze_directory(lr_dir)
            if lr_stats:
                split_results["LR"] = lr_stats

        if split_results:
            results[dataset_split] = split_results

    return results


def analyze_directory(directory):
    """Phân tích độ phân giải của tất cả ảnh trong một thư mục"""
    # Tìm tất cả các file ảnh
    image_files = []
    for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg", "*.tif", "*.tiff", "*.webp"]:
        image_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))

    if not image_files:
        print(f"  Không tìm thấy ảnh trong {directory}")
        return None

    print(f"  Tìm thấy {len(image_files)} ảnh")

    # Phân tích độ phân giải
    resolutions = []
    areas = []
    widths = []
    heights = []
    aspect_ratios = []

    for i, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                resolutions.append((width, height))
                areas.append(width * height)
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height if height > 0 else 0)

            # Hiển thị tiến độ
            if (i + 1) % 100 == 0 or i + 1 == len(image_files):
                print(f"  Đã phân tích {i + 1}/{len(image_files)} ảnh")

        except Exception as e:
            print(f"  Lỗi khi mở ảnh {img_path}: {e}")

    if not resolutions:
        return None

    # Tính các phân phối
    width_dist = Counter(widths)
    height_dist = Counter(heights)
    resolution_dist = Counter(resolutions)

    # Lấy các giá trị thống kê
    return {
        "total_images": len(resolutions),
        "min_resolution": min(resolutions, key=lambda x: x[0] * x[1]),
        "max_resolution": max(resolutions, key=lambda x: x[0] * x[1]),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "min_area": min(areas),
        "max_area": max(areas),
        "most_common_resolution": resolution_dist.most_common(1)[0],
        "most_common_width": width_dist.most_common(1)[0],
        "most_common_height": height_dist.most_common(1)[0],
        "resolution_dist": resolution_dist,
        "width_dist": width_dist,
        "height_dist": height_dist,
        "areas": areas,
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios
    }


def format_resolution(res):
    """Format độ phân giải để in ra dạng đẹp hơn"""
    return f"{res[0]}×{res[1]}"


def print_sr_results(results):
    """In kết quả phân tích"""
    print("\n" + "=" * 60)
    print("KẾT QUẢ PHÂN TÍCH ĐỘ PHÂN GIẢI SUPER RESOLUTION DATASET")
    print("=" * 60)

    # Tính tổng số lượng ảnh
    total_hr = sum(split_results.get("HR", {}).get("total_images", 0) for split_results in results.values())
    total_lr = sum(split_results.get("LR", {}).get("total_images", 0) for split_results in results.values())

    print(f"\nTổng số ảnh: {total_hr + total_lr}")
    print(f"- High Resolution (HR): {total_hr} ảnh")
    print(f"- Low Resolution (LR): {total_lr} ảnh")

    # Thông tin từng tập dữ liệu
    for dataset_split, split_results in results.items():
        print(f"\n{'-' * 60}")
        print(f"PHÂN TÍCH TẬP {dataset_split.upper()}")
        print(f"{'-' * 60}")

        hr_count = split_results.get("HR", {}).get("total_images", 0)
        lr_count = split_results.get("LR", {}).get("total_images", 0)

        print(f"- Số ảnh HR: {hr_count}")
        print(f"- Số ảnh LR: {lr_count}")

        # Tỉ lệ số lượng HR/LR
        if hr_count > 0 and lr_count > 0:
            ratio = lr_count / hr_count
            print(f"- Tỉ lệ số lượng LR/HR: {ratio:.2f}")

        # Phân tích HR và LR
        for res_type, stats in split_results.items():
            print(f"\n{'-' * 50}")
            print(f"PHÂN TÍCH {dataset_split} - {res_type} ({stats['total_images']} ảnh)")
            print(f"{'-' * 50}")

            # Thông tin độ phân giải
            min_res = stats['min_resolution']
            max_res = stats['max_resolution']
            most_common_res = stats['most_common_resolution']

            print(f"\nĐộ phân giải:")
            print(f"- Nhỏ nhất: {format_resolution(min_res)} ({min_res[0] * min_res[1]:,} pixels)")
            print(f"- Lớn nhất: {format_resolution(max_res)} ({max_res[0] * max_res[1]:,} pixels)")
            print(f"- Phổ biến nhất: {format_resolution(most_common_res[0])} ({most_common_res[1]} ảnh)")

            # Thông tin chiều rộng
            print(f"\nChiều rộng:")
            print(f"- Nhỏ nhất: {stats['min_width']:,} pixels")
            print(f"- Lớn nhất: {stats['max_width']:,} pixels")
            print(f"- Phổ biến nhất: {stats['most_common_width'][0]:,} pixels ({stats['most_common_width'][1]} ảnh)")

            # Thông tin chiều cao
            print(f"\nChiều cao:")
            print(f"- Nhỏ nhất: {stats['min_height']:,} pixels")
            print(f"- Lớn nhất: {stats['max_height']:,} pixels")
            print(f"- Phổ biến nhất: {stats['most_common_height'][0]:,} pixels ({stats['most_common_height'][1]} ảnh)")

            # Top 5 độ phân giải phổ biến
            print(f"\nTop 5 độ phân giải phổ biến:")
            for i, (res, count) in enumerate(stats['resolution_dist'].most_common(5), 1):
                print(f"{i}. {format_resolution(res):15} - {count:4} ảnh ({res[0] * res[1]:,} pixels)")

        # Nếu có cả HR và LR, tính tỉ lệ giữa HR và LR
        if "HR" in split_results and "LR" in split_results:
            hr_stats = split_results["HR"]
            lr_stats = split_results["LR"]

            hr_most_common = hr_stats["most_common_resolution"][0]
            lr_most_common = lr_stats["most_common_resolution"][0]

            width_ratio = hr_most_common[0] / lr_most_common[0]
            height_ratio = hr_most_common[1] / lr_most_common[1]
            area_ratio = (hr_most_common[0] * hr_most_common[1]) / (lr_most_common[0] * lr_most_common[1])

            print(f"\n{'-' * 50}")
            print(f"TỈ LỆ GIỮA HR VÀ LR TRONG TẬP {dataset_split}")
            print(f"{'-' * 50}")
            print(f"- Độ phân giải phổ biến nhất HR: {format_resolution(hr_most_common)}")
            print(f"- Độ phân giải phổ biến nhất LR: {format_resolution(lr_most_common)}")
            print(f"- Tỉ lệ chiều rộng: {width_ratio:.2f}x")
            print(f"- Tỉ lệ chiều cao: {height_ratio:.2f}x")
            print(f"- Tỉ lệ diện tích: {area_ratio:.2f}x")

            # Tính thêm tỉ lệ trung bình các diện tích
            hr_avg_area = sum(hr_stats["areas"]) / len(hr_stats["areas"])
            lr_avg_area = sum(lr_stats["areas"]) / len(lr_stats["areas"])
            avg_area_ratio = hr_avg_area / lr_avg_area

            print(f"- Diện tích trung bình HR: {hr_avg_area:,.2f} pixels")
            print(f"- Diện tích trung bình LR: {lr_avg_area:,.2f} pixels")
            print(f"- Tỉ lệ diện tích trung bình: {avg_area_ratio:.2f}x")


def plot_sr_charts(results, output_dir="sr_dataset_charts"):
    """Vẽ biểu đồ phân tích cho bộ dữ liệu"""
    os.makedirs(output_dir, exist_ok=True)

    # Vẽ biểu đồ cho từng tập dữ liệu
    for dataset_split, split_results in results.items():
        split_dir = os.path.join(output_dir, dataset_split)
        os.makedirs(split_dir, exist_ok=True)

        # 1. Biểu đồ so sánh phân bố diện tích giữa HR và LR trong cùng tập
        if "HR" in split_results and "LR" in split_results:
            plt.figure(figsize=(14, 8))

            hr_areas = split_results["HR"]["areas"]
            lr_areas = split_results["LR"]["areas"]

            # Log scale cho vùng giá trị rộng
            hr_log_areas = np.log10(hr_areas)
            lr_log_areas = np.log10(lr_areas)

            plt.hist([hr_log_areas, lr_log_areas], bins=50,
                     alpha=0.7, label=['High Resolution', 'Low Resolution'],
                     color=['royalblue', 'salmon'])

            plt.title(f'Pixel area distribution - {dataset_split} (log scale)', fontsize=16)
            plt.xlabel('Log10(Pixel area)', fontsize=14)
            plt.ylabel('Number of images', fontsize=14)

            # Tạo tick labels thực tế thay vì log
            ticks = plt.xticks()[0]
            plt.xticks(ticks, [f"{10 ** x:,.0f}" for x in ticks], rotation=45)

            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(split_dir, 'area_comparison_hist.png'), dpi=300)
            plt.close()

        # 2. Vẽ biểu đồ cho HR và LR
        for res_type, stats in split_results.items():
            # 2.1. Histogram phân bố diện tích
            plt.figure(figsize=(14, 8))
            areas = stats["areas"]
            log_areas = np.log10(areas)

            plt.hist(log_areas, bins=50, color='skyblue', edgecolor='black')
            plt.title(f'Pixel area distribution - {dataset_split} {res_type} (log scale)', fontsize=16)
            plt.xlabel('Log10(Pixel Area)', fontsize=14)
            plt.ylabel('Number of images', fontsize=14)

            # Tạo tick labels thực tế
            ticks = plt.xticks()[0]
            plt.xticks(ticks, [f"{10 ** x:,.0f}" for x in ticks], rotation=45)

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(split_dir, f'{res_type}_area_hist.png'), dpi=300)
            plt.close()

            # 2.2. Scatter plot chiều rộng vs chiều cao
            plt.figure(figsize=(12, 12))
            plt.scatter(stats["widths"], stats["heights"], alpha=0.3, s=10, color='blue')
            plt.title(f'Width and Height - {dataset_split} {res_type}', fontsize=16)
            plt.xlabel('Width (pixels)', fontsize=14)
            plt.ylabel('Height (pixels)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(split_dir, f'{res_type}_width_height_scatter.png'), dpi=300)
            plt.close()

            # 2.3. Histogram tỉ lệ khung hình (aspect ratio)
            plt.figure(figsize=(14, 8))
            valid_ratios = [r for r in stats["aspect_ratios"] if 0.1 < r < 10]
            plt.hist(valid_ratios, bins=50, color='lightgreen', edgecolor='black')
            plt.title(f'Aspect Ratio Distribution - {dataset_split} {res_type}', fontsize=16)
            plt.xlabel('Aspect ratio (width/height)', fontsize=14)
            plt.ylabel('Number of images', fontsize=14)

            # Đánh dấu các tỉ lệ khung hình phổ biến
            common_ratios = {
                '1:1': 1.0,
                '4:3': 4 / 3,
                '16:9': 16 / 9,
                '3:2': 3 / 2,
                '2:3': 2 / 3,
                '9:16': 9 / 16,
                '3:4': 3 / 4
            }

            y_max = plt.gca().get_ylim()[1]
            for name, ratio in common_ratios.items():
                if min(valid_ratios) <= ratio <= max(valid_ratios):
                    plt.axvline(x=ratio, color='red', linestyle='--', alpha=0.7)
                    plt.text(ratio, y_max * 0.9, name, rotation=90, color='red',
                             ha='right', va='top', backgroundcolor='white', alpha=0.7)

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(split_dir, f'{res_type}_aspect_ratio_hist.png'), dpi=300)
            plt.close()

            # 2.4. Top độ phân giải phổ biến
            plt.figure(figsize=(16, 10))
            top_n = min(20, len(stats['resolution_dist']))
            top_resolutions = stats['resolution_dist'].most_common(top_n)
            resolution_labels = [f"{w}×{h}" for (w, h), _ in top_resolutions]
            counts = [count for _, count in top_resolutions]

            plt.bar(range(len(counts)), counts, color='lightblue', edgecolor='black')
            plt.xticks(range(len(counts)), resolution_labels, rotation=90, ha='center', fontsize=10)
            plt.title(f'Top {top_n} common resolution - {dataset_split} {res_type}', fontsize=16)
            plt.xlabel('Resolution', fontsize=14)
            plt.ylabel('NUmber of images', fontsize=14)
            plt.grid(axis='y', alpha=0.3)

            # Thêm giá trị số lượng trên mỗi cột
            for i, count in enumerate(counts):
                plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(split_dir, f'{res_type}_top_resolutions.png'), dpi=300)
            plt.close()

    # 3. Biểu đồ tổng hợp so sánh giữa các tập Train/Val/Test
    # 3.1. So sánh tỉ lệ phóng to giữa các tập
    plt.figure(figsize=(12, 8))

    scale_factors = []
    labels = []

    for dataset_split, split_results in results.items():
        if "HR" in split_results and "LR" in split_results:
            hr_most_common = split_results["HR"]["most_common_resolution"][0]
            lr_most_common = split_results["LR"]["most_common_resolution"][0]

            # Tính tỉ lệ phóng to (chiều rộng)
            scale_factor = hr_most_common[0] / lr_most_common[0]
            scale_factors.append(scale_factor)
            labels.append(dataset_split)

    if scale_factors:
        plt.bar(labels, scale_factors, color='skyblue')
        plt.title('Compare zoom ratios between datasets', fontsize=16)
        plt.ylabel('Enlargement ratio (width)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)

        # Thêm giá trị trên từng cột
        for i, value in enumerate(scale_factors):
            plt.text(i, value + 0.1, f"{value:.2f}x", ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scale_factor_comparison.png'), dpi=300)
        plt.close()


def generate_sr_summary(results, output_file="sr_dataset_analysis_summary.txt"):
    """Tạo báo cáo tổng hợp cho bộ dữ liệu Super Resolution"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO PHÂN TÍCH ĐỘ PHÂN GIẢI ẢNH SUPER RESOLUTION DATASET\n")
        f.write("=" * 70 + "\n\n")

        # Tổng hợp số lượng ảnh
        total_hr = sum(split_results.get("HR", {}).get("total_images", 0) for split_results in results.values())
        total_lr = sum(split_results.get("LR", {}).get("total_images", 0) for split_results in results.values())

        f.write(f"TỔNG SỐ ẢNH: {total_hr + total_lr}\n")
        f.write(f"- High Resolution (HR): {total_hr} ảnh\n")
        f.write(f"- Low Resolution (LR): {total_lr} ảnh\n\n")

        # Thông tin từng tập dữ liệu
        for dataset_split, split_results in results.items():
            f.write(f"\n{'=' * 60}\n")
            f.write(f"PHÂN TÍCH TẬP {dataset_split.upper()}\n")
            f.write(f"{'=' * 60}\n")

            hr_count = split_results.get("HR", {}).get("total_images", 0)
            lr_count = split_results.get("LR", {}).get("total_images", 0)

            f.write(f"- Số ảnh HR: {hr_count}\n")
            f.write(f"- Số ảnh LR: {lr_count}\n")

            # Tỉ lệ HR/LR
            if hr_count > 0 and lr_count > 0:
                ratio = lr_count / hr_count
                f.write(f"- Tỉ lệ số lượng LR/HR: {ratio:.2f}\n")

            # Chi tiết từng loại
            for res_type, stats in split_results.items():
                f.write(f"\n{'-' * 50}\n")
                f.write(f"{dataset_split} - {res_type} ({stats['total_images']} ảnh)\n")
                f.write(f"{'-' * 50}\n")

                # Thông tin độ phân giải
                min_res = stats['min_resolution']
                max_res = stats['max_resolution']
                most_common_res = stats['most_common_resolution']

                f.write("\nĐộ phân giải:\n")
                f.write(f"- Nhỏ nhất: {format_resolution(min_res)} ({min_res[0] * min_res[1]:,} pixels)\n")
                f.write(f"- Lớn nhất: {format_resolution(max_res)} ({max_res[0] * max_res[1]:,} pixels)\n")
                f.write(f"- Phổ biến nhất: {format_resolution(most_common_res[0])} ({most_common_res[1]} ảnh)\n")

                # Phân phối kích thước
                avg_width = sum(stats["widths"]) / len(stats["widths"])
                avg_height = sum(stats["heights"]) / len(stats["heights"])
                avg_area = sum(stats["areas"]) / len(stats["areas"])

                f.write(f"\nKích thước trung bình:\n")
                f.write(f"- Chiều rộng trung bình: {avg_width:,.2f} pixels\n")
                f.write(f"- Chiều cao trung bình: {avg_height:,.2f} pixels\n")
                f.write(f"- Diện tích trung bình: {avg_area:,.2f} pixels\n")

                # Top 10 độ phân giải phổ biến
                f.write("\nTop 10 độ phân giải phổ biến:\n")
                for i, (res, count) in enumerate(stats['resolution_dist'].most_common(10), 1):
                    percentage = count / stats['total_images'] * 100
                    f.write(f"{i}. {format_resolution(res):15} - {count:4} ảnh ({percentage:.2f}%)\n")

                # Phân tích tỉ lệ khung hình
                valid_ratios = [r for r in stats["aspect_ratios"] if 0.1 < r < 10]
                ratio_counter = Counter([round(r * 100) / 100 for r in valid_ratios])
                most_common_ratio = ratio_counter.most_common(1)[0] if ratio_counter else (0, 0)

                f.write("\nTỉ lệ khung hình:\n")
                if most_common_ratio[0] > 0:
                    f.write(f"- Phổ biến nhất: {most_common_ratio[0]:.2f} ({most_common_ratio[1]} ảnh)\n")

                # Kiểm tra tỉ lệ khung hình phổ biến
                common_ratios = {
                    '1:1': 1.0,
                    '4:3': 4 / 3,
                    '16:9': 16 / 9,
                    '3:2': 3 / 2,
                    '9:16': 9 / 16,
                    '2:3': 2 / 3
                }

                f.write("- Tỉ lệ khung hình phổ biến:\n")
                for name, ratio in common_ratios.items():
                    # Đếm số ảnh có tỉ lệ xấp xỉ với tỉ lệ phổ biến (độ lệch 5%)
                    count = sum(1 for r in valid_ratios if abs(r - ratio) / ratio < 0.05)
                    percentage = count / len(valid_ratios) * 100 if valid_ratios else 0
                    f.write(f"  * {name}: {count} ảnh ({percentage:.2f}%)\n")

            # So sánh giữa HR và LR trong cùng một tập
            if "HR" in split_results and "LR" in split_results:
                hr_stats = split_results["HR"]
                lr_stats = split_results["LR"]

                f.write(f"\n{'-' * 50}\n")
                f.write(f"SO SÁNH GIỮA HR VÀ LR TRONG TẬP {dataset_split}\n")
                f.write(f"{'-' * 50}\n")

                # Tỉ lệ diện tích trung bình
                hr_avg_area = sum(hr_stats["areas"]) / len(hr_stats["areas"])
                lr_avg_area = sum(lr_stats["areas"]) / len(lr_stats["areas"])
                area_ratio = hr_avg_area / lr_avg_area

                f.write(f"\nDiện tích trung bình:\n")
                f.write(f"- HR: {hr_avg_area:,.2f} pixels\n")
                f.write(f"- LR: {lr_avg_area:,.2f} pixels\n")
                f.write(f"- Tỉ lệ: {area_ratio:.2f}x\n")

                # Tỉ lệ các chiều
                hr_avg_width = sum(hr_stats["widths"]) / len(hr_stats["widths"])
                lr_avg_width = sum(lr_stats["widths"]) / len(lr_stats["widths"])
                width_ratio = hr_avg_width / lr_avg_width

                hr_avg_height = sum(hr_stats["heights"]) / len(hr_stats["heights"])
                lr_avg_height = sum(lr_stats["heights"]) / len(lr_stats["heights"])
                height_ratio = hr_avg_height / lr_avg_height

                f.write(f"\nChiều rộng trung bình:\n")
                f.write(f"- HR: {hr_avg_width:,.2f} pixels\n")
                f.write(f"- LR: {lr_avg_width:,.2f} pixels\n")
                f.write(f"- Tỉ lệ: {width_ratio:.2f}x\n")

                f.write(f"\nChiều cao trung bình:\n")
                f.write(f"- HR: {hr_avg_height:,.2f} pixels\n")
                f.write(f"- LR: {lr_avg_height:,.2f} pixels\n")
                f.write(f"- Tỉ lệ: {height_ratio:.2f}x\n")

                # Tỉ lệ dựa trên độ phân giải phổ biến nhất
                hr_common = hr_stats["most_common_resolution"][0]
                lr_common = lr_stats["most_common_resolution"][0]

                common_width_ratio = hr_common[0] / lr_common[0]
                common_height_ratio = hr_common[1] / lr_common[1]
                common_area_ratio = (hr_common[0] * hr_common[1]) / (lr_common[0] * lr_common[1])

                f.write(f"\nTỉ lệ dựa trên độ phân giải phổ biến nhất:\n")
                f.write(f"- HR phổ biến nhất: {format_resolution(hr_common)}\n")
                f.write(f"- LR phổ biến nhất: {format_resolution(lr_common)}\n")
                f.write(f"- Tỉ lệ chiều rộng: {common_width_ratio:.2f}x\n")
                f.write(f"- Tỉ lệ chiều cao: {common_height_ratio:.2f}x\n")
                f.write(f"- Tỉ lệ diện tích: {common_area_ratio:.2f}x\n")

        # Bảng so sánh tỉ lệ phóng to giữa các tập dữ liệu
        f.write(f"\n{'-' * 70}\n")
        f.write(f"SO SÁNH TỈ LỆ PHÓNG TO GIỮA CÁC TẬP DỮ LIỆU\n")
        f.write(f"{'-' * 70}\n\n")

        f.write("| Tập dữ liệu | Tỉ lệ chiều rộng | Tỉ lệ chiều cao | Tỉ lệ diện tích |\n")
        f.write("|-------------|-----------------|----------------|----------------|\n")

        for dataset_split, split_results in results.items():
            if "HR" in split_results and "LR" in split_results:
                hr_common = split_results["HR"]["most_common_resolution"][0]
                lr_common = split_results["LR"]["most_common_resolution"][0]

                width_ratio = hr_common[0] / lr_common[0]
                height_ratio = hr_common[1] / lr_common[1]
                area_ratio = (hr_common[0] * hr_common[1]) / (lr_common[0] * lr_common[1])

                f.write(f"| {dataset_split} | {width_ratio:.2f}x | {height_ratio:.2f}x | {area_ratio:.2f}x |\n")

        f.write("\n")


# Hàm main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Phân tích độ phân giải ảnh trong bộ dữ liệu Super Resolution')
    parser.add_argument('root_dir', type=str, help='Đường dẫn tới thư mục gốc của bộ dữ liệu')
    parser.add_argument('--output_dir', type=str, default='sr_dataset_charts',
                        help='Thư mục lưu các biểu đồ phân tích (mặc định: sr_dataset_charts)')
    parser.add_argument('--output_file', type=str, default='sr_dataset_analysis_summary.txt',
                        help='Tên file lưu báo cáo tổng hợp (mặc định: sr_dataset_analysis_summary.txt)')

    args = parser.parse_args()

    print(f"Bắt đầu phân tích bộ dữ liệu tại {args.root_dir}...")
    results = analyze_sr_dataset(args.root_dir)

    if not results:
        print("Không tìm thấy dữ liệu để phân tích!")
        exit(1)

    # In kết quả ra màn hình
    print_sr_results(results)

    # Tạo biểu đồ phân tích
    plot_sr_charts(results, args.output_dir)

    # Tạo báo cáo tổng hợp
    generate_sr_summary(results, args.output_file)

    print(f"\nĐã hoàn thành phân tích. Các biểu đồ được lưu trong thư mục '{args.output_dir}'")
    print(f"Báo cáo tổng hợp được lưu trong file '{args.output_file}'")