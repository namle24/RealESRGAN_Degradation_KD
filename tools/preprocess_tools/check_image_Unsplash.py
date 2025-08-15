import os
import glob
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def analyze_unsplash_resolutions(root_dir):
    """Phân tích độ phân giải của ảnh trong bộ dữ liệu Unsplash"""
    results = {}

    # Phân tích thư mục high res
    high_res_dir = os.path.join(root_dir, "high res")
    if os.path.exists(high_res_dir):
        print(f"Đang phân tích thư mục high res...")
        high_res_stats = analyze_directory(high_res_dir)
        if high_res_stats:
            results["high_res"] = high_res_stats

    # Phân tích thư mục low res
    low_res_dir = os.path.join(root_dir, "low res")
    if os.path.exists(low_res_dir):
        print(f"Đang phân tích thư mục low res...")
        low_res_stats = analyze_directory(low_res_dir)
        if low_res_stats:
            results["low_res"] = low_res_stats

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


def print_unsplash_results(results):
    """In kết quả phân tích"""
    print("\n" + "=" * 60)
    print("KẾT QUẢ PHÂN TÍCH ĐỘ PHÂN GIẢI UNSPLASH")
    print("=" * 60)

    # In tổng số lượng ảnh
    high_res_count = results.get("high_res", {}).get("total_images", 0)
    low_res_count = results.get("low_res", {}).get("total_images", 0)
    print(f"\nTổng số ảnh: {high_res_count + low_res_count}")
    print(f"- High resolution: {high_res_count} ảnh")
    print(f"- Low resolution: {low_res_count} ảnh")

    # Tỉ lệ số lượng high/low
    if high_res_count > 0 and low_res_count > 0:
        ratio = low_res_count / high_res_count
        print(f"- Tỉ lệ số lượng Low/High: {ratio:.2f}")

    # Phân tích chi tiết từng loại
    for res_type, stats in results.items():
        print(f"\n{'-' * 50}")
        print(f"PHÂN TÍCH {res_type.upper()} ({stats['total_images']} ảnh)")
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

    # Nếu có cả hai tập dữ liệu, tính toán tỉ lệ trung bình giữa high và low
    if "high_res" in results and "low_res" in results:
        high_most_common = results["high_res"]["most_common_resolution"][0]
        low_most_common = results["low_res"]["most_common_resolution"][0]

        width_ratio = high_most_common[0] / low_most_common[0]
        height_ratio = high_most_common[1] / low_most_common[1]
        area_ratio = (high_most_common[0] * high_most_common[1]) / (low_most_common[0] * low_most_common[1])

        print(f"\n{'-' * 50}")
        print("TỈ LỆ GIỮA HIGH RES VÀ LOW RES")
        print(f"{'-' * 50}")
        print(f"- Độ phân giải phổ biến nhất High: {format_resolution(high_most_common)}")
        print(f"- Độ phân giải phổ biến nhất Low: {format_resolution(low_most_common)}")
        print(f"- Tỉ lệ chiều rộng: {width_ratio:.2f}x")
        print(f"- Tỉ lệ chiều cao: {height_ratio:.2f}x")
        print(f"- Tỉ lệ diện tích: {area_ratio:.2f}x")

        # Tính thêm tỉ lệ trung bình các diện tích
        high_avg_area = sum(results["high_res"]["areas"]) / len(results["high_res"]["areas"])
        low_avg_area = sum(results["low_res"]["areas"]) / len(results["low_res"]["areas"])
        avg_area_ratio = high_avg_area / low_avg_area

        print(f"- Diện tích trung bình High: {high_avg_area:,.2f} pixels")
        print(f"- Diện tích trung bình Low: {low_avg_area:,.2f} pixels")
        print(f"- Tỉ lệ diện tích trung bình: {avg_area_ratio:.2f}x")


def plot_unsplash_charts(results, output_dir="unsplash_charts"):
    """Vẽ biểu đồ phân tích cho bộ dữ liệu Unsplash"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Biểu đồ so sánh phân bố diện tích
    if "high_res" in results and "low_res" in results:
        plt.figure(figsize=(14, 8))

        high_areas = results["high_res"]["areas"]
        low_areas = results["low_res"]["areas"]

        # Log scale cho vùng giá trị rộng
        high_log_areas = np.log10(high_areas)
        low_log_areas = np.log10(low_areas)

        plt.hist([high_log_areas, low_log_areas], bins=50,
                 alpha=0.7, label=['High Resolution', 'Low Resolution'],
                 color=['royalblue', 'salmon'])

        plt.title('Phân bố diện tích pixel (log scale)', fontsize=16)
        plt.xlabel('Log10(Diện tích pixel)', fontsize=14)
        plt.ylabel('Số lượng ảnh', fontsize=14)

        # Tạo tick labels thực tế thay vì log
        ticks = plt.xticks()[0]
        plt.xticks(ticks, [f"{10 ** x:,.0f}" for x in ticks], rotation=45)

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'area_comparison_hist.png'), dpi=300)
        plt.close()

    # 2. Vẽ biểu đồ cho từng loại ảnh
    for res_type, stats in results.items():
        # 2.1. Histogram phân bố diện tích
        plt.figure(figsize=(14, 8))
        areas = stats["areas"]
        log_areas = np.log10(areas)

        plt.hist(log_areas, bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Phân bố diện tích pixel - {res_type} (log scale)', fontsize=16)
        plt.xlabel('Log10(Diện tích pixel)', fontsize=14)
        plt.ylabel('Số lượng ảnh', fontsize=14)

        # Tạo tick labels thực tế
        ticks = plt.xticks()[0]
        plt.xticks(ticks, [f"{10 ** x:,.0f}" for x in ticks], rotation=45)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{res_type}_area_hist.png'), dpi=300)
        plt.close()

        # 2.2. Scatter plot chiều rộng vs chiều cao
        plt.figure(figsize=(12, 12))

        # Sử dụng alpha thấp để thấy mật độ tốt hơn khi có nhiều điểm chồng lên nhau
        plt.scatter(stats["widths"], stats["heights"], alpha=0.3, s=10, color='blue')
        plt.title(f'Chiều rộng vs Chiều cao - {res_type}', fontsize=16)
        plt.xlabel('Chiều rộng (pixels)', fontsize=14)
        plt.ylabel('Chiều cao (pixels)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{res_type}_width_height_scatter.png'), dpi=300)
        plt.close()

        # 2.3. Histogram tỉ lệ khung hình (aspect ratio)
        plt.figure(figsize=(14, 8))

        # Lọc các aspect ratio hợp lệ (loại bỏ các giá trị bất thường)
        valid_ratios = [r for r in stats["aspect_ratios"] if 0.1 < r < 10]

        plt.hist(valid_ratios, bins=50, color='lightgreen', edgecolor='black')
        plt.title(f'Phân bố tỉ lệ khung hình (chiều rộng/chiều cao) - {res_type}', fontsize=16)
        plt.xlabel('Tỉ lệ khung hình', fontsize=14)
        plt.ylabel('Số lượng ảnh', fontsize=14)

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
        plt.savefig(os.path.join(output_dir, f'{res_type}_aspect_ratio_hist.png'), dpi=300)
        plt.close()

        # 2.4. Top độ phân giải phổ biến
        plt.figure(figsize=(16, 10))

        top_n = min(20, len(stats['resolution_dist']))
        top_resolutions = stats['resolution_dist'].most_common(top_n)

        resolution_labels = [f"{w}×{h}" for (w, h), _ in top_resolutions]
        counts = [count for _, count in top_resolutions]

        plt.bar(range(len(counts)), counts, color='lightblue', edgecolor='black')
        plt.xticks(range(len(counts)), resolution_labels, rotation=90, ha='center', fontsize=10)
        plt.title(f'Top {top_n} độ phân giải phổ biến - {res_type}', fontsize=16)
        plt.xlabel('Độ phân giải', fontsize=14)
        plt.ylabel('Số lượng ảnh', fontsize=14)
        plt.grid(axis='y', alpha=0.3)

        # Thêm giá trị số lượng trên mỗi cột
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{res_type}_top_resolutions.png'), dpi=300)
        plt.close()

    # 3. Biểu đồ so sánh tỉ lệ khung hình
    if "high_res" in results and "low_res" in results:
        plt.figure(figsize=(14, 8))

        high_ratios = [r for r in results["high_res"]["aspect_ratios"] if 0.1 < r < 10]
        low_ratios = [r for r in results["low_res"]["aspect_ratios"] if 0.1 < r < 10]

        plt.hist([high_ratios, low_ratios], bins=50,
                 alpha=0.7, label=['High Resolution', 'Low Resolution'],
                 color=['royalblue', 'salmon'])

        plt.title('So sánh phân bố tỉ lệ khung hình', fontsize=16)
        plt.xlabel('Tỉ lệ khung hình (chiều rộng/chiều cao)', fontsize=14)
        plt.ylabel('Số lượng ảnh', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aspect_ratio_comparison.png'), dpi=300)
        plt.close()


def generate_unsplash_summary(results, output_file="unsplash_analysis_summary.txt"):
    """Tạo báo cáo tổng hợp cho Unsplash"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO PHÂN TÍCH ĐỘ PHÂN GIẢI ẢNH UNSPLASH\n")
        f.write("=" * 60 + "\n\n")

        # Tổng hợp số lượng ảnh
        high_res_count = results.get("high_res", {}).get("total_images", 0)
        low_res_count = results.get("low_res", {}).get("total_images", 0)

        f.write(f"TỔNG SỐ ẢNH: {high_res_count + low_res_count}\n")
        f.write(f"- High resolution: {high_res_count} ảnh\n")
        f.write(f"- Low resolution: {low_res_count} ảnh\n")

        if high_res_count > 0 and low_res_count > 0:
            ratio = low_res_count / high_res_count
            f.write(f"- Tỉ lệ số lượng Low/High: {ratio:.2f}\n\n")

        # Chi tiết từng loại
        for res_type, stats in results.items():
            f.write(f"\n{res_type.upper()} ({stats['total_images']} ảnh)\n")
            f.write("-" * 50 + "\n")

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
            most_common_ratio = ratio_counter.most_common(1)[0]

            f.write("\nTỉ lệ khung hình:\n")
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

        # So sánh giữa high res và low res
        if "high_res" in results and "low_res" in results:
            f.write("\n" + "=" * 50 + "\n")
            f.write("SO SÁNH GIỮA HIGH RES VÀ LOW RES\n")
            f.write("=" * 50 + "\n")

            high_res = results["high_res"]
            low_res = results["low_res"]

            # Tỉ lệ diện tích trung bình
            high_avg_area = sum(high_res["areas"]) / len(high_res["areas"])
            low_avg_area = sum(low_res["areas"]) / len(low_res["areas"])
            area_ratio = high_avg_area / low_avg_area

            f.write(f"\nDiện tích trung bình:\n")
            f.write(f"- High Res: {high_avg_area:,.2f} pixels\n")
            f.write(f"- Low Res: {low_avg_area:,.2f} pixels\n")
            f.write(f"- Tỉ lệ: {area_ratio:.2f}x\n")

            # Tỉ lệ các chiều
            high_avg_width = sum(high_res["widths"]) / len(high_res["widths"])
            low_avg_width = sum(low_res["widths"]) / len(low_res["widths"])
            width_ratio = high_avg_width / low_avg_width

            high_avg_height = sum(high_res["heights"]) / len(high_res["heights"])
            low_avg_height = sum(low_res["heights"]) / len(low_res["heights"])
            height_ratio = high_avg_height / low_avg_height

            f.write(f"\nChiều rộng trung bình:\n")
            f.write(f"- High Res: {high_avg_width:,.2f} pixels\n")
            f.write(f"- Low Res: {low_avg_width:,.2f} pixels\n")
            f.write(f"- Tỉ lệ: {width_ratio:.2f}x\n")

            f.write(f"\nChiều cao trung bình:\n")
            f.write(f"- High Res: {high_avg_height:,.2f} pixels\n")
            f.write(f"- Low Res: {low_avg_height:,.2f} pixels\n")
            f.write(f"- Tỉ lệ: {height_ratio:.2f}x\n")

            # Tỉ lệ dựa trên độ phân giải phổ biến nhất
            high_common = high_res["most_common_resolution"][0]
            low_common = low_res["most_common_resolution"][0]

            common_width_ratio = high_common[0] / low_common[0]
            common_height_ratio = high_common[1] / low_common[1]
            common_area_ratio = (high_common[0] * high_common[1]) / (low_common[0] * low_common[1])

            f.write(f"\nTỉ lệ dựa trên độ phân giải phổ biến nhất:\n")
            f.write(f"- High Res phổ biến nhất: {format_resolution(high_common)}\n")
            f.write(f"- Low Res phổ biến nhất: {format_resolution(low_common)}\n")
            f.write(f"- Tỉ lệ chiều rộng: {common_width_ratio:.2f}x\n")
            f.write(f"- Tỉ lệ chiều cao: {common_height_ratio:.2f}x\n")
            f.write(f"- Tỉ lệ diện tích: {common_area_ratio:.2f}x\n")


# Sử dụng hàm
if __name__ == "__main__":
    root_dir = "D:\\EnhanceVideo_ImageDLM\\data\\Image Super Resolution - Unsplash"  # Thay đổi đường dẫn này tới thư mục gốc Unsplash
    results = analyze_unsplash_resolutions(root_dir)

    # In kết quả ra màn hình
    print_unsplash_results(results)

    # Tạo biểu đồ phân tích
    plot_unsplash_charts(results)

    # Tạo báo cáo tổng hợp
    generate_unsplash_summary(results)

    print("\nĐã hoàn thành phân tích. Các biểu đồ được lưu trong thư mục 'unsplash_charts'")
    print("Báo cáo tổng hợp được lưu trong file 'unsplash_analysis_summary.txt'")