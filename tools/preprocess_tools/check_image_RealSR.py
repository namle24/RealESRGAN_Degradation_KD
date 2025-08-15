import os
from PIL import Image
from collections import Counter
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # Để định dạng số lớn trên trục


# --- Giữ nguyên hàm analyze_dataset_resolutions từ trước ---
def analyze_dataset_resolutions(dataset_root):
    """
    Phân tích độ phân giải của ảnh trong bộ dữ liệu RealSR.
    (Giữ nguyên phần quét file và phân tích cơ bản)
    """
    if not os.path.isdir(dataset_root):
        print(f"Lỗi: Thư mục '{dataset_root}' không tồn tại.")
        return None

    all_images = []
    hr_images = []
    lr_images = {2: [], 3: [], 4: []}
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    lr_pattern = re.compile(r"_LR([234])\.(?:png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)
    hr_suffix = '_hr'  # Để dễ thay đổi nếu cần

    print(f"Đang quét thư mục: {dataset_root}...")
    for root, _, files in os.walk(dataset_root):
        for filename in files:
            # Tách tên file và phần mở rộng (không phân biệt hoa thường)
            base, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower in image_extensions:
                filepath = os.path.join(root, filename)
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        area = width * height
                        resolution_tuple = (width, height)  # Lưu tuple WxH

                        image_info = {
                            'path': filepath,
                            'width': width,
                            'height': height,
                            'resolution': resolution_tuple,  # Thêm tuple WxH
                            'area': area,
                            'type': 'unknown',
                            'scale': None
                        }

                        # Phân loại HR/LR dựa trên tên file
                        lr_match = lr_pattern.search(filename)
                        # Kiểm tra chính xác hơn cho HR
                        if base.lower().endswith(hr_suffix):
                            image_info['type'] = 'HR'
                            hr_images.append(image_info)
                            all_images.append(image_info)
                        elif lr_match:
                            scale = int(lr_match.group(1))
                            image_info['type'] = 'LR'
                            image_info['scale'] = scale
                            if scale in lr_images:
                                lr_images[scale].append(image_info)
                            all_images.append(image_info)
                        # else:
                        #     # print(f"Cảnh báo: Không thể xác định loại HR/LR cho file: {filepath}")
                        #      all_images.append(image_info) # Vẫn thêm vào all_images

                except Exception as e:
                    print(f"Lỗi khi xử lý file '{filepath}': {e}")

    print(f"Quét xong. Tìm thấy tổng cộng {len(all_images)} ảnh.")

    if not all_images:
        print("Không tìm thấy file ảnh nào trong thư mục.")
        return None

    # --- Phân tích chung cho tất cả ảnh ---
    print("\n--- PHÂN TÍCH TỔNG QUÁT ---")
    all_images.sort(key=lambda x: x['area'])
    min_res_img = all_images[0]
    max_res_img = all_images[-1]
    min_area = min_res_img['area']
    max_area = max_res_img['area']
    count_min_res = sum(1 for img in all_images if img['area'] == min_area)
    count_max_res = sum(1 for img in all_images if img['area'] == max_area)

    print(f"Độ phân giải nhỏ nhất (WxH): {min_res_img['width']}x{min_res_img['height']} (Diện tích: {min_area:,})")
    print(f"   -> Có {count_min_res} ảnh với độ phân giải này.")
    print(f"Độ phân giải lớn nhất (WxH):  {max_res_img['width']}x{max_res_img['height']} (Diện tích: {max_area:,})")
    print(f"   -> Có {count_max_res} ảnh với độ phân giải này.")

    # --- Biểu đồ 1: Độ phân giải (WxH) phổ biến nhất ---
    resolution_counts = Counter(img['resolution'] for img in all_images)
    most_common_resolutions = resolution_counts.most_common(15)  # Lấy top 15

    if most_common_resolutions:
        print(f"\nTop {len(most_common_resolutions)} độ phân giải (WxH) phổ biến nhất:")
        for (w, h), count in most_common_resolutions:
            print(f"  - {w}x{h}: {count} ảnh")

        # Chuẩn bị dữ liệu cho biểu đồ cột
        labels = [f"{w}x{h}" for (w, h), count in most_common_resolutions]
        counts = [count for (w, h), count in most_common_resolutions]

        plt.figure(figsize=(12, 7))  # Tăng kích thước để chứa nhãn
        bars = plt.bar(labels, counts, color='skyblue', edgecolor='black')
        plt.xlabel("Resolution (Width x Height)")
        plt.ylabel("Number of photos")
        plt.title(f"Top {len(most_common_resolutions)} most common resolution in Dataset")
        plt.xticks(rotation=45, ha='right')  # Xoay nhãn trục X cho dễ đọc
        plt.tight_layout()  # Điều chỉnh layout
        # Thêm số lượng trên mỗi cột
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), va='bottom',
                     ha='center')  # Đặt số ở trên cột
        plt.show()

    else:
        print("\nKhông đủ dữ liệu để hiển thị độ phân giải phổ biến.")

    # --- Biểu đồ 2: Phân bố diện tích (Histogram) ---
    areas = [img['area'] for img in all_images]
    num_bins = 15  # Có thể điều chỉnh số bins
    plt.figure(figsize=(10, 6))
    counts_hist, bin_edges, _ = plt.hist(areas, bins=num_bins, edgecolor='black', color='lightblue')
    plt.title('Image Area Distribution (General)')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Number of photos')
    # Định dạng trục x để dễ đọc số lớn
    formatter = mticker.FormatStrFormatter('%d')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    # Tìm khoảng phổ biến nhất từ histogram
    if len(counts_hist) > 0:
        max_count_bin_index = np.argmax(counts_hist)
        most_common_range_start = bin_edges[max_count_bin_index]
        most_common_range_end = bin_edges[max_count_bin_index + 1]
        most_common_count = int(counts_hist[max_count_bin_index])
        print(f"\nKhoảng diện tích có nhiều ảnh nhất (từ histogram):")
        print(f"   -> Từ {most_common_range_start:,.0f} đến {most_common_range_end:,.0f} pixels")
        print(f"   -> Có khoảng {most_common_count} ảnh trong khoảng này.")

    # --- Phân tích chi tiết HR ---
    print("\n--- PHÂN TÍCH ẢNH HR ---")
    hr_areas = []
    if hr_images:
        hr_images.sort(key=lambda x: x['area'])
        min_hr_img = hr_images[0]
        max_hr_img = hr_images[-1]
        min_hr_area = min_hr_img['area']
        max_hr_area = max_hr_img['area']
        count_min_hr = sum(1 for img in hr_images if img['area'] == min_hr_area)
        count_max_hr = sum(1 for img in hr_images if img['area'] == max_hr_area)
        hr_areas = [img['area'] for img in hr_images]  # Lấy diện tích HR

        print(f"Tổng số ảnh HR: {len(hr_images)}")
        print(
            f"Độ phân giải HR nhỏ nhất: {min_hr_img['width']}x{min_hr_img['height']} ({min_hr_area:,} px) - {count_min_hr} ảnh")
        print(
            f"Độ phân giải HR lớn nhất:  {max_hr_img['width']}x{max_hr_img['height']} ({max_hr_area:,} px) - {count_max_hr} ảnh")
    else:
        print("Không tìm thấy ảnh HR nào.")

    # --- Phân tích chi tiết LR (theo từng scale) ---
    print("\n--- PHÂN TÍCH ẢNH LR ---")
    lr_areas_by_scale = {2: [], 3: [], 4: []}
    total_lr_count = 0
    for scale, images in lr_images.items():
        if images:
            total_lr_count += len(images)
            images.sort(key=lambda x: x['area'])
            min_lr_img = images[0]
            max_lr_img = images[-1]
            min_lr_area = min_lr_img['area']
            max_lr_area = max_lr_img['area']
            count_min_lr = sum(1 for img in images if img['area'] == min_lr_area)
            count_max_lr = sum(1 for img in images if img['area'] == max_lr_area)
            lr_areas_by_scale[scale] = [img['area'] for img in images]  # Lấy diện tích LR theo scale

            print(f"\n  -- Scale x{scale} --")
            print(f"  Số lượng ảnh LR x{scale}: {len(images)}")
            print(
                f"  Độ phân giải LR x{scale} nhỏ nhất: {min_lr_img['width']}x{min_lr_img['height']} ({min_lr_area:,} px) - {count_min_lr} ảnh")
            print(
                f"  Độ phân giải LR x{scale} lớn nhất:  {max_lr_img['width']}x{max_lr_img['height']} ({max_lr_area:,} px) - {count_max_lr} ảnh")
        else:
            print(f"\n  -- Scale x{scale} --")
            print(f"  Không tìm thấy ảnh LR x{scale} nào.")

    if total_lr_count == 0:
        print("Không tìm thấy ảnh LR nào.")

    # --- Biểu đồ 3: So sánh phân bố diện tích HR vs LR (Histogram chồng lớp) ---
    plt.figure(figsize=(12, 7))
    plot_data_lr = {scale: areas for scale, areas in lr_areas_by_scale.items() if areas}  # Chỉ lấy scale có data

    # Xác định bins chung dựa trên tất cả diện tích để so sánh công bằng
    combined_areas = hr_areas + [area for areas in plot_data_lr.values() for area in areas]
    if combined_areas:
        # Xác định số bins phù hợp, ví dụ dựa trên quy tắc Freedman-Diaconis hoặc Sturges
        # Hoặc đơn giản là chọn một số cố định hợp lý
        iqr = np.subtract(*np.percentile(combined_areas, [75, 25]))
        bin_width = 2 * iqr * (len(combined_areas) ** (-1 / 3)) if iqr > 0 else 50000  # Ước lượng độ rộng bin
        num_bins_combined = int(np.ceil(
            (max(combined_areas) - min(combined_areas)) / bin_width)) if bin_width > 0 else 20  # Số bins tự động
        num_bins_combined = max(10, min(num_bins_combined, 50))  # Giới hạn số bins

        print(f"\nĐang vẽ biểu đồ so sánh diện tích HR/LR với khoảng {num_bins_combined} bins...")

        all_plot_areas = {}
        if hr_areas:
            all_plot_areas['HR'] = hr_areas
        for scale, areas in plot_data_lr.items():
            all_plot_areas[f'LR x{scale}'] = areas

        # Chọn màu sắc khác nhau
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_plot_areas)))

        # Vẽ các histogram chồng lên nhau với độ trong suốt (alpha)
        bin_edges_combined = np.histogram_bin_edges(combined_areas, bins=num_bins_combined)

        for i, (label, data) in enumerate(all_plot_areas.items()):
            plt.hist(data, bins=bin_edges_combined, alpha=0.6, label=label, color=colors[i])

        plt.legend(loc='upper right')
        plt.title('Comparison of HR and LR Image Area Distribution')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Number of photos')
        formatter = mticker.FuncFormatter(lambda x, p: format(int(x), ','))
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("\nKhông đủ dữ liệu HR hoặc LR để vẽ biểu đồ so sánh.")

    # --- (Tùy chọn) Biểu đồ 4: So sánh diện tích HR vs LR (Box Plot) ---
    box_plot_data = []
    box_plot_labels = []
    if hr_areas:
        box_plot_data.append(hr_areas)
        box_plot_labels.append('HR')
    for scale, areas in plot_data_lr.items():
        box_plot_data.append(areas)
        box_plot_labels.append(f'LR x{scale}')

    if len(box_plot_data) > 1:  # Cần ít nhất 2 nhóm để so sánh
        plt.figure(figsize=(10, 6))
        plt.boxplot(box_plot_data, labels=box_plot_labels, patch_artist=True,
                    showfliers=True)  # patch_artist để tô màu, showfliers=False để ẩn điểm ngoại lai nếu muốn
        plt.title('Compare HR and LR Image Area Statistics (Box Plot)')
        plt.ylabel('Area (pixels)')
        formatter = mticker.FuncFormatter(lambda y, p: format(int(y), ','))
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Trả về kết quả chi tiết nếu cần
    return {
        "total_images": len(all_images),
        "min_resolution_img": min_res_img,
        "max_resolution_img": max_res_img,
        "min_resolution_count": count_min_res,
        "max_resolution_count": count_max_res,
        "most_common_resolutions": most_common_resolutions,
        "most_common_area_range": (
        most_common_range_start, most_common_range_end) if 'most_common_range_start' in locals() else None,
        "most_common_area_range_count": most_common_count if 'most_common_count' in locals() else None,
        "hr_analysis": {
            "count": len(hr_images),
            "min_res": hr_images[0] if hr_images else None,
            "max_res": hr_images[-1] if hr_images else None,
            "areas": hr_areas,
        } if hr_images else {"count": 0, "areas": []},
        "lr_analysis": {
            f"scale_{scale}": {
                "count": len(images),
                "min_res": images[0] if images else None,
                "max_res": images[-1] if images else None,
                "areas": lr_areas_by_scale.get(scale, []),
            } if images else {"count": 0, "areas": []}
            for scale, images in lr_images.items()
        }
    }


# --- Cách sử dụng ---
dataset_directory = 'D:\\EnhanceVideo_ImageDLM\\data\\RealSR(V3)'  # Thay đổi đường dẫn này

# (Giữ nguyên phần tạo dữ liệu giả nếu cần test)
# ... (code tạo dữ liệu giả như trước) ...
if not os.path.exists(dataset_directory):
    print("Tạo cấu trúc thư mục giả để demo...")
    os.makedirs(os.path.join(dataset_directory, 'Canon', 'Test', '2'), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, 'Canon', 'Train', '4'), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, 'Nikon', 'Test', '4'), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, 'Nikon', 'Test', '3'), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, 'Nikon', 'Train', '2'), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, 'Nikon', 'Train', '3'), exist_ok=True)  # Thêm thư mục cho scale 3

    # # Tạo vài file ảnh giả với kích thước khác nhau
    # try:
    #     # Cặp 1 (Canon, x2)
    #     Image.new('RGB', (100, 75)).save(os.path.join(dataset_directory, 'Canon', 'Test', '2', 'img001_LR2.png'))
    #     Image.new('RGB', (200, 150)).save(os.path.join(dataset_directory, 'Canon', 'Test', '2', 'img001_HR.png'))
    #     # Cặp 2 (Nikon, x2) - size khác
    #     Image.new('RGB', (150, 100)).save(os.path.join(dataset_directory, 'Nikon', 'Train', '2', 'img002_LR2.png'))
    #     Image.new('RGB', (300, 200)).save(os.path.join(dataset_directory, 'Nikon', 'Train', '2', 'img002_HR.png'))
    #     # Cặp 3 (Nikon, x3)
    #     Image.new('RGB', (120, 90)).save(os.path.join(dataset_directory, 'Nikon', 'Test', '3', 'img003_LR3.png'))
    #     Image.new('RGB', (360, 270)).save(
    #         os.path.join(dataset_directory, 'Nikon', 'Test', '3', 'img003_HR.png'))  # Sửa HR size cho đúng tỉ lệ 3
    #     # Cặp 4 (Canon, x4) - size lớn
    #     Image.new('RGB', (200, 150)).save(os.path.join(dataset_directory, 'Canon', 'Train', '4', 'img004_LR4.png'))
    #     Image.new('RGB', (800, 600)).save(os.path.join(dataset_directory, 'Canon', 'Train', '4', 'img004_HR.png'))
    #     # Cặp 5 (Canon, x4) - size nhỏ hơn, trùng LR size
    #     Image.new('RGB', (100, 75)).save(os.path.join(dataset_directory, 'Canon', 'Train', '4', 'img005_LR4.png'))
    #     Image.new('RGB', (400, 300)).save(os.path.join(dataset_directory, 'Canon', 'Train', '4', 'img005_HR.png'))
    #     # Thêm vài ảnh nữa để đa dạng hóa độ phân giải
    #     Image.new('RGB', (100, 75)).save(
    #         os.path.join(dataset_directory, 'Nikon', 'Train', '2', 'img006_LR2.png'))  # Thêm LR trùng size
    #     Image.new('RGB', (200, 150)).save(
    #         os.path.join(dataset_directory, 'Nikon', 'Train', '2', 'img006_HR.png'))  # Thêm HR trùng size
    #     Image.new('RGB', (160, 120)).save(os.path.join(dataset_directory, 'Nikon', 'Train', '3', 'img007_LR3.png'))
    #     Image.new('RGB', (480, 360)).save(os.path.join(dataset_directory, 'Nikon', 'Train', '3', 'img007_HR.png'))
    #
    #     print("Đã tạo file ảnh giả.")
    # except ImportError:
    #     print("\nLưu ý: Không thể tạo ảnh giả vì thiếu thư viện Pillow (PIL).")
    #     print("Vui lòng cài đặt: pip install Pillow")
    # except Exception as e:
    #     print(f"Lỗi khi tạo ảnh giả: {e}")


# Chạy phân tích
results = analyze_dataset_resolutions(dataset_directory)

# if results:
#      # Bạn có thể xem cấu trúc trả về nếu cần
#      # import json
#      # print(json.dumps(results, indent=2, default=str))
#      pass