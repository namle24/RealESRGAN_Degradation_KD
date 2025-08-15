import os

# Đường dẫn thư mục gốc
root_dir = r"D:\Download\archive\RealSR(V3)"

for brand in os.listdir(root_dir):
    brand_path = os.path.join(root_dir, brand)

    if not os.path.isdir(brand_path):
        continue

    for quality in ["HR", "LR"]:
        quality_path = os.path.join(brand_path, quality)

        if not os.path.isdir(quality_path):
            continue

        for filename in os.listdir(quality_path):
            old_path = os.path.join(quality_path, filename)

            if not filename.lower().endswith('.png'):
                continue

            if filename.endswith("_HR.png"):
                new_filename = filename.replace("_HR.png", ".png")
            elif filename.endswith("_LR4.png"):
                new_filename = filename.replace("_LR4.png", ".png")
            else:
                continue

            new_path = os.path.join(quality_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
