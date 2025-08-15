import os


def get_dataset_info(dataset_path):
    dataset_info = {}

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        dataset_info[split] = {}

        for lr_hr in ['HR', 'LR']:
            lr_hr_path = os.path.join(split_path, lr_hr)
            images = os.listdir(lr_hr_path)
            num_images = len(images)
            total_size = sum(os.path.getsize(os.path.join(lr_hr_path, img)) for img in images)

            dataset_info[split][lr_hr] = {
                'num_images': num_images,
                'total_size_gb': total_size / (1024 ** 3)
            }

    return dataset_info


dataset_path = r'D:\EnhanceVideo_ImageDLM\DIV2K+_scale_4x'
info = get_dataset_info(dataset_path)

for split, details in info.items():
    print(f"{split.capitalize()}:")
    for lr_hr, stats in details.items():
        print(f"  {lr_hr}:")
        print(f"    Số ảnh: {stats['num_images']}")
        print(f"    Kích thước (GB): {stats['total_size_gb']:.2f}")



# import os
# from PIL import Image
#
# def count_large_images(hr_path, min_size=(512, 512)):
#     count = 0
#     total_images = 0
#
#     for img_file in os.listdir(hr_path):
#         img_path = os.path.join(hr_path, img_file)
#         if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Kiểm tra định dạng ảnh
#             total_images += 1
#             with Image.open(img_path) as img:
#                 if img.size[0] > min_size[0] and img.size[1] > min_size[1]:
#                     count += 1
#
#     return count, total_images
#
# def count_large_hr_images_in_datasets(base_path):
#     results = {}
#     for split in ['train', 'val', 'test']:
#         hr_directory = os.path.join(base_path, split, 'HR')
#         large_count, total_count = count_large_images(hr_directory)
#         results[split] = {
#             'large_count': large_count,
#             'total_count': total_count
#         }
#     return results
#
# dataset_path = r'D:\DL_for_enhance_video_image_quality\dataset'  # Thay đổi đường dẫn nếu cần
# results = count_large_hr_images_in_datasets(dataset_path)
#
# for split, counts in results.items():
#     print(f"{split.capitalize()}:")
#     print(f"  Số ảnh HR lớn hơn 512x512: {counts['large_count']} / {counts['total_count']} ảnh")