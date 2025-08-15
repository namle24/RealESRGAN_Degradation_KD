import os
import shutil
from tqdm import tqdm


def merge_dataset(input_dir, output_dir):
    """
    Gộp dữ liệu từ cấu trúc DIV2K+ sang cấu trúc mới bằng cách chỉ sao chép ảnh.

    Args:
        input_dir (str): Thư mục gốc của DIV2K+ hiện tại.
        output_dir (str): Thư mục đầu ra cho cấu trúc mới (dataset).
    """
    # Định nghĩa ánh xạ từ cấu trúc cũ sang cấu trúc mới
    mapping = {
        'train': {
            'HR': [
                os.path.join(input_dir, 'train', 'DIV2K', 'DIV2K_HR'),
                os.path.join(input_dir, 'train', 'Flickr2K', 'Flickr2K_HR'),
                os.path.join(input_dir, 'train', 'OST', 'OST_HR')
            ],
            'LR': [
                os.path.join(input_dir, 'train', 'DIV2K', 'DIV2K_LR'),
                os.path.join(input_dir, 'train', 'Flickr2K', 'Flickr2K_LR'),
                os.path.join(input_dir, 'train', 'OST', 'OST_LR')
            ]
        },
        'val': {
            'HR': [os.path.join(input_dir, 'test', 'DIV2K_valid', 'DIV2K_valid_HR')],
            'LR': [os.path.join(input_dir, 'test', 'DIV2K_valid', 'DIV2K_valid_LR')]
        },
        'test': {
            'HR': [os.path.join(input_dir, 'test', 'OutdoorSceneTest300', 'OutdoorSceneTest300_HR')],
            'LR': [os.path.join(input_dir, 'test', 'OutdoorSceneTest300', 'OutdoorSceneTest300_LR')]
        }
    }

    # Tạo thư mục đầu ra
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'HR'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'LR'), exist_ok=True)

    # Hàm xử lý gộp file
    def merge_files(src_dirs, dst_dir, prefix=None):
        for src_dir in src_dirs:
            if not os.path.exists(src_dir):
                print(f"Thư mục {src_dir} không tồn tại, bỏ qua.")
                continue
            files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            for file in tqdm(files, desc=f"Gộp từ {src_dir} vào {dst_dir}"):
                src_path = os.path.join(src_dir, file)
                dst_path = os.path.join(dst_dir, file)
                # Sao chép
                shutil.copy2(src_path, dst_path)

    # Gộp từng phần
    print("Gộp dữ liệu train...")
    merge_files(mapping['train']['HR'], os.path.join(output_dir, 'train', 'HR'), prefix='train')
    merge_files(mapping['train']['LR'], os.path.join(output_dir, 'train', 'LR'), prefix='train')

    print("Gộp dữ liệu val...")
    merge_files(mapping['val']['HR'], os.path.join(output_dir, 'val', 'HR'), prefix='val')
    merge_files(mapping['val']['LR'], os.path.join(output_dir, 'val', 'LR'), prefix='val')

    print("Gộp dữ liệu test...")
    merge_files(mapping['test']['HR'], os.path.join(output_dir, 'test', 'HR'), prefix='test')
    merge_files(mapping['test']['LR'], os.path.join(output_dir, 'test', 'LR'), prefix='test')

    print("Hoàn thành gộp dữ liệu!")


if __name__ == "__main__":
    input_dir = "D:\\EnhanceVideo_ImageDLM\\data\\DIV2K+degra"  # Thay bằng đường dẫn thực tế
    output_dir = "D:\\EnhanceVideo_ImageDLM\\dataset"  # Thay bằng đường dẫn thực tế
    merge_dataset(input_dir, output_dir)