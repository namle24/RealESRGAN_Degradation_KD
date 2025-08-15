import os

# Định nghĩa cấu trúc thư mục
structure = {
    "DIV2K+": {
        "test": {
            "DIV2K_valid": {
                "DIV2K_valid_HR": {},
                "DIV2K_valid_LR": {}
            },
            "OutdoorSceneTest300": {
                "OutdoorSceneTest300_HR": {},
                "OutdoorSceneTest300_LR": {}
            }
        },
        "train": {
            "DIV2K": {
                "DIV2K_train_HR": {},
                "DIV2K_train_LR": {}
            },
            "Flickr2K": {
                "Flickr2K_HR": {},
                "Flickr2K_LR": {}
            },
            "OS": {
                "OS_HR": {},
                "OS_LR": {}
            }
        }
    }
}

# Hàm để tạo thư mục
def create_folders(base_path, structure):
    for folder_name, subfolders in structure.items():
        path = os.path.join(base_path, folder_name)
        os.makedirs(path, exist_ok=True)  # Tạo thư mục
        create_folders(path, subfolders)  # Tạo thư mục con

# Tạo cấu trúc thư mục
create_folders(".", structure)