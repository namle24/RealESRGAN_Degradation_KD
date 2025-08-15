import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from torchvision.transforms.functional import rgb_to_grayscale
from basicsr.utils import scandir  # Giả sử bạn đã cài BasicSR

def main():
    opt = {}
    opt['n_thread'] = 20  # Số luồng, tùy chỉnh theo CPU của bạn
    opt['compression_level'] = 3  # Mức nén PNG

    # Xử lý từng split: train, val, test
    for split in ['train', 'val', 'test']:
        # HR images
        opt['input_folder'] = f'D:\\EnhanceVideo_ImageDLM\\dataset\\{split}\\HR'
        opt['save_folder'] = f'D:\\EnhanceVideo_ImageDLM\\dataset_patches\\{split}\\HR_sub'
        opt['crop_size'] = 720  # HR patch 720x720
        opt['step'] = 360       # Overlap 50%
        opt['thresh_size'] = 0
        extract_subimages(opt)

        # LR images (thật + degradation)
        opt['input_folder'] = f'D:\\EnhanceVideo_ImageDLM\\dataset\\{split}\\LR'
        opt['save_folder'] = f'D:\\EnhanceVideo_ImageDLM\\dataset_patches\\{split}\\LR_sub'
        opt['crop_size'] = 360  # LR patch 360x360 (scale 2x)
        opt['step'] = 180
        opt['thresh_size'] = 0
        extract_subimages(opt)

def extract_subimages(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

def worker(path, opt):
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return f'Failed to read {img_name}'

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'),
                cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']]
            )
    process_info = f'Processing {img_name} ...'
    return process_info

if __name__ == '__main__':
    main()