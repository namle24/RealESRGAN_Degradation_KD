import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealisticDegradationGenerator:
    def __init__(self):
        self.resize_prob = 1.0
        self.noise_prob = 0.9
        self.blur_prob = 0.8
        self.jpeg_prob = 0.9
        self.camera_noise_prob = 0.7
        self.blur_kernel_size = 41
        self.noise_sigma_range = [1, 30]
        self.jpeg_quality_range = [30, 95]
        self.blur_types = ['gaussian', 'aniso', 'generalized', 'motion', 'defocus', 'sinc']
        self.blur_probs = [0.3, 0.15, 0.1, 0.2, 0.2, 0.05]

    def _random_resize(self, img, scale_factor=0.5):
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        method = cv2.INTER_AREA
        if random.random() < 0.3:
            aspect_jitter = random.uniform(0.9, 1.1)
            new_w = int(new_w * aspect_jitter)
        return cv2.resize(img, (new_w, new_h), interpolation=method)

    def _add_noise(self, img, is_second=False):
        noise_type = random.choice(['gaussian', 'sp', 'poisson', 'mixed'])
        sigma_range = [1, 20] if is_second else self.noise_sigma_range
        if noise_type == 'gaussian':
            sigma = random.uniform(*sigma_range)
            gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        elif noise_type == 'sp':
            prob = random.uniform(0.001, 0.01)
            output = img.copy()
            salt_mask = np.random.random(img.shape[:2]) < prob / 2
            output[salt_mask] = 255
            pepper_mask = np.random.random(img.shape[:2]) < prob / 2
            output[pepper_mask] = 0
            return output
        elif noise_type == 'poisson':
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            output = np.random.poisson(img * vals) / float(vals)
            return np.clip(output, 0, 255).astype(np.uint8)
        else:
            temp = self._add_noise(img, is_second)
            return self._add_noise(temp, is_second)

    def _add_blur(self, img):
        blur_type = random.choices(self.blur_types, self.blur_probs, k=1)[0]
        if blur_type == 'gaussian':
            sigma = random.uniform(0.2, 3.0)
            return cv2.GaussianBlur(img, (self.blur_kernel_size, self.blur_kernel_size), sigma)
        elif blur_type == 'aniso':
            sigma_x = random.uniform(0.2, 3.0)
            sigma_y = random.uniform(0.2, 3.0)
            angle = random.uniform(0, 180)
            kernel = self._create_aniso_kernel(sigma_x, sigma_y, angle, self.blur_kernel_size)
            return np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)
        elif blur_type == 'generalized':
            sigma = random.uniform(0.2, 3.0)
            beta = random.uniform(0.5, 4.0)
            x = np.arange(-self.blur_kernel_size // 2 + 1, self.blur_kernel_size // 2 + 1)
            kernel = np.exp(-np.abs(x / sigma) ** beta)
            kernel /= kernel.sum()
            kernel_2d = np.outer(kernel, kernel)
            return np.clip(cv2.filter2D(img, -1, kernel_2d), 0, 255).astype(np.uint8)
        elif blur_type == 'motion':
            degree = random.randint(5, 50)
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.hamming(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
            motion_blur_kernel /= motion_blur_kernel.sum()
            blurred = cv2.filter2D(img, -1, motion_blur_kernel)
            if random.random() < 0.2:
                blurred = self._add_blur(blurred)
            return np.clip(blurred, 0, 255).astype(np.uint8)
        elif blur_type == 'defocus':
            radius = random.randint(1, 10)
            return cv2.GaussianBlur(img, (radius * 2 + 1, radius * 2 + 1), 0)
        else:  # sinc
            kernel_size = random.randint(7, self.blur_kernel_size)
            kernel = np.sinc(np.linspace(-3, 3, kernel_size)).astype(np.float32)
            kernel /= kernel.sum()
            kernel_2d = np.outer(kernel, kernel)
            return cv2.filter2D(img, -1, kernel_2d)

    def _create_aniso_kernel(self, sigma_x, sigma_y, angle, size):
        x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 + 1))
        x_rot = x * np.cos(np.deg2rad(angle)) + y * np.sin(np.deg2rad(angle))
        y_rot = -x * np.sin(np.deg2rad(angle)) + y * np.cos(np.deg2rad(angle))
        kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
        kernel /= kernel.sum()
        return kernel

    def _jpeg_compression(self, img):
        quality = random.randint(*self.jpeg_quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

    def _add_camera_noise(self, img):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            b, g, r = cv2.split(img)
            shift = random.randint(3, 10)
            if random.random() < 0.5:
                r = np.pad(r, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
                b = np.pad(b, ((0, 0), (0, shift)), mode='edge')[:, shift:]
            else:
                r = np.pad(r, ((shift, 0), (0, 0)), mode='edge')[:-shift, :]
                b = np.pad(b, ((0, shift), (0, 0)), mode='edge')[shift:, :]
            img = cv2.merge([b, g, r])
        if random.random() < 0.4:
            pattern = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + pattern, 0, 255).astype(np.uint8)
        return img

    def apply_degradation(self, img):
        degraded = img.copy()
        steps = self._get_random_steps()
        degraded = self._apply_steps(degraded, steps, scale_factor=0.5)
        steps = self._get_random_steps()
        degraded = self._apply_steps(degraded, steps, None)
        degraded = self._add_sinc_filter(degraded)
        return degraded

    def _add_sinc_filter(self, img):
        if random.random() < 0.8:
            kernel_size = random.randint(7, self.blur_kernel_size)
            kernel = np.sinc(np.linspace(-3, 3, kernel_size)).astype(np.float32)
            kernel /= kernel.sum()
            kernel_2d = np.outer(kernel, kernel)
            return cv2.filter2D(img, -1, kernel_2d)
        return img

    def _get_random_steps(self):
        steps = []
        if random.random() < self.resize_prob:
            steps.append('resize')
        if random.random() < self.blur_prob:
            steps.append('blur')
        if random.random() < self.noise_prob:
            steps.append('noise')
        if random.random() < self.camera_noise_prob:
            steps.append('camera_noise')
        if random.random() < self.jpeg_prob:
            steps.append('jpeg')
        random.shuffle(steps[1:] if 'resize' in steps else steps)
        if 'resize' in steps:
            steps = ['resize'] + steps[1:]
        return steps

    def _apply_steps(self, img, steps, scale_factor):
        degraded = img.copy()
        for step in steps:
            if step == 'resize' and scale_factor is not None:
                degraded = self._random_resize(degraded, scale_factor)
            elif step == 'blur':
                degraded = self._add_blur(degraded)
            elif step == 'noise':
                degraded = self._add_noise(degraded, is_second=(scale_factor is not None))
            elif step == 'jpeg':
                degraded = self._jpeg_compression(degraded)
            elif step == 'camera_noise':
                degraded = self._add_camera_noise(degraded)
        return degraded

generator_instance = None

def init_worker():
    global generator_instance
    generator_instance = RealisticDegradationGenerator()

def process_image(args):
    img_path, hr_output_path, lr_output_path = args
    global generator_instance
    try:
        if os.path.exists(lr_output_path) and os.path.exists(hr_output_path):
            return
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Không thể đọc ảnh: {img_path}")
            return
        # Resize HR về 1024x1024
        hr_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        # Tạo LR từ HR 1024x1024
        lr_img = generator_instance.apply_degradation(hr_img)
        os.makedirs(os.path.dirname(hr_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(lr_output_path), exist_ok=True)
        cv2.imwrite(hr_output_path, hr_img)
        cv2.imwrite(lr_output_path, lr_img)
        logging.info(f"Đã xử lý: {hr_output_path} và {lr_output_path}")
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {img_path}: {e}")

def process_dataset(input_dir, output_dir, n_workers=cpu_count() - 1):
    for split in ['train', 'val', 'test']:
        hr_input_dir = os.path.join(input_dir, split, 'HR')
        hr_output_dir = os.path.join(output_dir, split, 'HR')
        lr_output_dir = os.path.join(output_dir, split, 'LR')
        if not os.path.exists(hr_input_dir):
            logging.warning(f"Thư mục không tồn tại: {hr_input_dir}")
            continue
        args_list = []
        for img_file in os.listdir(hr_input_dir):
            if img_file.endswith((".jpg", ".png")):
                hr_path = os.path.join(hr_input_dir, img_file)
                hr_output_path = os.path.join(hr_output_dir, img_file)
                lr_output_path = os.path.join(lr_output_dir, img_file)
                args_list.append((hr_path, hr_output_path, lr_output_path))
        logging.info(f"Đang xử lý {len(args_list)} ảnh trong {split}...")
        with Pool(n_workers, initializer=init_worker) as pool:
            list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))
        # Kiểm tra kết quả
        hr_count = len(os.listdir(hr_output_dir))
        lr_count = len(os.listdir(lr_output_dir))
        logging.info(f"{split} - HR: {hr_count} ảnh, LR: {lr_count} ảnh")

# Chạy
input_dir = 'D:\\EnhanceVideo_ImageDLM\\dataset'
output_dir = 'D:\\EnhanceVideo_ImageDLM\\dataset_resized'
process_dataset(input_dir, output_dir)
logging.info("Đã xử lý toàn bộ dữ liệu thành công!")