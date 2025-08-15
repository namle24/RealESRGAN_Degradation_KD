import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealisticDegradationGenerator:
    def __init__(self,
                 resize_prob=1.0,
                 noise_prob=0.9,
                 blur_prob=0.8,
                 jpeg_prob=0.9,
                 camera_noise_prob=0.7,
                 blur_kernel_size=21,
                 noise_sigma_range=[1, 25],
                 jpeg_quality_range=[50, 95]):
        self.resize_prob = resize_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.jpeg_prob = jpeg_prob
        self.camera_noise_prob = camera_noise_prob
        self.blur_kernel_size = blur_kernel_size
        self.noise_sigma_range = noise_sigma_range
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_types = ['gaussian', 'aniso', 'generalized', 'motion', 'defocus', 'sinc']
        self.blur_probs = [0.3, 0.15, 0.1, 0.2, 0.15, 0.2]

    def _random_resize(self, img, scale_factor=None):
        if scale_factor is None:
            scale_factor = random.uniform(0.5, 0.75)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        method = random.choice(methods)
        return cv2.resize(img, (new_w, new_h), interpolation=method)

    def _add_noise(self, img, is_second=False):
        noise_type = random.choice(['gaussian', 'sp', 'poisson', 'mixed'])
        sigma_range = [1, 20] if is_second else self.noise_sigma_range
        if noise_type == 'gaussian':
            mean = 0
            sigma = random.uniform(*sigma_range)
            gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
            noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
            return noisy
        elif noise_type == 'sp':
            prob = random.uniform(0.001, 0.01)
            output = np.copy(img)
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
        else:  # mixed
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
            blurred = cv2.filter2D(img, -1, kernel)
            return np.clip(blurred, 0, 255).astype(np.uint8)
        elif blur_type == 'generalized':
            sigma = random.uniform(0.2, 3.0)
            beta = random.uniform(0.5, 4.0)
            x = np.arange(-self.blur_kernel_size // 2 + 1, self.blur_kernel_size // 2 + 1)
            kernel = np.exp(-np.abs(x / sigma) ** beta)
            kernel /= kernel.sum()
            kernel_2d = np.outer(kernel, kernel)
            blurred = cv2.filter2D(img, -1, kernel_2d)
            return np.clip(blurred, 0, 255).astype(np.uint8)
        elif blur_type == 'motion':
            degree = random.randint(5, 30)
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.hamming(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
            motion_blur_kernel = motion_blur_kernel / motion_blur_kernel.sum()
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
            blurred = cv2.filter2D(img, -1, kernel_2d)
            return np.clip(blurred, 0, 255).astype(np.uint8)

    def _create_aniso_kernel(self, sigma_x, sigma_y, angle, size):
        x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                           np.arange(-size // 2 + 1, size // 2 + 1))
        x_rot = x * np.cos(np.deg2rad(angle)) + y * np.sin(np.deg2rad(angle))
        y_rot = -x * np.sin(np.deg2rad(angle)) + y * np.cos(np.deg2rad(angle))
        kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
        kernel /= kernel.sum()
        return kernel

    def _jpeg_compression(self, img):
        quality = random.randint(*self.jpeg_quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg

    def _add_camera_noise(self, img):
        h, w = img.shape[:2]
        if random.random() < 0.4:
            b, g, r = cv2.split(img)
            shift = random.randint(1, 3)
            if random.random() < 0.5:
                r = np.pad(r, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
                b = np.pad(b, ((0, 0), (0, shift)), mode='edge')[:, shift:]
                if r.shape != g.shape:
                    r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                if b.shape != g.shape:
                    b = cv2.resize(b, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                r = np.pad(r, ((shift, 0), (0, 0)), mode='edge')[:-shift, :]
                b = np.pad(b, ((0, shift), (0, 0)), mode='edge')[shift:, :]
                if r.shape != g.shape:
                    r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                if b.shape != g.shape:
                    b = cv2.resize(b, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            img = cv2.merge([b, g, r])
        if random.random() < 0.3:
            pattern = np.random.normal(0, random.uniform(1, 3), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + pattern, 0, 255).astype(np.uint8)
        return img

    def apply_degradation(self, img, scale_factor):
        degraded = img.copy()
        steps = self._get_random_steps()
        if 'resize' not in steps:
            steps.insert(0, 'resize')
        degraded = self._apply_steps(degraded, steps, scale_factor)
        steps = self._get_random_steps()
        if 'resize' in steps:
            steps.remove('resize')
        degraded = self._apply_steps(degraded, steps, None)
        h, w = img.shape[:2]
        target_h, target_w = int(h * scale_factor), int(w * scale_factor)
        current_h, current_w = degraded.shape[:2]
        # Chỉ resize nếu kích thước hiện tại nhỏ hơn target
        if current_h < target_h or current_w < target_w:
            logging.info(f"Resizing {current_w}x{current_h} to {target_w}x{target_h}")
            degraded = cv2.resize(degraded, (target_w, target_h), interpolation=cv2.INTER_AREA)
        elif current_h > target_h or current_w > target_w:
            logging.warning(f"Output {current_w}x{current_h} larger than target {target_w}x{target_h}, resizing down")
            degraded = cv2.resize(degraded, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return degraded

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
            if step == 'resize':
                degraded = self._random_resize(degraded, scale_factor)
            elif step == 'blur':
                degraded = self._add_blur(degraded)
            elif step == 'noise':
                degraded = self._add_noise(degraded, is_second=(scale_factor is None))
            elif step == 'jpeg':
                degraded = self._jpeg_compression(degraded)
            elif step == 'camera_noise':
                degraded = self._add_camera_noise(degraded)
        return degraded

# Instance chung cho multiprocessing
generator_instance = None

def init_worker(scale_factor):
    global generator_instance
    generator_instance = RealisticDegradationGenerator()
    global global_scale_factor
    global_scale_factor = scale_factor

def process_image(args):
    img_path, output_path = args
    global generator_instance, global_scale_factor
    try:
        if os.path.exists(output_path):
            logging.debug(f"Bỏ qua {output_path} vì đã tồn tại")
            return
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Không thể đọc ảnh: {img_path}")
            return
        h, w = img.shape[:2]
        target_h, target_w = int(h * global_scale_factor), int(w * global_scale_factor)
        # Kiểm tra ảnh đầu vào là HR
        if h < target_h or w < target_w:
            logging.warning(f"Ảnh {img_path} không phải HR (kích thước {w}x{h}, target LR {target_w}x{target_h})")
            return
        degraded = generator_instance.apply_degradation(img, global_scale_factor)
        current_h, current_w = degraded.shape[:2]
        if current_h != target_h or current_w != target_w:
            logging.warning(f"Output {current_w}x{current_h} không khớp target {target_w}x{target_h}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, degraded)
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {img_path}: {e}")

def process_dataset(input_dir, output_dir, scale_factor=0.25, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    logging.info(f"Processing with scale factor {scale_factor}")
    for mode in ['train', 'test']:
        mode_dir = os.path.join(input_dir, mode)
        if not os.path.exists(mode_dir):
            logging.warning(f"Thư mục {mode_dir} không tồn tại, bỏ qua")
            continue
        datasets = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
        for dataset in datasets:
            dataset_path = os.path.join(mode_dir, dataset)
            hr_dir = os.path.join(dataset_path, f"{dataset}_HR")
            if not os.path.exists(hr_dir):
                hr_dir = os.path.join(dataset_path, 'HR')
            if not os.path.exists(hr_dir):
                logging.warning(f"Thư mục HR {hr_dir} không tồn tại, bỏ qua")
                continue
            lr_dir = os.path.join(output_dir, mode, dataset, f"{dataset}_LR")
            new_hr_dir = os.path.join(output_dir, mode, dataset, os.path.basename(hr_dir))
            if hr_dir != new_hr_dir:
                os.makedirs(new_hr_dir, exist_ok=True)
                logging.info(f"Tạo liên kết ảnh HR từ {hr_dir} tới {new_hr_dir}")
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
                image_files.extend(glob.glob(os.path.join(hr_dir, '**', ext), recursive=True))
            if not image_files:
                logging.warning(f"Không tìm thấy ảnh HR trong {hr_dir}")
                continue
            args_list = []
            for img_path in image_files:
                rel_path = os.path.relpath(img_path, hr_dir)
                if hr_dir != new_hr_dir:
                    hr_output_path = os.path.join(new_hr_dir, rel_path)
                    os.makedirs(os.path.dirname(hr_output_path), exist_ok=True)
                    if not os.path.exists(hr_output_path):
                        import shutil
                        shutil.move(img_path, hr_output_path)
                lr_output_path = os.path.join(lr_dir, rel_path)
                args_list.append((img_path, lr_output_path))
            logging.info(f"Đang xử lý {len(args_list)} ảnh từ {hr_dir} vào {lr_dir}...")
            with Pool(n_workers, initializer=init_worker, initargs=(scale_factor,)) as pool:
                list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply realistic degradation to images')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing HR images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for degraded LR images')
    parser.add_argument('--scale', type=float, default=0.25, help='Scale factor for degradation (e.g., 0.25 for x4, 0.5 for x2)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.scale, args.workers)