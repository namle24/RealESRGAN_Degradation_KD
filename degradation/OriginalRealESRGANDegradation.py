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

class OriginalRealESRGANDegradation:
    def __init__(self,
                 # First degradation process params
                 resize_prob=[0.2, 0.7, 0.1],  # up, down, keep
                 resize_range=[0.15, 1.5],
                 gaussian_noise_prob=0.5,
                 noise_range=[1, 30],
                 poisson_scale_range=[0.05, 3],
                 gray_noise_prob=0.4,
                 jpeg_range=[30, 95],
                 # Second degradation process params
                 second_blur_prob=0.8,
                 resize_prob2=[0.3, 0.4, 0.3],  # up, down, keep
                 resize_range2=[0.3, 1.2],
                 gaussian_noise_prob2=0.5,
                 noise_range2=[1, 25],
                 poisson_scale_range2=[0.05, 2.5],
                 gray_noise_prob2=0.4,
                 jpeg_range2=[5, 50],
                 # Kernel sizes
                 kernel_size=21,
                 kernel_range=[0.2, 3.0],
                 kernel_types=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'sinc'],
                 kernel_probs=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1]):

        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob
        self.jpeg_range = jpeg_range

        self.second_blur_prob = second_blur_prob
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.gray_noise_prob2 = gray_noise_prob2
        self.jpeg_range2 = jpeg_range2

        self.kernel_size = kernel_size if kernel_size % 2 == 1 and kernel_size > 0 else 21
        self.kernel_range = kernel_range
        self.kernel_types = kernel_types
        self.kernel_probs = kernel_probs

        # Khởi tạo kernel
        self.kernel1 = self._generate_blur_kernel()
        if self.kernel1 is None:
            logging.warning("Failed to generate kernel1, using default Gaussian kernel.")
            self.kernel1 = self._generate_gaussian_kernel(self.kernel_size, random.uniform(self.kernel_range[0], self.kernel_range[1]))
        self.kernel2 = self._generate_blur_kernel()
        if self.kernel2 is None:
            logging.warning("Failed to generate kernel2, using default Gaussian kernel.")
            self.kernel2 = self._generate_gaussian_kernel(self.kernel_size, random.uniform(self.kernel_range[0], self.kernel_range[1]))
        self.sinc_kernel = self._generate_sinc_kernel(self.kernel_size)

    def _generate_blur_kernel(self, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        kernel_type = random.choices(self.kernel_types, weights=self.kernel_probs, k=1)[0]
        sigma_x = random.uniform(self.kernel_range[0], self.kernel_range[1])
        sigma_y = random.uniform(self.kernel_range[0], self.kernel_range[1])
        angle = random.uniform(-3.1416, 3.1416)
        beta = random.uniform(0.5, 4)  # Tham số beta cho generalized và plateau
        plateau_width = random.uniform(0.1, 0.5)  # Chiều rộng vùng plateau

        if kernel_type == 'iso':
            sigma = sigma_x
            return self._generate_gaussian_kernel(kernel_size, sigma)
        elif kernel_type == 'aniso':
            return self._generate_aniso_kernel(kernel_size, sigma_x, sigma_y, angle)
        elif kernel_type == 'generalized_iso':
            return self._generate_generalized_gaussian_kernel(kernel_size, sigma_x, beta)
        elif kernel_type == 'generalized_aniso':
            return self._generate_generalized_aniso_kernel(kernel_size, sigma_x, sigma_y, angle, beta)
        elif kernel_type == 'plateau_iso':
            return self._generate_plateau_kernel(kernel_size, sigma_x, plateau_width)
        elif kernel_type == 'plateau_aniso':
            return self._generate_plateau_aniso_kernel(kernel_size, sigma_x, sigma_y, angle, plateau_width)
        elif kernel_type == 'sinc':
            return self._generate_sinc_kernel(kernel_size)
        return None

    def _generate_gaussian_kernel(self, kernel_size=21, sigma=1.0):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        return kernel / np.sum(kernel)

    def _generate_aniso_kernel(self, kernel_size=21, sigma_x=1.0, sigma_y=1.0, angle=0.0):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        x, y = np.meshgrid(x, y)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        kernel = np.exp(-0.5 * ((x_rot**2 / sigma_x**2) + (y_rot**2 / sigma_y**2)))
        return kernel / np.sum(kernel)

    def _generate_generalized_gaussian_kernel(self, kernel_size=21, sigma=1.0, beta=1.0):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        kernel = np.exp(-((r / sigma) ** beta))
        return kernel / np.sum(kernel)

    def _generate_generalized_aniso_kernel(self, kernel_size=21, sigma_x=1.0, sigma_y=1.0, angle=0.0, beta=1.0):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        x, y = np.meshgrid(x, y)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        kernel = np.exp(-((np.abs(x_rot) / sigma_x) ** beta + (np.abs(y_rot) / sigma_y) ** beta))
        return kernel / np.sum(kernel)

    def _generate_plateau_kernel(self, kernel_size=21, sigma=1.0, plateau_width=0.3):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        kernel = np.ones_like(xx, dtype=np.float32)
        plateau_radius = kernel_size * plateau_width / 2
        gaussian_part = np.exp(-0.5 * ((r - plateau_radius) / sigma) ** 2)
        kernel[r <= plateau_radius] = 1.0
        kernel[r > plateau_radius] = gaussian_part[r > plateau_radius]
        return kernel / np.sum(kernel)

    def _generate_plateau_aniso_kernel(self, kernel_size=21, sigma_x=1.0, sigma_y=1.0, angle=0.0, plateau_width=0.3):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        x, y = np.meshgrid(x, y)
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        r_x = np.abs(x_rot) / sigma_x
        r_y = np.abs(y_rot) / sigma_y
        kernel = np.ones_like(x_rot, dtype=np.float32)
        plateau_radius_x = kernel_size * plateau_width / 2
        plateau_radius_y = kernel_size * plateau_width / 2
        gaussian_part = np.exp(-0.5 * (((r_x - plateau_radius_x) / sigma_x) ** 2 + ((r_y - plateau_radius_y) / sigma_y) ** 2))
        kernel[(r_x <= plateau_radius_x) & (r_y <= plateau_radius_y)] = 1.0
        kernel[(r_x > plateau_radius_x) | (r_y > plateau_radius_y)] = gaussian_part[(r_x > plateau_radius_x) | (r_y > plateau_radius_y)]
        return kernel / np.sum(kernel)

    def _generate_sinc_kernel(self, kernel_size=21):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            logging.warning(f"Invalid kernel size {kernel_size}. Using default size 21.")
            kernel_size = 21
        x = np.linspace(-3, 3, kernel_size)
        y = np.linspace(-3, 3, kernel_size)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2) + 1e-6
        sinc = np.sin(r) / r
        return sinc / np.sum(sinc)

    def _usm_sharp(self, img, amount=1.0, radius=50, threshold=10):
        if radius <= 0 or radius % 2 == 0:
            radius = 51
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        sharp = float(amount + 1) * img - float(amount) * blur
        sharp = np.maximum(sharp, np.zeros(sharp.shape))
        sharp = np.minimum(sharp, 255 * np.ones(sharp.shape))
        mask = np.abs(img - blur) > threshold
        result = img * (1 - mask) + sharp * mask
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_blur(self, img, kernel):
        if kernel is None or kernel.shape[0] <= 0 or kernel.shape[1] <= 0 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            logging.warning(f"Invalid kernel {kernel}. Using default Gaussian blur.")
            return cv2.GaussianBlur(img, (21, 21), 0)
        result = cv2.filter2D(img, -1, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _random_resize(self, img, updown_type, scale_range, target_size=None):
        h, w = img.shape[:2]
        if updown_type == 'up':
            scale = np.random.uniform(1, scale_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(scale_range[0], 1)
        else:
            scale = 1.0
        if target_size is not None:
            target_w, target_h = target_size
            scale = target_w / w
        methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        method = random.choice(methods)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        result = cv2.resize(img, (new_w, new_h), interpolation=method)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_gaussian_noise(self, img, sigma_range, gray_prob):
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        if random.random() < gray_prob and len(img.shape) == 3:
            noise = np.random.normal(0, sigma, img.shape[:2])
            noise = np.stack([noise] * 3, axis=2)
        else:
            noise = np.random.normal(0, sigma, img.shape)
        result = img.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_poisson_noise(self, img, scale_range, gray_prob):
        scale = random.uniform(scale_range[0], scale_range[1])
        img_norm = img.astype(np.float32) / 255.0
        img_norm = np.clip(img_norm, 0, 1)  # Giới hạn giá trị trong [0, 1]
        if np.any(np.isnan(img_norm)):
            logging.warning("NaN values found in img_norm, replacing with 0.")
            img_norm = np.nan_to_num(img_norm, nan=0.0)
        lam = img_norm * scale
        if random.random() < gray_prob and len(img.shape) == 3:
            noise = np.random.poisson(lam[:, :, 0]) / scale
            noise = np.stack([noise] * 3, axis=2)
        else:
            noise = np.random.poisson(lam) / scale
        result = noise * 255.0
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_noise(self, img, noise_range, poisson_scale_range, gaussian_noise_prob, gray_noise_prob):
        if random.random() < gaussian_noise_prob:
            return self._add_gaussian_noise(img, noise_range, gray_noise_prob)
        else:
            return self._add_poisson_noise(img, poisson_scale_range, gray_noise_prob)

    def _jpeg_compression(self, img, quality_range):
        quality = random.randint(quality_range[0], quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if not result:
            logging.warning("JPEG encoding failed, returning original image")
            return img
        decimg = cv2.imdecode(encimg, 1)
        return np.clip(decimg, 0, 255).astype(np.uint8)

    def _flip(self, img, direction='horizontal'):
        if direction == 'horizontal':
            return cv2.flip(img, 1)
        elif direction == 'vertical':
            return cv2.flip(img, 0)
        return img

    def apply_first_degradation(self, img, target_size=None):
        img_usm = self._usm_sharp(img)
        degraded = img_usm.copy()
        steps = ['blur', 'resize', 'noise', 'jpeg']
        random.shuffle(steps)
        for step in steps:
            if step == 'blur':
                degraded = self._add_blur(degraded, self.kernel1)
            elif step == 'resize':
                updown_type = random.choices(['up', 'down', 'keep'], weights=self.resize_prob)[0]
                degraded = self._random_resize(degraded, updown_type, self.resize_range)
            elif step == 'noise':
                degraded = self._add_noise(degraded, self.noise_range, self.poisson_scale_range,
                                          self.gaussian_noise_prob, self.gray_noise_prob)
            elif step == 'jpeg':
                degraded = self._jpeg_compression(degraded, self.jpeg_range)
        return degraded

    def apply_second_degradation(self, img, target_size=None):
        degraded = img.copy()
        steps = ['blur', 'resize', 'noise']
        random.shuffle(steps)
        for step in steps:
            if step == 'blur' and random.random() < self.second_blur_prob:
                degraded = self._add_blur(degraded, self.kernel2)
            elif step == 'resize':
                updown_type = random.choices(['up', 'down', 'keep'], weights=self.resize_prob2)[0]
                degraded = self._random_resize(degraded, updown_type, self.resize_range2)
            elif step == 'noise':
                degraded = self._add_noise(degraded, self.noise_range2, self.poisson_scale_range2,
                                          self.gaussian_noise_prob2, self.gray_noise_prob2)
        if random.random() < 0.5:
            degraded = self._add_blur(degraded, self.sinc_kernel)
            degraded = self._jpeg_compression(degraded, self.jpeg_range2)
        else:
            degraded = self._jpeg_compression(degraded, self.jpeg_range2)
            degraded = self._add_blur(degraded, self.sinc_kernel)
        return degraded

    def apply_degradation(self, img, scale_factor):
        h, w = img.shape[:2]
        target_h, target_w = int(h * scale_factor), int(w * scale_factor)
        target_h = max(1, target_h)
        target_w = max(1, target_w)
        target_size = (target_w, target_h)

        degraded = self.apply_first_degradation(img)
        degraded = self.apply_second_degradation(degraded)

        if random.random() < 0.5:
            degraded = self._flip(degraded, 'horizontal')
        if random.random() < 0.5:
            degraded = self._flip(degraded, 'vertical')

        current_h, current_w = degraded.shape[:2]
        if current_h != target_h or current_w != target_w:
            logging.info(f"Resizing from {current_w}x{current_h} to {target_w}x{target_h} to match scale_factor {scale_factor}")
            degraded = cv2.resize(degraded, target_size, interpolation=cv2.INTER_AREA)
            degraded = np.clip(degraded, 0, 255).astype(np.uint8)

        return degraded

# Instance used for multiprocessing
generator_instance = None

def init_worker(scale_factor):
    global generator_instance
    generator_instance = OriginalRealESRGANDegradation()
    global global_scale_factor
    global_scale_factor = scale_factor

def process_image(args):
    img_path, output_path = args
    global generator_instance, global_scale_factor
    try:
        if os.path.exists(output_path):
            logging.debug(f"Skip {output_path} because it already exists")
            return
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Could not read image: {img_path}")
            return
        h, w = img.shape[:2]
        target_h, target_w = int(h * global_scale_factor), int(w * global_scale_factor)
        target_h = max(1, target_h)
        target_w = max(1, target_w)

        if h < target_h or w < target_w:
            logging.warning(f"Image {img_path} is not HR (size {w}x{h}, target LR {target_w}x{target_h})")
            return

        try:
            degraded = generator_instance.apply_degradation(img, global_scale_factor)
            current_h, current_w = degraded.shape[:2]
            if current_h != target_h or current_w != target_w:
                logging.warning(f"Output {current_w}x{current_h} doesn't match target {target_w}x{target_h}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, degraded)
        except Exception as e:
            logging.error(f"Error during degradation of {img_path}: {str(e)}")

    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")

def process_dataset(input_dir, output_dir, scale_factor=0.25, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    logging.info(f"Processing with scale factor {scale_factor}")
    for mode in ['train', 'test']:
        mode_dir = os.path.join(input_dir, mode)
        if not os.path.exists(mode_dir):
            logging.warning(f"Directory {mode_dir} does not exist, skipping")
            continue
        datasets = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
        for dataset in datasets:
            dataset_path = os.path.join(mode_dir, dataset)
            hr_dir = os.path.join(dataset_path, f"{dataset}_HR")
            if not os.path.exists(hr_dir):
                hr_dir = os.path.join(dataset_path, 'HR')
            if not os.path.exists(hr_dir):
                logging.warning(f"HR directory {hr_dir} does not exist, skipping")
                continue
            lr_dir = os.path.join(output_dir, mode, dataset, f"{dataset}_LR")
            new_hr_dir = os.path.join(output_dir, mode, dataset, os.path.basename(hr_dir))
            if hr_dir != new_hr_dir:
                os.makedirs(new_hr_dir, exist_ok=True)
                logging.info(f"Creating link for HR images from {hr_dir} to {new_hr_dir}")
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
                image_files.extend(glob.glob(os.path.join(hr_dir, '**', ext), recursive=True))
            if not image_files:
                logging.warning(f"No HR images found in {hr_dir}")
                continue
            args_list = []
            for img_path in image_files:
                rel_path = os.path.relpath(img_path, hr_dir)
                if hr_dir != new_hr_dir:
                    hr_output_path = os.path.join(new_hr_dir, rel_path)
                    os.makedirs(os.path.dirname(hr_output_path), exist_ok=True)
                    if not os.path.exists(hr_output_path):
                        import shutil
                        shutil.copy(img_path, hr_output_path)
                lr_output_path = os.path.join(lr_dir, rel_path)
                args_list.append((img_path, lr_output_path))
            logging.info(f"Processing {len(args_list)} images from {hr_dir} to {lr_dir}...")
            with Pool(n_workers, initializer=init_worker, initargs=(scale_factor,)) as pool:
                list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply original Real-ESRGAN degradation to images')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing HR images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for degraded LR images')
    parser.add_argument('--scale', type=float, default=0.25, help='Scale factor for degradation (e.g., 0.25 for x4)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.scale, args.workers)