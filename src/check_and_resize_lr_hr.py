import os
import cv2
import glob
from tqdm import tqdm
import argparse
import logging
import time

def setup_logging(log_file=None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter(log_format))
    return logger

def normalize_filename(filename):
    return filename.replace("_HR", "").replace("_LR", "")

def check_and_resize_lr_hr(dataset_root, output_root, scale_factor=0.5, min_size=360, max_size=1500, overwrite=False):
    logging.info(f"Starting processing with parameters:")
    logging.info(f"- Input directory: {dataset_root}")
    logging.info(f"- Output directory: {output_root}")
    logging.info(f"- Scale factor: {scale_factor}")
    logging.info(f"- Minimum size: {min_size}")
    logging.info(f"- Maximum size: {max_size}")
    logging.info(f"- Overwrite original: {overwrite}")

    stats = {'total_files_checked': 0, 'files_resized': 0, 'files_unchanged': 0, 'files_skipped': 0, 'processing_time': 0}
    start_time = time.time()
    scale = 1 / scale_factor

    for split in ['train', 'val', 'test']:
        hr_dir = os.path.join(dataset_root, split, 'HR')
        lr_dir = os.path.join(dataset_root, split, 'LR')

        if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
            logging.warning(f"Thư mục {hr_dir} hoặc {lr_dir} không tồn tại, bỏ qua")
            continue

        logging.info(f"Processing {split} dataset from {hr_dir} and {lr_dir}")

        hr_files = glob.glob(os.path.join(hr_dir, '*.[jp][pn][gf]'))
        lr_files = glob.glob(os.path.join(lr_dir, '*.[jp][pn][gf]'))

        hr_map = {normalize_filename(os.path.basename(f)): os.path.basename(f) for f in hr_files}
        lr_map = {normalize_filename(os.path.basename(f)): os.path.basename(f) for f in lr_files}

        common_keys = set(hr_map.keys()) & set(lr_map.keys())
        missing_in_lr = set(hr_map.keys()) - set(lr_map.keys())
        missing_in_hr = set(lr_map.keys()) - set(hr_map.keys())

        if missing_in_lr:
            logging.warning(f"Found {len(missing_in_lr)} files in HR but missing in LR: {list(missing_in_lr)[:5]}...")
        if missing_in_hr:
            logging.warning(f"Found {len(missing_in_hr)} files in LR but missing in HR: {list(missing_in_hr)[:5]}...")

        common_files = [(hr_map[k], lr_map[k]) for k in common_keys]
        if not common_files:
            logging.warning(f"Không tìm thấy cặp HR/LR nào trong {split}")
            continue

        output_hr_dir = os.path.join(output_root, split, 'HR')
        output_lr_dir = os.path.join(output_root, split, 'LR')
        os.makedirs(output_hr_dir, exist_ok=True)
        os.makedirs(output_lr_dir, exist_ok=True)

        logging.info(f"Created output directories: {output_hr_dir} and {output_lr_dir}")

        for hr_name, lr_name in tqdm(common_files, desc=f"Checking {split}"):
            hr_path = os.path.join(hr_dir, hr_name)
            lr_path = os.path.join(lr_dir, lr_name)
            output_hr_path = os.path.join(output_hr_dir, hr_name)
            output_lr_path = os.path.join(output_lr_dir, lr_name)

            stats['total_files_checked'] += 1

            hr_img = cv2.imread(hr_path)
            lr_img = cv2.imread(lr_path)
            if hr_img is None or lr_img is None:
                logging.warning(f"Bỏ qua {hr_name}: Không đọc được HR hoặc LR")
                stats['files_skipped'] += 1
                continue

            hr_h, hr_w = hr_img.shape[:2]
            lr_h, lr_w = lr_img.shape[:2]
            target_lr_h, target_lr_w = int(hr_h * scale_factor), int(hr_w * scale_factor)

            if hr_h < min_size or hr_w < min_size or target_lr_h < min_size or target_lr_w < min_size:
                logging.warning(f"Bỏ qua {hr_name}: Kích thước nhỏ hơn min_size {min_size} (HR: {hr_w}x{hr_h})")
                stats['files_skipped'] += 1
                continue

            if hr_h > max_size or hr_w > max_size:
                scale_resize = min(max_size / hr_h, max_size / hr_w)
                hr_img = cv2.resize(hr_img, (int(hr_w * scale_resize), int(hr_h * scale_resize)), interpolation=cv2.INTER_AREA)
                target_lr_h = int(hr_img.shape[0] * scale_factor)
                target_lr_w = int(hr_img.shape[1] * scale_factor)
                lr_img = cv2.resize(lr_img, (target_lr_w, target_lr_h), interpolation=cv2.INTER_AREA)

            scale_diff_h = abs(lr_img.shape[0] - target_lr_h)
            scale_diff_w = abs(lr_img.shape[1] - target_lr_w)

            if scale_diff_h <= 5 and scale_diff_w <= 5:
                cv2.imwrite(output_hr_path, hr_img)
                cv2.imwrite(output_lr_path, lr_img)
                stats['files_unchanged'] += 1
            else:
                lr_img = cv2.resize(lr_img, (target_lr_w, target_lr_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_hr_path, hr_img)
                cv2.imwrite(output_lr_path, lr_img)
                stats['files_resized'] += 1

                if overwrite:
                    cv2.imwrite(lr_path, lr_img)
                    logging.info(f"Overwritten original LR file: {lr_path}")

    stats['processing_time'] = time.time() - start_time
    logging.info("=" * 50)
    logging.info("Processing Summary:")
    logging.info(f"- Total files checked: {stats['total_files_checked']}")
    logging.info(f"- Files with correct scale (unchanged): {stats['files_unchanged']}")
    logging.info(f"- Files resized to correct scale: {stats['files_resized']}")
    logging.info(f"- Files skipped due to issues: {stats['files_skipped']}")
    logging.info(f"- Total processing time: {stats['processing_time']:.2f} seconds")
    logging.info("=" * 50)
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check and resize LR/HR pairs to correct scale')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing HR and LR images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed HR and LR images')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale factor (0.5 for x2, 0.25 for x4)')
    parser.add_argument('--min-size', type=int, default=360, help='Minimum size for HR/LR images')
    parser.add_argument('--max-size', type=int, default=1500, help='Maximum size for HR images')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite original LR images if resized')
    parser.add_argument('--log-file', type=str, help='Path to save log file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logging.info(f"Starting program at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Python version: {os.sys.version}")
    logging.info(f"OpenCV version: {cv2.__version__}")

    try:
        stats = check_and_resize_lr_hr(
            args.input,
            args.output,
            args.scale,
            args.min_size,
            args.max_size,
            args.overwrite
        )
        logging.info("Program completed successfully")
    except Exception as e:
        logging.exception(f"Program failed with error: {str(e)}")
