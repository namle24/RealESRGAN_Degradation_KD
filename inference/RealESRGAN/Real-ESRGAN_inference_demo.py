import argparse
import cv2
import glob
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import subprocess
from realesrgan import RealESRGANer

def calculate_psnr(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
    img1_y = img1[:, :, 0].astype(np.float64)  # Kênh Y
    img2_y = img2[:, :, 0].astype(np.float64)  # Kênh Y
    return psnr(img1_y, img2_y, data_range=255)

def calculate_ssim(img1, img2):
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray, img2_gray = img1, img2
    return ssim(img1_gray, img2_gray, data_range=255)

def calculate_vmaf(ref_path, dist_path, model_path="/content/Real-ESRGAN/vmaf_v0.6.1.json"):
    cmd = [
        "ffmpeg", "-i", dist_path, "-i", ref_path,
        "-lavfi", f"[0:v][1:v]libvmaf=model=path={model_path}:log_path=vmaf_log.json:log_fmt=json",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"VMAF calculation failed: {result.stderr.decode()}")
        return None
    try:
        with open("vmaf_log.json", "r") as f:
            import json
            return json.load(f)["pooled_metrics"]["vmaf"]["mean"]
    except Exception as e:
        print(f"Error reading VMAF log: {e}")
        return None

def find_gt_image(gt_dir, imgname, extensions=['.png', '.jpg', '.jpeg']):
    # Thử các tên file GT có thể có
    base_name = imgname.replace('_LR4', '')  # Loại bỏ '_LR4' nếu có
    for ext in extensions:
        # Thử tên giống LR
        gt_path = os.path.join(gt_dir, f"{imgname}{ext}")
        if os.path.exists(gt_path):
            return gt_path
        # Thử tên với '_HR' thay vì '_LR4'
        gt_path = os.path.join(gt_dir, f"{base_name}_HR{ext}")
        if os.path.exists(gt_path):
            return gt_path
        # Thử tên gốc không suffix
        gt_path = os.path.join(gt_dir, f"{base_name}{ext}")
        if os.path.exists(gt_path):
            return gt_path
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/content/Real-ESRGAN/upload/LR', help='Input image or folder (LR)')
    parser.add_argument('-gt', '--ground_truth', type=str, default='/content/Real-ESRGAN/upload/GT', help='Ground truth folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('-o', '--output', type=str, default='/content/Real-ESRGAN/results/LR', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension')
    args = parser.parse_args()

    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = load_file_from_url(url=file_url[0], model_dir=os.path.join(ROOT_DIR, 'weights'))

    upsampler = RealESRGANer(
        scale=netscale, model_path=model_path, model=model,
        tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, half=not args.fp32
    )

    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler
        )

    vmaf_model_path = "/content/Real-ESRGAN/vmaf_v0.6.1.json"
    if not os.path.exists(vmaf_model_path):
        os.system("wget https://github.com/Netflix/vmaf/raw/master/model/vmaf_v0.6.1.json -P /content/Real-ESRGAN/")

    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    paths = [args.input] if os.path.isfile(args.input) else sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print(f'Testing {idx} {imgname}')

        # Đọc ảnh LR
        img_lr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_lr is None:
            print(f"Cannot read LR image: {path}")
            continue

        # Tìm ảnh GT linh hoạt
        gt_path = find_gt_image(args.ground_truth, imgname)
        if gt_path:
            img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        else:
            print(f"Cannot find GT image for {imgname} in {args.ground_truth}")
            continue

        # Cải thiện ảnh LR
        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img_lr, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img_lr, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('Try reducing --tile size.')
            continue

        # Lưu ảnh cải thiện
        output_path = os.path.join(args.output, f"{imgname}_{args.suffix}.png")
        print(f"Saving enhanced image to: {output_path}")
        cv2.imwrite(output_path, output)

        # Đọc ảnh cải thiện
        output_img = cv2.imread(output_path, cv2.IMREAD_COLOR)

        # Resize về kích thước GT để so sánh
        output_img_resized = cv2.resize(output_img, (img_gt.shape[1], img_gt.shape[0]))

        # Tính metrics so với GT
        psnr_value = calculate_psnr(img_gt, output_img_resized)
        ssim_value = calculate_ssim(img_gt, output_img_resized)

        resized_output_path = os.path.join(args.output, f"{imgname}_{args.suffix}_resized.png")
        print(f"Saving resized image to: {resized_output_path}")
        cv2.imwrite(resized_output_path, output_img_resized)
        vmaf_value = calculate_vmaf(gt_path, resized_output_path)

        print(f"Metrics for {os.path.basename(path)} (compared to GT):")
        print(f"  PSNR: {psnr_value:.2f} dB")
        print(f"  SSIM: {ssim_value:.4f}")
        if vmaf_value is not None:
            print(f"  VMAF: {vmaf_value:.2f}")

        with open(os.path.join(args.output, "metrics.txt"), "a") as f:
            f.write(f"Metrics for {os.path.basename(path)} (compared to GT):\n")
            f.write(f"  PSNR: {psnr_value:.2f} dB\n")
            f.write(f"  SSIM: {ssim_value:.4f}\n")
            if vmaf_value is not None:
                f.write(f"  VMAF: {vmaf_value:.2f}\n")

        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{args.ext if args.ext != "auto" else extension[1:]}')
        print(f"Saving final image to: {save_path}")
        cv2.imwrite(save_path, output)

if __name__ == '__main__':
    main()