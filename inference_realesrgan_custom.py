import os
import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from realesrgan import RealESRGANer
import argparse

def preprocess_image(img):
    """Preprocess image for inference."""
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Resize to multiple of 4 for better upscaling
    h, w = img.shape[:2]
    h, w = h - h % 4, w - w % 4
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--netscale', type=int, default=4, help='Network upscale factor')
    parser.add_argument('--outscale', type=float, default=4, help='Output upscale factor')
    parser.add_argument('--suffix', type=str, default='sr', help='Suffix for output files')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN for face enhancement')
    args = parser.parse_args()

    # Initialize model
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=args.netscale
    )
    netscale = args.netscale
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=args.model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=0
    )
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    # Prepare input
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))])

    # Inference
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print(f'Processing {imgname}...')
        
        # Read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'Failed to read {path}')
            continue
        img = preprocess_image(img)
        
        # Inference
        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as e:
            print(f'Error processing {imgname}: {e}')
            continue
        
        # Save output
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.png')
        cv2.imwrite(save_path, output)
        print(f'Saved {save_path}')

if __name__ == '__main__':
    main()