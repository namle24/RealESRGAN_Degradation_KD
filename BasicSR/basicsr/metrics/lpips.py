import torch
import lpips
import numpy as np

def calculate_lpips(img1, img2, net_type='vgg', normalize=True):
    """Calculate LPIPS between two images.
    
    Args:
        img1 (ndarray): Image with range [0, 255], shape (H, W, C).
        img2 (ndarray): Image with range [0, 255], shape (H, W, C).
        net_type (str): Network type for LPIPS ('vgg', 'alex'). Default: 'vgg'.
        normalize (bool): Whether to normalize images to [-1, 1]. Default: True.
    
    Returns:
        float: LPIPS score.
    """
    # Convert to tensor
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net=net_type)
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
    
    # Normalize to [-1, 1] if required
    if normalize:
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_score = lpips_model(img1, img2).item()
    
    return lpips_score