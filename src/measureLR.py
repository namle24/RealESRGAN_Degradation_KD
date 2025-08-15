import cv2
import numpy as np
from skimage.restoration import estimate_sigma


def measure_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Variance of Laplacian (Sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Tenengrad (Gradient Energy)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    tenengrad = np.sqrt(sobel_x ** 2 + sobel_y ** 2).mean()

    return laplacian_var, tenengrad


def measure_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Estimate Gaussian Noise sigma
    sigma_est = estimate_sigma(gray, multichannel=False, average_sigmas=True)

    return sigma_est


# Example usage
image = cv2.imread('D:\\EnhanceVideo_ImageDLM\\data\\RealSR(V3)\\Nikon\\Train\\4\\Nikon_001_LR4.png')

laplacian_var, tenengrad = measure_blur(image)
sigma_noise = measure_noise(image)

print(f"Variance of Laplacian (Blur Sharpness): {laplacian_var:.2f}")
print(f"Tenengrad (Edge Strength): {tenengrad:.2f}")
print(f"Estimated Noise Sigma: {sigma_noise:.2f}")
