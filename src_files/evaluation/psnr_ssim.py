import cv2
import math
from skimage.metrics import structural_similarity

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0: 
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(img1, img2):
    return structural_similarity(img1, img2)
