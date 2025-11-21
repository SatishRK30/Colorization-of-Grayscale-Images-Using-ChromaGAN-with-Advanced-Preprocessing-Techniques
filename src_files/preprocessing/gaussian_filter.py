import cv2

def apply_gaussian(image, ksize=(5,5), sigma=1.0):
    return cv2.GaussianBlur(image, ksize, sigma)
