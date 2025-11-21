"""
Utility functions for loading/saving and color conversions.

Report PDF reference:
/mnt/data/Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques (1).pdf
"""

import cv2
import numpy as np
import os

REPORT_PDF_PATH = r"Report/Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques (1).pdf"

def load_image_grayscale(path, target_size=(224,224)):
    """
    Loads image in grayscale (single channel) and resizes.
    Returns uint8 image shape (H,W).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found: " + path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def save_bgr_image(path, bgr_image):
    """
    Save BGR uint8 image to path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr_image)

def bgr_to_lab_scaled(bgr):
    """
    Convert BGR uint8 [0-255] → LAB with L in [0,100], a,b in [-128,127]
    Returns float32
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype('float32')
    L = lab[..., 0:1]  # 0..255 scaled in OpenCV to 0..255 then to 0..100 we can scale
    a = lab[..., 1:2] - 128.0
    b = lab[..., 2:3] - 128.0
    # scale L from [0,255] to [0,100]
    L = (L / 255.0) * 100.0
    return np.concatenate([L, a, b], axis=-1)

def lab_scaled_to_bgr(L_channel, ab_channels):
    """
    L_channel: float32 L in [0,100] or [0,1]*100
    ab_channels: float32 in range approximately [-128,127]
    Returns BGR uint8 [0-255]
    """
    L = L_channel.copy()
    ab = ab_channels.copy()
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype='float32')
    lab[...,0] = (L[...,0] / 100.0) * 255.0
    lab[...,1] = ab[...,0] + 128.0
    lab[...,2] = ab[...,1] + 128.0
    lab = lab.astype('uint8')
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def preprocess_L_from_gray(gray_uint8, target_size=(224,224)):
    """
    Input: gray_uint8 shape (H,W) 0..255
    Output: L channel float32 shape (H,W,1) scaled to [0,100]
    """
    gray = cv2.resize(gray_uint8, target_size, interpolation=cv2.INTER_AREA)
    L = gray.astype('float32')
    L = (L / 255.0) * 100.0
    return L[..., None]

def postprocess_ab_output(pred_ab):
    """
    pred_ab: model output in tanh range [-1,1] expected shape (H,W,2)
    Scale to ab range roughly [-128,127]
    """
    return (pred_ab * 128.0).astype('float32')
