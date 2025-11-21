import cv2
import numpy as np

def apply_guided(image, radius=8, eps=0.04):
    return cv2.ximgproc.guidedFilter(image, image, radius, eps)
