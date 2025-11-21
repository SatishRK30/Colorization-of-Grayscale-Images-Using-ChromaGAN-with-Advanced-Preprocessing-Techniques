from .gaussian_filter import apply_gaussian
from .bilateral_filter import apply_bilateral
from .clahe_filter import apply_clahe
from .guided_filter import apply_guided

def process_all(image):
    g = apply_gaussian(image)
    b = apply_bilateral(g)
    return b
