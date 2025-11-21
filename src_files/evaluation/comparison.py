from .psnr_ssim import compute_psnr, compute_ssim

def evaluate(original, generated):
    return {
        "psnr": compute_psnr(original, generated),
        "ssim": compute_ssim(original, generated)
    }
