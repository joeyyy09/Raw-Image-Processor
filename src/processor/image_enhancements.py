import cv2
import numpy as np

def apply_adaptive_noise_reduction(image):
    """Apply noise reduction based on image characteristics"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_level = np.std(gray)
    h = int(noise_level * 2)
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

def apply_smart_sharpening(image):
    """Apply content-aware sharpening"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    strength = min(1.5, max(0.5, 1.0 / (laplacian_var + 1e-5)))
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]]) * strength
    return cv2.filter2D(image, -1, kernel)