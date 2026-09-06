# corruptions.py (کتابخانه‌ی توابع corruption)

import numpy as np
import cv2
from PIL import Image
from io import BytesIO

def apply_blur(img, severity):
    kernel = {1: 5, 2: 15}[severity]
    arr = np.array(img)
    arr = cv2.GaussianBlur(arr, (kernel, kernel), 0)
    return Image.fromarray(arr)

def apply_jpeg(img, severity):
    quality = {1: 30, 2: 5}[severity]
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_noise(img, severity):
    std = {1: 15, 2: 40}[severity]
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_resize(img, severity):
    scale = {1: 0.4, 2: 0.15}[severity]
    w, h = img.size
    small = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def apply_crop(img, severity):
    frac = {1: 0.7, 2: 0.4}[severity]
    w, h = img.size
    new_w, new_h = int(w * frac**0.5), int(h * frac**0.5)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)

def apply_lowlight(img, severity):
    factor = {1: 0.4, 2: 0.15}[severity]
    arr = np.array(img).astype(np.float32) * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_fog(img, severity):
    alpha = {1: 0.4, 2: 0.7}[severity]
    arr = np.array(img).astype(np.float32)
    white = np.ones_like(arr) * 255
    arr = arr * (1 - alpha) + white * alpha
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


CORRUPTIONS = {
    "blur": apply_blur,
    "jpeg": apply_jpeg,
    "noise": apply_noise,
    "resize": apply_resize,
    "crop": apply_crop,
    "lowlight": apply_lowlight,
    "fog": apply_fog,
}
SEVERITIES = [1, 2]