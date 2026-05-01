from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def apply_gaussian_noise(image: Image.Image, sigma: float = 18.0) -> Image.Image:
    arr = np.asarray(ensure_rgb(image)).astype(np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_motion_blur(image: Image.Image, radius: float = 2.5) -> Image.Image:
    return ensure_rgb(image).filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(image: Image.Image, quality: int = 22) -> Image.Image:
    buffer = BytesIO()
    ensure_rgb(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_pixelate(image: Image.Image, factor: int = 5) -> Image.Image:
    rgb = ensure_rgb(image)
    small_size = (max(1, rgb.width // factor), max(1, rgb.height // factor))
    small = rgb.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize(rgb.size, Image.Resampling.NEAREST)


def apply_low_contrast(image: Image.Image, factor: float = 0.45) -> Image.Image:
    return ImageEnhance.Contrast(ensure_rgb(image)).enhance(factor)


DEMO_CORRUPTIONS = {
    "None": lambda image: ensure_rgb(image),
    "Gaussian noise": apply_gaussian_noise,
    "Blur": apply_motion_blur,
    "JPEG compression": apply_jpeg_compression,
    "Pixelate": apply_pixelate,
    "Low contrast": apply_low_contrast,
}

EVAL_CORRUPTIONS = {
    "gaussian_noise": apply_gaussian_noise,
    "blur": apply_motion_blur,
    "jpeg_compression": apply_jpeg_compression,
    "pixelate": apply_pixelate,
    "low_contrast": apply_low_contrast,
}
