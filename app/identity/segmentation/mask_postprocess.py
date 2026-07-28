from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def dilate_mask(mask: Image.Image, dilation_px: int) -> Image.Image:
	"""Grow a binary mask outward by dilation_px so previously garment-covered
	skin (e.g. an arm under a removed sleeve) falls inside the inpaint region
	rather than being left as the original garment pixels."""
	if dilation_px <= 0:
		return mask
	array = np.array(mask.convert("L"), dtype=np.uint8)
	kernel_size = 2 * dilation_px + 1
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	dilated = cv2.dilate(array, kernel)
	return Image.fromarray(dilated, mode="L")


def feather_mask(mask: Image.Image, feather_px: int) -> Image.Image:
	"""Gaussian-blur the mask edge so the inpaint boundary blends without a hard seam."""
	if feather_px <= 0:
		return mask
	array = np.array(mask.convert("L"), dtype=np.uint8)
	kernel_size = 2 * feather_px + 1
	blurred = cv2.GaussianBlur(array, (kernel_size, kernel_size), 0)
	return Image.fromarray(blurred, mode="L")
