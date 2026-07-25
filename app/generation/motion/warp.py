from __future__ import annotations

from PIL import Image

from app.generation.interfaces import FrameTransform


class AffineFrameWarper:
	"""Applies a FrameTransform (scale + translate) as a crop-and-resize warp.

	Output size always matches the source image size -- resizing to a model's
	target resolution (e.g. 1024x1024 for SDXL) is the renderer's concern, not
	the warper's.
	"""

	def warp(self, source_image: Image.Image, transform: FrameTransform) -> Image.Image:
		width, height = source_image.size
		crop_w, crop_h = width / transform.scale, height / transform.scale

		center_x = width / 2.0 + transform.translate_x
		center_y = height / 2.0 + transform.translate_y

		x1 = center_x - crop_w / 2.0
		y1 = center_y - crop_h / 2.0
		# Defensive clamp: keeps the crop box inside the source even if a caller
		# hands us a FrameTransform that wasn't produced by a face-safe planner.
		x1 = max(0.0, min(x1, width - crop_w)) if crop_w <= width else 0.0
		y1 = max(0.0, min(y1, height - crop_h)) if crop_h <= height else 0.0
		x2, y2 = x1 + crop_w, y1 + crop_h

		cropped = source_image.crop((x1, y1, x2, y2))
		return cropped.resize((width, height), Image.LANCZOS)
