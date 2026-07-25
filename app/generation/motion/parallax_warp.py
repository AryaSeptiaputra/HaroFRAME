from __future__ import annotations

import numpy as np
from PIL import Image

from app.identity.controlnet.depth_estimator import DepthEstimator
from app.identity.exceptions import ModelLoadError
from app.generation.interfaces import FrameTransform
from app.generation.motion.warp import AffineFrameWarper


class DepthParallaxWarper:
	"""FrameWarper that adds pseudo-3D parallax on top of the same crop/pan/zoom
	trajectory a plain Ken Burns pass would use: pixels are displaced according
	to estimated scene depth (nearer content shifts more than farther content)
	before the usual crop+resize, instead of shifting every pixel uniformly.

	Depth is estimated once per source image and cached -- it doesn't change
	frame to frame, only the requested camera transform does.
	"""

	def __init__(self, depth_estimator: DepthEstimator | None = None, parallax_strength: float = 0.05) -> None:
		self._depth_estimator = depth_estimator or DepthEstimator()
		self._parallax_strength = parallax_strength
		self._depth_map: np.ndarray | None = None
		self._depth_source_id: int | None = None
		self._affine_warper = AffineFrameWarper()

	def _ensure_depth_map(self, source_image: Image.Image) -> np.ndarray:
		if self._depth_map is not None and self._depth_source_id == id(source_image):
			return self._depth_map
		depth_image = self._depth_estimator.estimate(source_image).convert("L").resize(source_image.size)
		depth = np.asarray(depth_image, dtype=np.float32) / 255.0
		self._depth_map = depth
		self._depth_source_id = id(source_image)
		return depth

	def warp(self, source_image: Image.Image, transform: FrameTransform) -> Image.Image:
		try:
			import cv2
		except ImportError as exc:
			raise ModelLoadError("opencv-python is not installed; cannot warp with depth parallax") from exc

		width, height = source_image.size
		depth = self._ensure_depth_map(source_image)

		# Displacement scales with how far this frame's transform has already
		# panned -- reuses the same translate_x/translate_y a plain 2D pan would
		# use, just applied per-pixel (weighted by depth) instead of uniformly.
		shift_x = transform.translate_x * self._parallax_strength * 2
		shift_y = transform.translate_y * self._parallax_strength * 2
		# MiDaS-style output is inverse depth (higher == nearer), so subtracting
		# the mean makes near content shift one way and far content the other.
		parallax_factor = depth - depth.mean()

		x_coords, y_coords = np.meshgrid(
			np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
		)
		map_x = x_coords + parallax_factor * shift_x
		map_y = y_coords + parallax_factor * shift_y

		source_array = np.asarray(source_image.convert("RGB"))
		displaced = cv2.remap(source_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
		displaced_image = Image.fromarray(displaced)

		# Apply the transform's usual crop/zoom on top of the now-parallax-shifted image.
		return self._affine_warper.warp(displaced_image, transform)
