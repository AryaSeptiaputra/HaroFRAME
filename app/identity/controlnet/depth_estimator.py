from __future__ import annotations

from typing import Literal

from PIL import Image

from app.identity.exceptions import ModelLoadError


class DepthEstimator:
	"""Lazily-loaded monocular depth estimator (Midas or Zoe).

	Shared by :class:`~app.identity.controlnet.provider.depth.DepthConditioner`
	(the ControlNet-conditioning half) and by depth-based motion planning
	(``app/generation/motion/parallax_warp.py``) -- kept independent of both
	call sites so neither has to duplicate model loading or depend on the other.
	"""

	def __init__(self, backend: Literal["midas", "zoe"] = "midas") -> None:
		self._backend = backend
		self._estimator = None

	def _ensure_loaded(self):
		if self._estimator is not None:
			return self._estimator
		try:
			if self._backend == "midas":
				from controlnet_aux import MidasDetector

				self._estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")
			else:
				from transformers import pipeline as hf_pipeline

				depth_pipe = hf_pipeline("depth-estimation", model="Intel/zoedepth-nyu-kitti")
				self._estimator = lambda image: depth_pipe(image)["depth"]
		except ImportError as exc:
			raise ModelLoadError("no depth estimation backend is installed") from exc
		return self._estimator

	def estimate(self, image: Image.Image) -> Image.Image:
		estimator = self._ensure_loaded()
		return estimator(image)
