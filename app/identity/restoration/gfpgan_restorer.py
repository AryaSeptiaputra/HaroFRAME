from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.config import RestorationConfig
from app.identity.exceptions import ModelLoadError


class GfpganRestorer:
	"""FaceRestorer backed by GFPGAN.

	basicsr's sdist build is currently broken on Python 3.13 (setup.py version
	parsing bug: ``KeyError '__version__'``) and it ships no wheel on PyPI, so the
	gfpgan/basicsr/facexlib imports are deferred to first use and guarded here:
	constructing this class stays cheap, and calling :meth:`restore` raises a clear
	:class:`ModelLoadError` until a fixed release or fork is available.
	"""

	def __init__(self, config: RestorationConfig) -> None:
		self._config = config
		self._restorer = None

	def _ensure_loaded(self):
		if self._restorer is not None:
			return self._restorer
		try:
			from gfpgan import GFPGANer
		except ImportError as exc:
			raise ModelLoadError(
				"gfpgan (and its basicsr/facexlib dependencies) are not installed; install "
				"this project's 'restoration' extra to use GfpganRestorer. Note: basicsr "
				"currently has no working build on Python 3.13 -- see pyproject.toml."
			) from exc
		if not self._config.model_path.exists():
			raise ModelLoadError(f"GFPGAN model weights not found at {self._config.model_path}")
		self._restorer = GFPGANer(
			model_path=str(self._config.model_path),
			upscale=self._config.upscale,
			arch="clean",
			channel_multiplier=2,
		)
		return self._restorer

	def restore(self, image: Image.Image) -> Image.Image:
		restorer = self._ensure_loaded()
		image_bgr = np.array(image.convert("RGB"))[:, :, ::-1]
		_, _, restored_bgr = restorer.enhance(
			image_bgr, has_aligned=False, only_center_face=False, paste_back=True
		)
		return Image.fromarray(restored_bgr[:, :, ::-1])
