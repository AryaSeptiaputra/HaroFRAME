from __future__ import annotations

from typing import Any

from PIL import Image

from app.core.config import ControlNetConfig
from app.identity.controlnet.depth_estimator import DepthEstimator
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import StructureHint


class DepthConditioner:
	"""ControlNet structure conditioner driven by a monocular depth map.

	Complements pose keypoints with 3D body/shape structure rather than just 2D joints.
	"""

	def __init__(self, config: ControlNetConfig) -> None:
		self._config = config
		self._depth_estimator = DepthEstimator(config.depth_estimator)
		self._controlnet = None

	def _ensure_controlnet(self):
		if self._controlnet is not None:
			return self._controlnet
		try:
			from diffusers import ControlNetModel
		except ImportError as exc:
			raise ModelLoadError("diffusers is not installed; cannot load the depth ControlNet") from exc
		self._controlnet = ControlNetModel.from_pretrained(self._config.depth_repo_id)
		return self._controlnet

	def preprocess(self, image: Image.Image) -> Image.Image:
		return self._depth_estimator.estimate(image)

	def build_control(self, hint: StructureHint) -> dict[str, Any]:
		if hint.depth_image is None:
			raise ModelLoadError("StructureHint.depth_image is required for depth conditioning")
		control_image = self.preprocess(hint.depth_image)
		return {
			"controlnet": self._ensure_controlnet(),
			"control_image": control_image,
			"controlnet_conditioning_scale": self._config.depth_conditioning_scale,
		}
