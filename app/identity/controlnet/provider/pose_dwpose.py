from __future__ import annotations

from typing import Any

from PIL import Image

from app.core.config import ControlNetConfig
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import StructureHint


class DwPoseConditioner:
	"""ControlNet structure conditioner driven by body pose keypoints.

	Prefers DWPose (richer hand/face keypoints); falls back to OpenPose if the
	DWPose ONNX weights can't be loaded in the current environment.
	"""

	def __init__(self, config: ControlNetConfig) -> None:
		self._config = config
		self._detector = None
		self._controlnet = None

	def _ensure_detector(self):
		if self._detector is not None:
			return self._detector
		try:
			from controlnet_aux import DWposeDetector

			self._detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
		except Exception:
			try:
				from controlnet_aux import OpenposeDetector

				self._detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
			except ImportError as exc:
				raise ModelLoadError(
					"controlnet_aux is not installed; cannot build a pose conditioner"
				) from exc
		return self._detector

	def _ensure_controlnet(self):
		if self._controlnet is not None:
			return self._controlnet
		try:
			from diffusers import ControlNetModel
		except ImportError as exc:
			raise ModelLoadError("diffusers is not installed; cannot load the pose ControlNet") from exc
		self._controlnet = ControlNetModel.from_pretrained(self._config.pose_repo_id)
		return self._controlnet

	def preprocess(self, image: Image.Image) -> Image.Image:
		detector = self._ensure_detector()
		return detector(image)

	def build_control(self, hint: StructureHint) -> dict[str, Any]:
		if hint.pose_image is None:
			raise ModelLoadError("StructureHint.pose_image is required for pose conditioning")
		control_image = self.preprocess(hint.pose_image)
		return {
			"controlnet": self._ensure_controlnet(),
			"control_image": control_image,
			"controlnet_conditioning_scale": self._config.pose_conditioning_scale,
		}
