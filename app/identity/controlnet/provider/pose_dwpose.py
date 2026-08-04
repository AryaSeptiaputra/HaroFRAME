from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.core.config import ControlNetConfig
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import StructureHint
from app.identity.segmentation.interfaces import BodyKeypoints


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
			except (ImportError, AttributeError) as exc:
				# AttributeError, not just ImportError: controlnet_aux imports its
				# mediapipe_face submodule eagerly and only tolerates mediapipe being
				# *absent* (it warns and degrades). A mediapipe that is installed but
				# broken -- as on some vast.ai template images -- gets past that guard
				# and dies on `mp.solutions`, taking the whole package import with it.
				raise ModelLoadError(
					f"could not import a pose detector from controlnet_aux: {exc}. "
					"If that mentions mediapipe, it is installed but broken in this "
					"environment; nothing here uses mediapipe, so `pip uninstall -y "
					"mediapipe` is the fix. Otherwise install controlnet_aux."
				) from exc
		return self._detector

	def release(self) -> None:
		"""Drop the cached detector and ControlNet so their VRAM comes back.

		Both are rebuilt lazily on next use, so this is safe to call whenever the
		caller knows it is done with pose conditioning for now.
		"""
		self._detector = None
		self._controlnet = None

	def ensure_controlnet(self):
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

	def detect_body_keypoints(self, image: Image.Image) -> BodyKeypoints:
		"""Return normalized 18-point body keypoints for the most confident detected
		person, used by SAM prompt derivation for the stage-1 garment/body mask
		(app/identity/segmentation/) -- unrelated to preprocess()/build_control()
		above, which stay on the detector's normal rendered-skeleton path.

		Dispatches on whichever pose backend _ensure_detector() actually resolved
		to, rather than assuming true DWPose loaded: DWPose's Wholebody hard-
		requires mmcv/mmpose/mmdet, none of which are installed or declared in
		this project's dependencies, so _ensure_detector()'s except-Exception
		fallback to OpenposeDetector is what is actually active. OpenposeDetector
		exposes detect_poses() (18-point body, already pre-normalized to [0,1]);
		if the OpenMMLab stack is ever added, DWposeDetector exposes
		pose_estimation() (candidate/subset arrays, pixel coords, first 18 rows
		of the 133-point COCO-WholeBody layout are the same 18 body points).
		"""
		detector = self._ensure_detector()
		width, height = image.size
		rgb = np.array(image.convert("RGB"), dtype=np.uint8)

		if hasattr(detector, "detect_poses"):
			poses = detector.detect_poses(rgb)
			if not poses:
				raise ModelLoadError("no person detected for garment/body mask keypoint extraction")
			best = max(poses, key=lambda pose: pose.body.total_score)
			points = np.full((18, 2), -1.0, dtype=np.float32)
			scores = np.full((18,), -1.0, dtype=np.float32)
			for index, keypoint in enumerate(best.body.keypoints[:18]):
				if keypoint is not None:
					points[index] = (keypoint.x, keypoint.y)
					scores[index] = keypoint.score
			return BodyKeypoints(points=points, scores=scores, image_size=(width, height))

		if hasattr(detector, "pose_estimation"):
			candidate, subset = detector.pose_estimation(rgb[:, :, ::-1])
			if candidate.shape[0] == 0:
				raise ModelLoadError("no person detected for garment/body mask keypoint extraction")
			best_index = int(np.argmax(np.clip(subset[:, :18], 0, None).sum(axis=1)))
			points = candidate[best_index, :18, :2].astype(np.float32).copy()
			points[:, 0] /= float(width)
			points[:, 1] /= float(height)
			scores = subset[best_index, :18].astype(np.float32)
			return BodyKeypoints(points=points, scores=scores, image_size=(width, height))

		raise ModelLoadError("active pose backend exposes no raw-keypoint accessor")

	def build_control(self, hint: StructureHint) -> dict[str, Any]:
		if hint.pose_image is None:
			raise ModelLoadError("StructureHint.pose_image is required for pose conditioning")
		control_image = self.preprocess(hint.pose_image)
		return {
			"controlnet": self.ensure_controlnet(),
			"control_image": control_image,
			"controlnet_conditioning_scale": self._config.pose_conditioning_scale,
		}
