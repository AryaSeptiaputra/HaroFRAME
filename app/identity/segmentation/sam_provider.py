from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.core.config import ControlNetConfig, InpaintConfig
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.exceptions import ModelLoadError
from app.identity.segmentation.interfaces import GarmentMask
from app.identity.segmentation.keypoint_prompts import garment_region_prompts
from app.identity.segmentation.mask_postprocess import dilate_mask, feather_mask


class SamGarmentMaskGenerator:
	"""GarmentMaskGenerator backed by SAM (segment-anything).

	Lazily-loaded, mirroring InsightFaceAnalyzer's/DepthEstimator's
	lazy-load-on-first-use pattern: construction stays cheap when the stage-1
	inpaint editor is never used, and SAM model weights load only on the first
	generate_mask() call. Body-region prompts are derived from DWPose/OpenPose
	keypoints (see
	app.identity.controlnet.provider.pose_dwpose.detect_body_keypoints), via a
	DwPoseConditioner instance owned by this class -- deliberately independent
	of IdentityConfig.controlnet's pose_enabled toggle, which governs a
	different feature (img2img structure conditioning).
	"""

	def __init__(
		self,
		config: InpaintConfig,
		*,
		device: str,
		pose_conditioner: DwPoseConditioner | None = None,
	) -> None:
		self._config = config
		self._device = device
		self._pose_conditioner = pose_conditioner or DwPoseConditioner(ControlNetConfig())
		self._predictor: Any = None

	def _ensure_predictor(self) -> Any:
		if self._predictor is not None:
			return self._predictor
		try:
			from segment_anything import SamPredictor, sam_model_registry
		except ImportError as exc:
			raise ModelLoadError(
				"segment-anything is not installed; install the optional 'garment' "
				'extra (pip install -e ".[garment]") and download a SAM checkpoint'
			) from exc
		checkpoint_path = self._config.sam.checkpoint_path
		if not checkpoint_path.is_file():
			raise ModelLoadError(
				f"SAM checkpoint not found at {checkpoint_path}; download one and set "
				"HAROFRAME_GENERATION__INPAINT__SAM__CHECKPOINT_PATH (see VAST_GUIDE.md)"
			)
		sam = sam_model_registry[self._config.sam.model_type](checkpoint=str(checkpoint_path))
		sam.to(device=self._device)
		self._predictor = SamPredictor(sam)
		return self._predictor

	def generate_mask(self, image: Image.Image) -> GarmentMask:
		keypoints = self._pose_conditioner.detect_body_keypoints(image)
		prompt_set = garment_region_prompts(
			keypoints,
			image.size,
			include_arms=self._config.include_arms_in_mask,
			include_legs=self._config.include_legs_in_mask,
			min_score=self._config.mask_min_confidence,
		)
		predictor = self._ensure_predictor()
		predictor.set_image(np.array(image.convert("RGB")))
		masks, scores, _ = predictor.predict(
			point_coords=prompt_set.points,
			point_labels=prompt_set.labels,
			box=np.array(prompt_set.box) if prompt_set.box is not None else None,
			multimask_output=True,
		)
		best_mask = masks[int(np.argmax(scores))]
		mask_image = Image.fromarray(best_mask.astype(np.uint8) * 255, mode="L")
		mask_image = dilate_mask(mask_image, self._config.mask_dilation_px)
		mask_image = feather_mask(mask_image, self._config.mask_feather_px)
		return GarmentMask(mask=mask_image, prompt_set=prompt_set)
