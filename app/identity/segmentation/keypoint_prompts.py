from __future__ import annotations

import numpy as np

from app.identity.segmentation.interfaces import BodyKeypoints, SamPromptSet

NOSE, NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, \
	R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE, R_EYE, L_EYE, R_EAR, L_EAR = range(18)

_TORSO_INDICES = (NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP)
_ARM_INDICES = (R_ELBOW, R_WRIST, L_ELBOW, L_WRIST)
_LEG_INDICES = (R_KNEE, R_ANKLE, L_KNEE, L_ANKLE)
_FACE_INDICES = (NOSE, R_EYE, L_EYE, R_EAR, L_EAR)


def _valid_points_px(
	keypoints: BodyKeypoints, indices: tuple[int, ...], target_size: tuple[int, int], min_score: float
) -> np.ndarray:
	width, height = target_size
	scale = np.array([width, height], dtype=np.float32)
	selected = [
		keypoints.points[i] * scale
		for i in indices
		if keypoints.scores[i] >= min_score and keypoints.points[i][0] >= 0 and keypoints.points[i][1] >= 0
	]
	return np.array(selected, dtype=np.float32) if selected else np.empty((0, 2), dtype=np.float32)


def garment_region_prompts(
	keypoints: BodyKeypoints,
	target_size: tuple[int, int],
	*,
	include_arms: bool = True,
	include_legs: bool = False,
	box_padding_ratio: float = 0.15,
	min_score: float = 0.3,
) -> SamPromptSet:
	"""Turn body keypoints into SAM point/box prompts covering the garment-bearing
	region of the body: torso always, arms/legs per the include_* flags.

	Positive points bias SAM toward including that body region; negative points
	(face keypoints) bias it away from including the head/hair, since face-region
	identity is preserved separately (via IdentityEngine.build_conditioning(), not
	the inpaint mask). The box is a padded bounding box over the positive points,
	giving SAM a coarse region hint that improves mask contiguity for baggy
	garments (e.g. a winter coat) where sparse points alone under-segment.

	Raises ValueError if fewer than 2 valid positive keypoints are found.
	"""
	positive_indices = _TORSO_INDICES
	if include_arms:
		positive_indices += _ARM_INDICES
	if include_legs:
		positive_indices += _LEG_INDICES

	positive_points = _valid_points_px(keypoints, positive_indices, target_size, min_score)
	if positive_points.shape[0] < 2:
		raise ValueError(
			f"only {positive_points.shape[0]} valid body keypoint(s) found (need at least 2) "
			"to derive a garment/body mask region"
		)
	negative_points = _valid_points_px(keypoints, _FACE_INDICES, target_size, min_score)

	points = np.concatenate([positive_points, negative_points], axis=0)
	labels = np.concatenate(
		[np.ones(len(positive_points), dtype=np.int64), np.zeros(len(negative_points), dtype=np.int64)]
	)

	x0, y0 = positive_points.min(axis=0)
	x1, y1 = positive_points.max(axis=0)
	pad = box_padding_ratio * float(np.hypot(x1 - x0, y1 - y0))
	width, height = target_size
	box = (
		max(0.0, x0 - pad),
		max(0.0, y0 - pad),
		min(float(width), x1 + pad),
		min(float(height), y1 + pad),
	)

	return SamPromptSet(points=points, labels=labels, box=box)
