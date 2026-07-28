from __future__ import annotations

import numpy as np
import pytest

from app.identity.segmentation.interfaces import BodyKeypoints
from app.identity.segmentation.keypoint_prompts import garment_region_prompts


def _keypoints_all_present(image_size=(100, 200)) -> BodyKeypoints:
	# 18 evenly-spread normalized points, all confident, none at the sentinel.
	points = np.array([[0.5, i / 18] for i in range(18)], dtype=np.float32)
	scores = np.ones(18, dtype=np.float32)
	return BodyKeypoints(points=points, scores=scores, image_size=image_size)


def test_garment_region_prompts_rescales_to_target_size():
	keypoints = _keypoints_all_present()
	target_size = (100, 200)

	prompt_set = garment_region_prompts(keypoints, target_size, include_arms=False, include_legs=False)

	# Torso-only: neck, r_shoulder, l_shoulder, r_hip, l_hip -> 5 positive points.
	assert prompt_set.points.shape[0] == 5 + 5  # + 5 face keypoints as negatives
	assert (prompt_set.labels == 1).sum() == 5
	assert (prompt_set.labels == 0).sum() == 5
	# All points must be in pixel space of target_size, not normalized [0,1].
	assert prompt_set.points[:, 0].max() <= target_size[0]
	assert prompt_set.points[:, 1].max() <= target_size[1]


def test_garment_region_prompts_include_arms_and_legs_grows_positive_set():
	keypoints = _keypoints_all_present()
	target_size = (100, 200)

	torso_only = garment_region_prompts(keypoints, target_size, include_arms=False, include_legs=False)
	with_arms = garment_region_prompts(keypoints, target_size, include_arms=True, include_legs=False)
	with_arms_and_legs = garment_region_prompts(keypoints, target_size, include_arms=True, include_legs=True)

	assert (with_arms.labels == 1).sum() > (torso_only.labels == 1).sum()
	assert (with_arms_and_legs.labels == 1).sum() > (with_arms.labels == 1).sum()


def test_garment_region_prompts_filters_by_min_score():
	points = np.array([[0.5, i / 18] for i in range(18)], dtype=np.float32)
	scores = np.ones(18, dtype=np.float32)
	scores[2] = 0.1  # r_shoulder below threshold
	scores[5] = 0.1  # l_shoulder below threshold
	keypoints = BodyKeypoints(points=points, scores=scores, image_size=(100, 200))

	prompt_set = garment_region_prompts(
		keypoints, (100, 200), include_arms=False, include_legs=False, min_score=0.3
	)

	# Only neck, r_hip, l_hip remain confident enough (3 of the 5 torso points).
	assert (prompt_set.labels == 1).sum() == 3


def test_garment_region_prompts_box_padded_and_clamped():
	keypoints = _keypoints_all_present()

	prompt_set = garment_region_prompts(keypoints, (100, 200), include_arms=False, include_legs=False)

	assert prompt_set.box is not None
	x0, y0, x1, y1 = prompt_set.box
	assert 0.0 <= x0 <= x1 <= 100.0
	assert 0.0 <= y0 <= y1 <= 200.0


def test_garment_region_prompts_raises_with_too_few_keypoints():
	points = np.full((18, 2), -1.0, dtype=np.float32)
	scores = np.full(18, -1.0, dtype=np.float32)
	keypoints = BodyKeypoints(points=points, scores=scores, image_size=(100, 200))

	with pytest.raises(ValueError):
		garment_region_prompts(keypoints, (100, 200))
