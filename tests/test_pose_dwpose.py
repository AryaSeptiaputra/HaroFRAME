from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core.config import ControlNetConfig
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.exceptions import ModelLoadError


class _FakeOpenposeDetector:
	def __init__(self, poses):
		self._poses = poses

	def detect_poses(self, rgb_image):
		return self._poses


class _FakePoseEstimationDetector:
	def __init__(self, candidate, subset):
		self._candidate = candidate
		self._subset = subset

	def pose_estimation(self, bgr_image):
		return self._candidate, self._subset


def _pose_result(xy_or_none, total_score=1.0):
	keypoints = [SimpleNamespace(x=x, y=y, score=1.0) if (x, y) != (None, None) else None for x, y in xy_or_none]
	body = SimpleNamespace(keypoints=keypoints, total_score=total_score, total_parts=len(keypoints))
	return SimpleNamespace(body=body)


def test_detect_body_keypoints_via_detect_poses_backend():
	xy = [(0.5, i / 18.0) for i in range(18)]
	detector = _FakeOpenposeDetector([_pose_result(xy)])
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = detector

	result = conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))

	assert result.points.shape == (18, 2)
	assert result.image_size == (100, 200)
	assert np.allclose(result.points[0], (0.5, 0.0))
	assert (result.scores >= 0).all()


def test_detect_body_keypoints_marks_missing_points_with_sentinel():
	xy = [(0.5, i / 18.0) for i in range(18)]
	xy[3] = (None, None)
	detector = _FakeOpenposeDetector([_pose_result(xy)])
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = detector

	result = conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))

	assert tuple(result.points[3]) == (-1.0, -1.0)
	assert result.scores[3] == -1.0


def test_detect_body_keypoints_picks_highest_scoring_person():
	xy = [(0.5, i / 18.0) for i in range(18)]
	low = _pose_result(xy, total_score=1.0)
	high = _pose_result([(0.9, y) for _, y in xy], total_score=5.0)
	detector = _FakeOpenposeDetector([low, high])
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = detector

	result = conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))

	assert np.allclose(result.points[0], (0.9, 0.0))


def test_detect_body_keypoints_raises_when_no_person_detected_detect_poses():
	detector = _FakeOpenposeDetector([])
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = detector

	with pytest.raises(ModelLoadError):
		conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))


def test_detect_body_keypoints_via_pose_estimation_backend_rescales_to_normalized():
	candidate = np.zeros((1, 18, 4), dtype=np.float32)
	candidate[0, :, 0] = 50.0  # x = 50px
	candidate[0, :, 1] = 100.0  # y = 100px
	subset = np.ones((1, 18), dtype=np.float32)
	detector = _FakePoseEstimationDetector(candidate, subset)
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = detector

	result = conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))

	assert np.allclose(result.points[0], (0.5, 0.5))


def test_detect_body_keypoints_raises_when_no_person_detected_pose_estimation():
	candidate = np.zeros((0, 18, 4), dtype=np.float32)
	subset = np.zeros((0, 18), dtype=np.float32)
	detector = _FakePoseEstimationDetector(candidate, subset)
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = detector

	with pytest.raises(ModelLoadError):
		conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))


def test_ensure_detector_reports_a_broken_controlnet_aux_import_as_model_load_error(mocker):
	# A half-installed mediapipe makes `import controlnet_aux` raise AttributeError
	# rather than ImportError, which used to escape as a raw traceback.
	import builtins

	real_import = builtins.__import__

	def _explode(name, *args, **kwargs):
		if name == "controlnet_aux":
			raise AttributeError("module 'mediapipe' has no attribute 'solutions'")
		return real_import(name, *args, **kwargs)

	mocker.patch.object(builtins, "__import__", side_effect=_explode)
	conditioner = DwPoseConditioner(ControlNetConfig())

	with pytest.raises(ModelLoadError, match="mediapipe"):
		conditioner._ensure_detector()


def test_detect_body_keypoints_raises_when_backend_exposes_neither_accessor():
	conditioner = DwPoseConditioner(ControlNetConfig())
	conditioner._detector = object()

	with pytest.raises(ModelLoadError):
		conditioner.detect_body_keypoints(Image.new("RGB", (100, 200)))
