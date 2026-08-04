from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from app.core.config import InpaintConfig, SamConfig
from app.identity.exceptions import ModelLoadError
from app.identity.segmentation.interfaces import BodyKeypoints
from app.identity.segmentation.sam_provider import SamGarmentMaskGenerator


class _FakePoseConditioner:
	def __init__(self, keypoints):
		self._keypoints = keypoints
		self.received_images = []

	def detect_body_keypoints(self, image):
		self.received_images.append(image)
		return self._keypoints


def _keypoints():
	points = np.array([[0.5, i / 18.0] for i in range(18)], dtype=np.float32)
	scores = np.ones(18, dtype=np.float32)
	return BodyKeypoints(points=points, scores=scores, image_size=(64, 64))


def _install_fake_sam_module(mocker, *, predict_return):
	fake_predictor_instance = mocker.Mock()
	fake_predictor_instance.predict = mocker.Mock(return_value=predict_return)

	fake_sam_model = mocker.Mock()
	fake_sam_model.to = mocker.Mock(return_value=fake_sam_model)

	fake_module = types.ModuleType("segment_anything")
	fake_module.sam_model_registry = {"vit_b": mocker.Mock(return_value=fake_sam_model)}
	fake_module.SamPredictor = mocker.Mock(return_value=fake_predictor_instance)
	mocker.patch.dict(sys.modules, {"segment_anything": fake_module})
	return fake_predictor_instance, fake_module


def test_generate_mask_raises_when_segment_anything_not_installed():
	config = InpaintConfig()
	generator = SamGarmentMaskGenerator(
		config, device="cpu", pose_conditioner=_FakePoseConditioner(_keypoints())
	)

	with pytest.raises(ModelLoadError, match="segment-anything is not installed"):
		generator.generate_mask(Image.new("RGB", (64, 64)))


def test_generate_mask_raises_when_checkpoint_missing(mocker, tmp_path):
	_install_fake_sam_module(mocker, predict_return=(np.zeros((1, 64, 64), dtype=bool), np.array([0.9]), None))
	config = InpaintConfig(sam=SamConfig(checkpoint_path=tmp_path / "missing.pth"))
	generator = SamGarmentMaskGenerator(
		config, device="cpu", pose_conditioner=_FakePoseConditioner(_keypoints())
	)

	with pytest.raises(ModelLoadError, match="checkpoint not found"):
		generator.generate_mask(Image.new("RGB", (64, 64)))


def test_generate_mask_wires_prompts_and_postprocesses_best_mask(mocker, tmp_path):
	checkpoint_path = tmp_path / "sam.pth"
	checkpoint_path.write_bytes(b"x")

	raw_masks = np.zeros((3, 64, 64), dtype=bool)
	raw_masks[1, 20:40, 20:40] = True  # best-scoring mask is index 1
	scores = np.array([0.1, 0.9, 0.5])
	predictor, _ = _install_fake_sam_module(mocker, predict_return=(raw_masks, scores, None))

	config = InpaintConfig(sam=SamConfig(checkpoint_path=checkpoint_path), mask_dilation_px=0, mask_feather_px=0)
	pose_conditioner = _FakePoseConditioner(_keypoints())
	generator = SamGarmentMaskGenerator(config, device="cpu", pose_conditioner=pose_conditioner)

	image = Image.new("RGB", (64, 64))
	result = generator.generate_mask(image)

	assert pose_conditioner.received_images == [image]
	call_kwargs = predictor.predict.call_args.kwargs
	assert call_kwargs["point_coords"].shape[1] == 2
	assert call_kwargs["point_labels"].ndim == 1
	assert call_kwargs["box"] is not None
	assert call_kwargs["multimask_output"] is True

	assert result.mask.mode == "L"
	assert result.mask.size == (64, 64)
	mask_array = np.array(result.mask)
	assert mask_array[30, 30] == 255  # inside the best-scoring mask region
	assert mask_array[0, 0] == 0  # outside it


def test_generate_mask_reuses_predictor_across_calls(mocker, tmp_path):
	checkpoint_path = tmp_path / "sam.pth"
	checkpoint_path.write_bytes(b"x")
	raw_masks = np.zeros((1, 64, 64), dtype=bool)
	raw_masks[0, 10:20, 10:20] = True
	_, fake_module = _install_fake_sam_module(mocker, predict_return=(raw_masks, np.array([0.9]), None))

	config = InpaintConfig(sam=SamConfig(checkpoint_path=checkpoint_path))
	generator = SamGarmentMaskGenerator(
		config, device="cpu", pose_conditioner=_FakePoseConditioner(_keypoints())
	)

	generator.generate_mask(Image.new("RGB", (64, 64)))
	generator.generate_mask(Image.new("RGB", (64, 64)))

	fake_module.SamPredictor.assert_called_once()
