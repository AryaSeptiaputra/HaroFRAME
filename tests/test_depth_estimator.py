from __future__ import annotations

import pytest
from PIL import Image

from app.core.config import ControlNetConfig
from app.identity.controlnet.depth_estimator import DepthEstimator
from app.identity.controlnet.provider.depth import DepthConditioner
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import StructureHint


def test_depth_estimator_loads_backend_once_but_estimates_each_call(mocker):
	from_pretrained = mocker.patch(
		"controlnet_aux.MidasDetector.from_pretrained",
		return_value=mocker.Mock(return_value=Image.new("L", (4, 4))),
	)
	estimator = DepthEstimator(backend="midas")
	image = Image.new("RGB", (4, 4))

	estimator.estimate(image)
	estimator.estimate(image)

	from_pretrained.assert_called_once()


def test_depth_conditioner_preprocess_delegates_to_shared_estimator(mocker):
	fake_depth_image = Image.new("L", (4, 4))
	mock_estimate = mocker.patch(
		"app.identity.controlnet.depth_estimator.DepthEstimator.estimate", return_value=fake_depth_image
	)
	conditioner = DepthConditioner(ControlNetConfig())
	source = Image.new("RGB", (4, 4))

	result = conditioner.preprocess(source)

	assert result is fake_depth_image
	mock_estimate.assert_called_once_with(source)


def test_depth_conditioner_build_control_requires_depth_image():
	conditioner = DepthConditioner(ControlNetConfig())

	with pytest.raises(ModelLoadError):
		conditioner.build_control(StructureHint())
