from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import IpAdapterConfig
from app.identity.exceptions import ModelLoadError
from app.identity.ipadapter.provider.faceid_sdxl import FaceIdSdxlProvider


class _FakePipeline:
	def __init__(self):
		self.ip_adapter_calls = []
		self.lora_calls = []
		self.scales = []

	def load_ip_adapter(self, repo_id, **kwargs):
		self.ip_adapter_calls.append((repo_id, kwargs))

	def load_lora_weights(self, repo_id, **kwargs):
		self.lora_calls.append((repo_id, kwargs))

	def set_ip_adapter_scale(self, scale):
		self.scales.append(scale)


def test_load_skips_the_image_encoder():
	# FaceID conditions on the ArcFace vector, not pixels, and
	# h94/IP-Adapter-FaceID ships no image_encoder/ folder -- asking for one
	# downloads something that isn't there.
	pipeline = _FakePipeline()

	FaceIdSdxlProvider(IpAdapterConfig()).load(pipeline)

	_, kwargs = pipeline.ip_adapter_calls[0]
	assert kwargs["image_encoder_folder"] is None


def test_load_never_passes_a_none_subfolder():
	# Regression: diffusers builds the CLIP path as Path(subfolder,
	# image_encoder_folder), and Path(None, ...) raises
	# "argument should be a str or an os.PathLike object ... not 'NoneType'".
	pipeline = _FakePipeline()
	assert IpAdapterConfig().subfolder is None  # the default that used to crash

	FaceIdSdxlProvider(IpAdapterConfig()).load(pipeline)

	_, kwargs = pipeline.ip_adapter_calls[0]
	assert kwargs["subfolder"] is not None
	Path(kwargs["subfolder"], "image_encoder")  # must not raise


def test_load_keeps_an_explicit_subfolder():
	pipeline = _FakePipeline()

	FaceIdSdxlProvider(IpAdapterConfig(subfolder="sdxl_models")).load(pipeline)

	assert pipeline.ip_adapter_calls[0][1]["subfolder"] == "sdxl_models"


def test_load_attaches_the_companion_lora_under_the_reserved_name():
	pipeline = _FakePipeline()

	FaceIdSdxlProvider(IpAdapterConfig()).load(pipeline)

	repo_id, kwargs = pipeline.lora_calls[0]
	assert repo_id == "h94/IP-Adapter-FaceID"
	assert kwargs["adapter_name"] == "faceid"
	assert kwargs["weight_name"] == "ip-adapter-faceid_sdxl_lora.safetensors"


def test_load_skips_the_companion_lora_when_not_configured():
	pipeline = _FakePipeline()

	FaceIdSdxlProvider(IpAdapterConfig(lora_weight_name=None)).load(pipeline)

	assert pipeline.lora_calls == []


def test_load_applies_the_configured_scale():
	pipeline = _FakePipeline()

	FaceIdSdxlProvider(IpAdapterConfig(scale=0.85)).load(pipeline)

	assert pipeline.scales == [0.85]


def test_load_is_idempotent_on_one_provider_instance():
	pipeline = _FakePipeline()
	provider = FaceIdSdxlProvider(IpAdapterConfig())

	provider.load(pipeline)
	provider.load(pipeline)

	assert len(pipeline.ip_adapter_calls) == 1


def test_load_rejects_a_pipeline_without_ip_adapter_support():
	with pytest.raises(ModelLoadError):
		FaceIdSdxlProvider(IpAdapterConfig()).load(object())
