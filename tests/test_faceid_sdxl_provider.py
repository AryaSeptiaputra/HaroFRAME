from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.core.config import IpAdapterConfig
from app.identity.exceptions import ModelLoadError, NoFaceDetectedError
from app.identity.interfaces import FaceEmbedding, IdentityReference
from app.identity.ipadapter.provider.faceid_sdxl import FaceIdSdxlProvider


class _FakePipeline:
	def __init__(self, dtype=torch.float16):
		self.ip_adapter_calls = []
		self.lora_calls = []
		self.scales = []
		self.dtype = dtype

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


# --- build_conditioning: the shape contract diffusers actually enforces --------


def _reference(vector=None):
	reference = IdentityReference(images=[])
	reference.fused_embedding = FaceEmbedding(
		vector=np.arange(512, dtype=np.float32) if vector is None else vector,
		det_score=0.9,
		bbox=(0.0, 0.0, 1.0, 1.0),
	)
	return reference


def _embeds(provider):
	return provider.build_conditioning(_reference()).adapter_kwargs["ip_adapter_image_embeds"][0]


def test_conditioning_embeds_are_batch_num_images_embed_dim():
	# MultiIPAdapterImageProjection documents [batch_size, num_images, embed_dim].
	embeds = _embeds(FaceIdSdxlProvider(IpAdapterConfig()))

	assert embeds.shape == (2, 1, 512)


def test_conditioning_embeds_satisfy_check_inputs_rank():
	# Regression: an unsqueeze gave (1, 512) and diffusers refused it with
	# "`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is 2D".
	embeds = _embeds(FaceIdSdxlProvider(IpAdapterConfig()))

	assert embeds.ndim in (3, 4)


def test_conditioning_embeds_carry_a_zeroed_negative_half_first():
	# prepare_ip_adapter_image_embeds() does single_image_embeds.chunk(2) under
	# classifier-free guidance and takes the first half as the negative.
	vector = np.arange(512, dtype=np.float32)

	embeds = FaceIdSdxlProvider(IpAdapterConfig()).build_conditioning(
		_reference(vector)
	).adapter_kwargs["ip_adapter_image_embeds"][0]
	negative, positive = embeds.chunk(2)

	assert torch.count_nonzero(negative) == 0
	assert torch.allclose(positive.reshape(-1), torch.from_numpy(vector))


def test_conditioning_embeds_take_the_pipelines_dtype_after_load():
	# diffusers moves supplied embeds to the pipeline device but never casts them.
	provider = FaceIdSdxlProvider(IpAdapterConfig())
	provider.load(_FakePipeline(dtype=torch.float16))

	assert _embeds(provider).dtype == torch.float16


def test_conditioning_embeds_stay_float32_before_any_load():
	assert _embeds(FaceIdSdxlProvider(IpAdapterConfig())).dtype == torch.float32


def test_build_conditioning_without_any_embedding_raises():
	provider = FaceIdSdxlProvider(IpAdapterConfig())

	with pytest.raises(NoFaceDetectedError):
		provider.build_conditioning(IdentityReference(images=[]))
