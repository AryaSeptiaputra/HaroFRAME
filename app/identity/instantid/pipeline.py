from __future__ import annotations

import torch

from app.core.config import IdentityConfig
from app.identity.exceptions import ModelLoadError
from app.identity.instantid.vendor.pipeline_stable_diffusion_xl_instantid import (
	StableDiffusionXLInstantIDPipeline,
)

_TORCH_DTYPES = {
	"fp16": torch.float16,
	"bf16": torch.bfloat16,
	"fp32": torch.float32,
}


def build_instantid_pipeline(config: IdentityConfig) -> StableDiffusionXLInstantIDPipeline:
	"""Construct the vendored InstantID pipeline with its IdentityNet ControlNet attached.

	This only wires the base SDXL model and IdentityNet ControlNet together; the
	face ip-adapter weights (ArcFace cross-attention) are attached separately via
	:meth:`app.identity.instantid.provider.InstantIdProvider.load`.
	"""
	try:
		from diffusers import ControlNetModel
	except ImportError as exc:
		raise ModelLoadError("diffusers is not installed; cannot build the InstantID pipeline") from exc

	dtype = _TORCH_DTYPES[config.dtype]
	hf_token = config.hf_token.get_secret_value() if config.hf_token else None

	controlnet = ControlNetModel.from_pretrained(
		config.instantid.controlnet_repo_id,
		subfolder=config.instantid.controlnet_subfolder,
		torch_dtype=dtype,
		cache_dir=config.cache_dir,
		token=hf_token,
	)
	pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
		config.base_sdxl_model,
		controlnet=controlnet,
		torch_dtype=dtype,
		cache_dir=config.cache_dir,
		token=hf_token,
	)
	pipeline.to(config.device, dtype)
	return pipeline
