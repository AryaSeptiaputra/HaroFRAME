from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_VARIANT_BY_DTYPE = {
	torch.float16: "fp16",
	torch.bfloat16: "bf16",
}


def weight_variant_for(torch_dtype: Any) -> str | None:
	"""Which Hugging Face weight *variant* matches ``torch_dtype``, or None for
	the repo's default (usually fp32) weights.

	Public because scripts/prefetch_models.py has to warm exactly the files
	load_sdxl_pipeline() will later ask for -- prefetching the wrong variant is
	a cache miss, i.e. a silent second multi-GB download at generation time.
	"""
	return _VARIANT_BY_DTYPE.get(torch_dtype)


def load_sdxl_pipeline(
	pipeline_cls: Any,
	base_model: str,
	*,
	torch_dtype: Any,
	cache_dir: Path,
	token: str | None = None,
	**extra_kwargs: Any,
) -> Any:
	"""Build an SDXL pipeline from either a Hugging Face repo_id / local diffusers
	directory (``from_pretrained``) or a single-file checkpoint
	(``from_single_file``) -- e.g. a `.safetensors` downloaded from Civitai or a
	direct link via app.generation.source_resolver.resolve_model_source.

	Shared by the IP-Adapter img2img renderer and the InstantID pipeline builder
	so both branches support single-file checkpoints identically.

	For repo_ids it asks for the ``variant`` matching ``torch_dtype`` first. Many
	SDXL community checkpoints ship both fp32 and fp16 weights (RealVisXL V5.0:
	~14GB vs ~7GB) and some ship *only* fp16 (Juggernaut XL v9) -- without the
	variant the first kind wastes half the download to cast down to fp16 anyway,
	and the second kind fails outright. Repos that carry only default-named
	weights raise on the variant, so that case falls back rather than failing.
	"""
	if Path(base_model).is_file():
		return pipeline_cls.from_single_file(str(base_model), torch_dtype=torch_dtype, **extra_kwargs)

	kwargs: dict[str, Any] = dict(torch_dtype=torch_dtype, cache_dir=cache_dir, token=token, **extra_kwargs)
	variant = weight_variant_for(torch_dtype)
	if variant is not None:
		try:
			return pipeline_cls.from_pretrained(base_model, variant=variant, **kwargs)
		except (OSError, ValueError):
			pass
	return pipeline_cls.from_pretrained(base_model, **kwargs)
