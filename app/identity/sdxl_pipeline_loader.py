from __future__ import annotations

from pathlib import Path
from typing import Any


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
	"""
	if Path(base_model).is_file():
		return pipeline_cls.from_single_file(str(base_model), torch_dtype=torch_dtype, **extra_kwargs)
	return pipeline_cls.from_pretrained(
		base_model, torch_dtype=torch_dtype, cache_dir=cache_dir, token=token, **extra_kwargs
	)
