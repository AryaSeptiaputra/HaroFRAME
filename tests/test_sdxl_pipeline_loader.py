from __future__ import annotations

from unittest.mock import MagicMock

import torch

from app.identity.sdxl_pipeline_loader import load_sdxl_pipeline, weight_variant_for


def test_weight_variant_for_maps_only_the_reduced_precision_dtypes():
	assert weight_variant_for(torch.float16) == "fp16"
	assert weight_variant_for(torch.bfloat16) == "bf16"
	assert weight_variant_for(torch.float32) is None


def test_load_sdxl_pipeline_uses_from_pretrained_for_repo_id(tmp_path):
	pipeline_cls = MagicMock()
	pipeline_cls.from_pretrained.return_value = "pretrained"

	result = load_sdxl_pipeline(
		pipeline_cls,
		"stabilityai/stable-diffusion-xl-base-1.0",
		torch_dtype="fp16-marker",
		cache_dir=tmp_path,
		token="tok",
	)

	assert result == "pretrained"
	# an unrecognised dtype marker maps to no variant, so this is the plain call
	pipeline_cls.from_pretrained.assert_called_once_with(
		"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype="fp16-marker", cache_dir=tmp_path, token="tok"
	)
	pipeline_cls.from_single_file.assert_not_called()


def test_load_sdxl_pipeline_requests_the_fp16_variant_for_float16(tmp_path):
	# Halves the download on repos shipping both precisions, and is the only way
	# to load repos that ship fp16 weights alone (e.g. Juggernaut XL v9).
	pipeline_cls = MagicMock()
	pipeline_cls.from_pretrained.return_value = "pretrained"

	result = load_sdxl_pipeline(
		pipeline_cls, "SG161222/RealVisXL_V5.0", torch_dtype=torch.float16, cache_dir=tmp_path, token="tok"
	)

	assert result == "pretrained"
	pipeline_cls.from_pretrained.assert_called_once_with(
		"SG161222/RealVisXL_V5.0", variant="fp16", torch_dtype=torch.float16, cache_dir=tmp_path, token="tok"
	)


def test_load_sdxl_pipeline_falls_back_when_the_repo_has_no_such_variant(tmp_path):
	pipeline_cls = MagicMock()
	pipeline_cls.from_pretrained.side_effect = [OSError("no fp16 variant"), "pretrained"]

	result = load_sdxl_pipeline(
		pipeline_cls, "some/repo", torch_dtype=torch.float16, cache_dir=tmp_path, token=None
	)

	assert result == "pretrained"
	assert pipeline_cls.from_pretrained.call_count == 2
	assert "variant" not in pipeline_cls.from_pretrained.call_args.kwargs


def test_load_sdxl_pipeline_uses_from_single_file_for_local_file(tmp_path):
	checkpoint_path = tmp_path / "model.safetensors"
	checkpoint_path.write_bytes(b"x")
	pipeline_cls = MagicMock()
	pipeline_cls.from_single_file.return_value = "single-file"

	result = load_sdxl_pipeline(pipeline_cls, str(checkpoint_path), torch_dtype="fp16-marker", cache_dir=tmp_path)

	assert result == "single-file"
	pipeline_cls.from_single_file.assert_called_once_with(str(checkpoint_path), torch_dtype="fp16-marker")
	pipeline_cls.from_pretrained.assert_not_called()


def test_load_sdxl_pipeline_passes_extra_kwargs_through_for_repo_id(tmp_path):
	pipeline_cls = MagicMock()
	controlnet_marker = object()

	load_sdxl_pipeline(
		pipeline_cls,
		"some/repo",
		torch_dtype="fp16-marker",
		cache_dir=tmp_path,
		token=None,
		controlnet=controlnet_marker,
	)

	pipeline_cls.from_pretrained.assert_called_once_with(
		"some/repo", torch_dtype="fp16-marker", cache_dir=tmp_path, token=None, controlnet=controlnet_marker
	)


def test_load_sdxl_pipeline_passes_extra_kwargs_through_for_single_file(tmp_path):
	checkpoint_path = tmp_path / "model.safetensors"
	checkpoint_path.write_bytes(b"x")
	pipeline_cls = MagicMock()
	controlnet_marker = object()

	load_sdxl_pipeline(
		pipeline_cls, str(checkpoint_path), torch_dtype="fp16-marker", cache_dir=tmp_path, controlnet=controlnet_marker
	)

	pipeline_cls.from_single_file.assert_called_once_with(
		str(checkpoint_path), torch_dtype="fp16-marker", controlnet=controlnet_marker
	)
