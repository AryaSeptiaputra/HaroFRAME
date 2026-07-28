from __future__ import annotations

from unittest.mock import MagicMock

from app.identity.sdxl_pipeline_loader import load_sdxl_pipeline


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
	pipeline_cls.from_pretrained.assert_called_once_with(
		"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype="fp16-marker", cache_dir=tmp_path, token="tok"
	)
	pipeline_cls.from_single_file.assert_not_called()


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
