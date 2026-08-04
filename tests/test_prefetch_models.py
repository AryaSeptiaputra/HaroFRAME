from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.prefetch_models as prefetch_models  # noqa: E402
from app.core.config import IdentityConfig, LoraEntryConfig  # noqa: E402
from app.generation.source_resolver import ModelSourceError  # noqa: E402


def _identity(tmp_path, **overrides) -> IdentityConfig:
	return IdentityConfig(device="cpu", cache_dir=tmp_path, **overrides)


def test_prefetch_base_model_requests_the_fp16_variant_for_a_repo_id(mocker, tmp_path):
	# Must match load_sdxl_pipeline() exactly -- prefetching a different variant
	# is a cache miss, i.e. a silent second multi-GB download at generation time.
	download = mocker.patch("diffusers.DiffusionPipeline.download", return_value="/cache/realvis")

	step = prefetch_models.prefetch_base_model(_identity(tmp_path), None)

	assert step.ok
	download.assert_called_once_with("SG161222/RealVisXL_V5.0", variant="fp16", cache_dir=tmp_path, token=None)


def test_prefetch_base_model_falls_back_when_the_repo_has_no_fp16_variant(mocker, tmp_path):
	download = mocker.patch(
		"diffusers.DiffusionPipeline.download", side_effect=[OSError("no variant"), "/cache/plain"]
	)

	step = prefetch_models.prefetch_base_model(_identity(tmp_path), None)

	assert step.ok
	assert download.call_count == 2
	assert "variant" not in download.call_args.kwargs


def test_prefetch_base_model_uses_source_resolver_for_a_url(mocker, tmp_path):
	resolve = mocker.patch.object(prefetch_models, "resolve_model_source", return_value="/cache/model.safetensors")
	download = mocker.patch("diffusers.DiffusionPipeline.download")

	step = prefetch_models.prefetch_base_model(
		_identity(tmp_path, base_sdxl_model="https://example.com/model.safetensors"), "civitai-key"
	)

	assert step.ok
	resolve.assert_called_once_with(
		"https://example.com/model.safetensors", tmp_path, "civitai-key", subdir="checkpoints"
	)
	download.assert_not_called()


def test_prefetch_lora_downloads_a_named_weight_file_from_a_repo_id(mocker, tmp_path):
	hf_hub_download = mocker.patch("huggingface_hub.hf_hub_download", return_value="/cache/lora.safetensors")
	entry = LoraEntryConfig(adapter_name="detail", source="some/repo", weight_name="lora.safetensors")

	step = prefetch_models.prefetch_lora(entry, _identity(tmp_path), None)

	assert step.ok
	assert hf_hub_download.call_args.kwargs["repo_id"] == "some/repo"
	assert hf_hub_download.call_args.kwargs["filename"] == "lora.safetensors"


def test_prefetch_lora_snapshots_a_repo_id_without_a_named_weight(mocker, tmp_path):
	snapshot_download = mocker.patch("huggingface_hub.snapshot_download", return_value="/cache/repo")
	entry = LoraEntryConfig(adapter_name="detail", source="some/repo")

	step = prefetch_models.prefetch_lora(entry, _identity(tmp_path), None)

	assert step.ok
	assert snapshot_download.call_args.kwargs["repo_id"] == "some/repo"


def test_prefetch_lora_uses_source_resolver_for_a_civitai_url(mocker, tmp_path):
	resolve = mocker.patch.object(prefetch_models, "resolve_model_source", return_value="/cache/loras/x.safetensors")
	entry = LoraEntryConfig(adapter_name="detail", source="https://civitai.com/models/123")

	step = prefetch_models.prefetch_lora(entry, _identity(tmp_path), "key")

	assert step.ok
	resolve.assert_called_once_with("https://civitai.com/models/123", tmp_path, "key", subdir="loras")


def test_prefetch_sam_checkpoint_skips_an_existing_file(mocker, tmp_path):
	checkpoint = tmp_path / "sam.pth"
	checkpoint.write_bytes(b"x")
	urlretrieve = mocker.patch.object(prefetch_models.urllib.request, "urlretrieve")

	step = prefetch_models.prefetch_sam_checkpoint(checkpoint, "vit_b")

	assert step.ok
	urlretrieve.assert_not_called()


def test_prefetch_sam_checkpoint_downloads_via_a_partial_file(mocker, tmp_path):
	# An interrupted download must not leave a truncated file that later looks
	# "already present" to is_file().
	checkpoint = tmp_path / "nested" / "sam_vit_b.pth"

	def _fake_urlretrieve(url, destination):
		Path(destination).write_bytes(b"weights")

	urlretrieve = mocker.patch.object(
		prefetch_models.urllib.request, "urlretrieve", side_effect=_fake_urlretrieve
	)

	step = prefetch_models.prefetch_sam_checkpoint(checkpoint, "vit_b")

	assert step.ok
	assert checkpoint.read_bytes() == b"weights"
	assert not checkpoint.with_suffix(".pth.partial").exists()
	assert urlretrieve.call_args.args[0] == prefetch_models._SAM_URLS["vit_b"]
	assert str(urlretrieve.call_args.args[1]).endswith(".partial")


def test_prefetch_sam_checkpoint_leaves_no_file_behind_when_the_download_fails(mocker, tmp_path):
	checkpoint = tmp_path / "sam.pth"
	mocker.patch.object(prefetch_models.urllib.request, "urlretrieve", side_effect=OSError("connection reset"))

	with pytest.raises(OSError):
		prefetch_models.prefetch_sam_checkpoint(checkpoint, "vit_b")

	assert not checkpoint.exists()


def test_sam_urls_cover_every_configurable_model_type():
	from app.core.config import SamConfig

	model_types = SamConfig.model_fields["model_type"].annotation.__args__
	assert set(model_types) == set(prefetch_models._SAM_URLS)


def test_sam_urls_agree_with_the_configs_default_filenames():
	# config owns the default path, prefetch owns the URL -- if they drift, a
	# machine downloads one backbone's weights under another's filename.
	from app.core.config import SAM_CHECKPOINT_FILENAMES

	for model_type, url in prefetch_models._SAM_URLS.items():
		assert url.rsplit("/", 1)[-1] == SAM_CHECKPOINT_FILENAMES[model_type]


@pytest.mark.parametrize("model_type", ["vit_b", "vit_l", "vit_h"])
def test_sam_default_checkpoint_path_follows_model_type(model_type):
	from app.core.config import SAM_CHECKPOINT_FILENAMES, SamConfig

	config = SamConfig(model_type=model_type)

	assert config.checkpoint_path.name == SAM_CHECKPOINT_FILENAMES[model_type]


def test_sam_explicit_checkpoint_path_wins_over_model_type(tmp_path):
	from app.core.config import SamConfig

	config = SamConfig(model_type="vit_h", checkpoint_path=tmp_path / "my-sam.pth")

	assert config.checkpoint_path == tmp_path / "my-sam.pth"


def _run_main(mocker, argv, *, settings):
	mocker.patch.object(prefetch_models, "get_settings", return_value=settings)
	mocker.patch.object(sys, "argv", ["prefetch_models.py", *argv])
	return prefetch_models.main()


def test_main_reports_failure_without_stopping_the_remaining_steps(mocker, tmp_path):
	from app.core.config import GenerationConfig, InpaintConfig, LoraConfig, Settings

	settings = Settings(
		identity=_identity(tmp_path),
		generation=GenerationConfig(
			lora=LoraConfig(entries=[LoraEntryConfig(adapter_name="detail", source="some/repo")]),
			inpaint=InpaintConfig(enabled=False),
		),
	)
	mocker.patch.object(
		prefetch_models, "prefetch_base_model", side_effect=ModelSourceError("404 from the hub")
	)
	lora_step = prefetch_models._Step("lora:detail", True, "ok")
	prefetch_lora = mocker.patch.object(prefetch_models, "prefetch_lora", return_value=lora_step)

	exit_code = _run_main(mocker, [], settings=settings)

	assert exit_code == 1
	prefetch_lora.assert_called_once()


def test_main_skips_sam_when_inpainting_is_disabled(mocker, tmp_path):
	from app.core.config import GenerationConfig, InpaintConfig, Settings

	settings = Settings(
		identity=_identity(tmp_path),
		generation=GenerationConfig(inpaint=InpaintConfig(enabled=False)),
	)
	mocker.patch.object(
		prefetch_models, "prefetch_base_model", return_value=prefetch_models._Step("base model", True, "ok")
	)
	prefetch_sam = mocker.patch.object(prefetch_models, "prefetch_sam_checkpoint")

	assert _run_main(mocker, [], settings=settings) == 0
	prefetch_sam.assert_not_called()


def test_main_prefetches_sam_when_inpainting_is_enabled(mocker, tmp_path):
	from app.core.config import GenerationConfig, InpaintConfig, Settings

	settings = Settings(
		identity=_identity(tmp_path),
		generation=GenerationConfig(inpaint=InpaintConfig(prompt="a red hoodie")),
	)
	mocker.patch.object(
		prefetch_models, "prefetch_base_model", return_value=prefetch_models._Step("base model", True, "ok")
	)
	prefetch_sam = mocker.patch.object(
		prefetch_models, "prefetch_sam_checkpoint", return_value=prefetch_models._Step("sam", True, "ok")
	)

	assert _run_main(mocker, [], settings=settings) == 0
	prefetch_sam.assert_called_once()


def test_main_honours_the_skip_flags(mocker, tmp_path):
	from app.core.config import GenerationConfig, InpaintConfig, Settings

	settings = Settings(
		identity=_identity(tmp_path),
		generation=GenerationConfig(inpaint=InpaintConfig(prompt="a red hoodie")),
	)
	prefetch_base = mocker.patch.object(prefetch_models, "prefetch_base_model")
	prefetch_sam = mocker.patch.object(prefetch_models, "prefetch_sam_checkpoint")

	assert _run_main(mocker, ["--skip-base", "--skip-loras", "--skip-sam"], settings=settings) == 0
	prefetch_base.assert_not_called()
	prefetch_sam.assert_not_called()
