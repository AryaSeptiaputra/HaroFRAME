"""Download every model weight the configured pipeline will need, up front.

Called by entrypoint.sh at instance setup so the first generate() doesn't stall
for tens of GB mid-run. Everything here is driven by config -- this script picks
nothing on its own, it just warms whatever Settings already says will be used:

  1. the base SDXL checkpoint (IdentityConfig.base_sdxl_model)
  2. every enabled LoRA (GenerationConfig.lora.entries)
  3. the SAM checkpoint for stage-1 inpainting, when that stage is enabled

Each step is independent: one failure is reported and the rest still run, the
same way scripts/test_real_images.py treats a batch. Exit code is 1 if anything
failed, so entrypoint.sh can warn without aborting the instance.

Idempotent by construction -- huggingface_hub and resolve_model_source() both
no-op on an already-cached file, so re-running on every instance restart is
cheap.

Usage:
    python scripts/prefetch_models.py            # everything the config implies
    python scripts/prefetch_models.py --skip-sam # base model + LoRAs only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import IdentityConfig, LoraEntryConfig, get_settings
from app.identity.sdxl_pipeline_loader import weight_variant_for
from app.generation.source_resolver import ModelSourceError, resolve_model_source

_TORCH_DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}

# Canonical Meta-hosted SAM checkpoints, keyed by SamConfig.model_type.
_SAM_URLS = {
	"vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
	"vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
	"vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


@dataclass
class _Step:
	name: str
	ok: bool
	detail: str


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--skip-base", action="store_true", help="don't prefetch the base SDXL checkpoint")
	parser.add_argument("--skip-loras", action="store_true", help="don't prefetch configured LoRAs")
	parser.add_argument(
		"--skip-sam", action="store_true", help="don't prefetch the SAM checkpoint even if inpainting is enabled"
	)
	return parser.parse_args()


def _is_remote_or_local_file(source: str) -> bool:
	"""True when resolve_model_source() will hand back a concrete file path
	rather than passing a Hugging Face repo_id straight through."""
	return urlparse(source).scheme in ("http", "https") or Path(source).exists()


def _torch_dtype(identity_config: IdentityConfig):
	import torch

	return getattr(torch, _TORCH_DTYPES[identity_config.dtype])


def prefetch_base_model(identity_config: IdentityConfig, civitai_api_key: str | None) -> _Step:
	"""Warm the base checkpoint. A URL/Civitai/local source is a single file that
	load_sdxl_pipeline() will open with from_single_file(); a repo_id is a
	diffusers layout, downloaded through DiffusionPipeline.download() so the file
	selection (and crucially the fp16 variant) matches what from_pretrained()
	will ask for later."""
	source = identity_config.base_sdxl_model
	if _is_remote_or_local_file(source):
		resolved = resolve_model_source(source, identity_config.cache_dir, civitai_api_key, subdir="checkpoints")
		return _Step("base model", True, f"single-file checkpoint at {resolved}")

	from diffusers import DiffusionPipeline

	token = identity_config.hf_token.get_secret_value() if identity_config.hf_token else None
	variant = weight_variant_for(_torch_dtype(identity_config))
	kwargs = dict(cache_dir=identity_config.cache_dir, token=token)
	if variant is not None:
		try:
			path = DiffusionPipeline.download(source, variant=variant, **kwargs)
			return _Step("base model", True, f"{source} ({variant} variant) -> {path}")
		except (OSError, ValueError):
			# No such variant in this repo -- load_sdxl_pipeline() falls back the
			# same way, so warm the default weights instead.
			pass
	path = DiffusionPipeline.download(source, **kwargs)
	return _Step("base model", True, f"{source} (default weights) -> {path}")


def prefetch_lora(entry: LoraEntryConfig, identity_config: IdentityConfig, civitai_api_key: str | None) -> _Step:
	name = f"lora:{entry.adapter_name}"
	if _is_remote_or_local_file(entry.source):
		resolved = resolve_model_source(entry.source, identity_config.cache_dir, civitai_api_key, subdir="loras")
		return _Step(name, True, str(resolved))

	token = identity_config.hf_token.get_secret_value() if identity_config.hf_token else None
	if entry.weight_name:
		from huggingface_hub import hf_hub_download

		path = hf_hub_download(
			repo_id=entry.source,
			filename=entry.weight_name,
			subfolder=entry.subfolder,
			cache_dir=identity_config.cache_dir,
			token=token,
		)
	else:
		from huggingface_hub import snapshot_download

		path = snapshot_download(repo_id=entry.source, cache_dir=identity_config.cache_dir, token=token)
	return _Step(name, True, f"{entry.source} -> {path}")


def prefetch_sam_checkpoint(checkpoint_path: Path, model_type: str) -> _Step:
	if checkpoint_path.is_file():
		return _Step("sam checkpoint", True, f"already present at {checkpoint_path}")
	url = _SAM_URLS[model_type]
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	# Download beside the target and rename, so an interrupted run can't leave a
	# truncated file that later looks "already present" to is_file().
	partial_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
	print(f"  downloading {url} ...")
	urllib.request.urlretrieve(url, partial_path)  # noqa: S310 -- fixed https URL from _SAM_URLS
	partial_path.replace(checkpoint_path)
	return _Step("sam checkpoint", True, f"{model_type} -> {checkpoint_path}")


def main() -> int:
	args = _parse_args()
	settings = get_settings()
	identity_config = settings.identity
	generation_config = settings.generation
	civitai_api_key = (
		generation_config.lora.civitai_api_key.get_secret_value()
		if generation_config.lora.civitai_api_key
		else None
	)

	steps: list[_Step] = []

	if not args.skip_base:
		print(f"--- base model: {identity_config.base_sdxl_model} ---")
		try:
			steps.append(prefetch_base_model(identity_config, civitai_api_key))
		except (ModelSourceError, OSError, ValueError) as exc:
			steps.append(_Step("base model", False, str(exc)))

	if not args.skip_loras:
		enabled_loras = [entry for entry in generation_config.lora.entries if entry.enabled]
		print(f"--- loras: {len(enabled_loras)} enabled ---")
		for entry in enabled_loras:
			try:
				steps.append(prefetch_lora(entry, identity_config, civitai_api_key))
			except (ModelSourceError, OSError, ValueError) as exc:
				steps.append(_Step(f"lora:{entry.adapter_name}", False, str(exc)))

	inpaint = generation_config.inpaint
	if not args.skip_sam and inpaint.enabled:
		print(f"--- sam checkpoint: {inpaint.sam.model_type} ---")
		try:
			steps.append(prefetch_sam_checkpoint(inpaint.sam.checkpoint_path, inpaint.sam.model_type))
		except (OSError, ValueError) as exc:
			steps.append(_Step("sam checkpoint", False, str(exc)))

	print("\n=== prefetch summary ===")
	if not steps:
		print("(nothing to prefetch)")
		return 0
	for step in steps:
		print(f"{'OK  ' if step.ok else 'FAIL'} {step.name}: {step.detail}")
	failed = [step for step in steps if not step.ok]
	print(f"{len(steps) - len(failed)}/{len(steps)} succeeded")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
