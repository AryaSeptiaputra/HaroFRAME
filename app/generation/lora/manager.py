from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import LoraConfig
from app.generation.source_resolver import resolve_model_source


class PeftLoraManager:
	"""Loads and activates a stack of named LoRA adapters on an SDXL pipeline via
	diffusers' PEFT multi-adapter support (``load_lora_weights(adapter_name=...)``
	followed by one combined ``set_adapters()`` call).

	Always folds in whatever adapters are already attached to the pipeline (e.g.
	the reserved ``"faceid"`` companion LoRA loaded separately by
	FaceIdSdxlProvider) into that combined call -- diffusers only guarantees the
	weight of adapters explicitly named in ``set_adapters()``, so leaving an
	existing adapter out of the call risks silently resetting/disabling it.
	"""

	def __init__(self, config: LoraConfig, cache_dir: Path) -> None:
		self._config = config
		self._cache_dir = cache_dir
		self._loaded_adapter_names: list[str] = []

	def load(self, pipeline: Any) -> None:
		enabled_entries = [entry for entry in self._config.entries if entry.enabled]
		if not enabled_entries:
			return

		civitai_api_key = (
			self._config.civitai_api_key.get_secret_value() if self._config.civitai_api_key else None
		)
		names: list[str] = []
		weights: list[float] = []
		for entry in enabled_entries:
			resolved_source = resolve_model_source(entry.source, self._cache_dir, civitai_api_key, subdir="loras")
			pipeline.load_lora_weights(
				resolved_source,
				weight_name=entry.weight_name,
				subfolder=entry.subfolder,
				adapter_name=entry.adapter_name,
			)
			names.append(entry.adapter_name)
			weights.append(entry.scale)

		self._loaded_adapter_names = names
		all_names, all_weights = self._merge_with_existing_adapters(pipeline, names, weights)
		pipeline.set_adapters(all_names, adapter_weights=all_weights)

	def _merge_with_existing_adapters(
		self, pipeline: Any, names: list[str], weights: list[float]
	) -> tuple[list[str], list[float]]:
		merged_names = list(names)
		merged_weights = list(weights)
		for name in self._existing_adapter_names(pipeline):
			if name not in merged_names:
				merged_names.append(name)
				merged_weights.append(1.0)
		return merged_names, merged_weights

	@staticmethod
	def _existing_adapter_names(pipeline: Any) -> list[str]:
		# Best-effort: get_list_adapters() maps component name (e.g. "unet") to
		# the list of adapter names attached to it. If unavailable (older
		# diffusers, or nothing attached yet) there's simply nothing to merge in
		# besides the adapters this call itself just loaded.
		try:
			per_component = pipeline.get_list_adapters()
		except Exception:
			return []
		names: set[str] = set()
		for component_names in per_component.values():
			names.update(component_names)
		return sorted(names)

	def unload(self, pipeline: Any) -> None:
		if not self._loaded_adapter_names:
			return
		try:
			pipeline.delete_adapters(self._loaded_adapter_names)
		except Exception:
			pass
		self._loaded_adapter_names = []

	def active_adapter_names(self, pipeline: Any) -> list[str]:
		return list(self._loaded_adapter_names)
