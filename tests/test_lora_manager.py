from __future__ import annotations

from pathlib import Path

from app.core.config import LoraConfig, LoraEntryConfig
from app.generation.lora.manager import PeftLoraManager


class _FakePipeline:
	def __init__(self, existing_adapters=None):
		self.load_lora_weights_calls = []
		self.set_adapters_calls = []
		self.delete_adapters_calls = []
		self._existing_adapters = existing_adapters or {}

	def load_lora_weights(self, source, *, weight_name, subfolder, adapter_name):
		self.load_lora_weights_calls.append((source, weight_name, subfolder, adapter_name))

	def set_adapters(self, names, adapter_weights):
		self.set_adapters_calls.append((list(names), list(adapter_weights)))

	def get_list_adapters(self):
		return self._existing_adapters

	def delete_adapters(self, names):
		self.delete_adapters_calls.append(list(names))


def test_load_does_nothing_when_no_enabled_entries(tmp_path):
	manager = PeftLoraManager(LoraConfig(entries=[]), tmp_path)
	pipeline = _FakePipeline()

	manager.load(pipeline)

	assert pipeline.load_lora_weights_calls == []
	assert pipeline.set_adapters_calls == []


def test_load_skips_disabled_entries_and_resolves_each_source(tmp_path, mocker):
	local_file = tmp_path / "style.safetensors"
	local_file.write_bytes(b"x")
	entries = [
		LoraEntryConfig(adapter_name="a", source=str(local_file), scale=0.5, enabled=True),
		LoraEntryConfig(adapter_name="b", source="some/repo", scale=0.7, enabled=False),
	]
	manager = PeftLoraManager(LoraConfig(max_active_loras=3, entries=entries), tmp_path)
	pipeline = _FakePipeline()

	manager.load(pipeline)

	assert len(pipeline.load_lora_weights_calls) == 1
	source, weight_name, subfolder, adapter_name = pipeline.load_lora_weights_calls[0]
	assert source == str(local_file)
	assert adapter_name == "a"


def test_load_calls_set_adapters_with_names_and_weights():
	entries = [
		LoraEntryConfig(adapter_name="a", source="repo/a", scale=0.5),
		LoraEntryConfig(adapter_name="b", source="repo/b", scale=0.9),
	]
	manager = PeftLoraManager(LoraConfig(entries=entries), Path("/tmp"))
	pipeline = _FakePipeline()

	manager.load(pipeline)

	names, weights = pipeline.set_adapters_calls[0]
	assert names == ["a", "b"]
	assert weights == [0.5, 0.9]


def test_load_merges_existing_adapters_like_faceid_into_set_adapters_call():
	entries = [LoraEntryConfig(adapter_name="anime", source="repo/anime", scale=0.6)]
	manager = PeftLoraManager(LoraConfig(entries=entries), Path("/tmp"))
	pipeline = _FakePipeline(existing_adapters={"unet": ["faceid"]})

	manager.load(pipeline)

	names, weights = pipeline.set_adapters_calls[0]
	assert "faceid" in names
	assert "anime" in names
	assert weights[names.index("faceid")] == 1.0


def test_load_does_not_duplicate_already_tracked_adapter():
	entries = [LoraEntryConfig(adapter_name="anime", source="repo/anime", scale=0.6)]
	manager = PeftLoraManager(LoraConfig(entries=entries), Path("/tmp"))
	pipeline = _FakePipeline(existing_adapters={"unet": ["anime"]})

	manager.load(pipeline)

	names, _ = pipeline.set_adapters_calls[0]
	assert names.count("anime") == 1


def test_unload_deletes_tracked_adapters():
	entries = [LoraEntryConfig(adapter_name="anime", source="repo/anime")]
	manager = PeftLoraManager(LoraConfig(entries=entries), Path("/tmp"))
	pipeline = _FakePipeline()
	manager.load(pipeline)

	manager.unload(pipeline)

	assert pipeline.delete_adapters_calls == [["anime"]]
	assert manager.active_adapter_names(pipeline) == []


def test_unload_is_noop_when_nothing_loaded():
	manager = PeftLoraManager(LoraConfig(entries=[]), Path("/tmp"))
	pipeline = _FakePipeline()

	manager.unload(pipeline)

	assert pipeline.delete_adapters_calls == []


def test_active_adapter_names_returns_tracked_list():
	entries = [LoraEntryConfig(adapter_name="anime", source="repo/anime")]
	manager = PeftLoraManager(LoraConfig(entries=entries), Path("/tmp"))
	pipeline = _FakePipeline()

	manager.load(pipeline)

	assert manager.active_adapter_names(pipeline) == ["anime"]
