from __future__ import annotations

import pytest

from pydantic import ValidationError

from app.core.config import GenerationConfig, IdentityConfig, LoraConfig, LoraEntryConfig, Settings, get_settings


def test_identity_config_defaults():
	config = IdentityConfig()

	assert config.device == "cuda"
	assert config.dtype == "fp16"
	assert config.face.fusion_strategy == "mean"
	assert config.ipadapter.enabled is False
	assert config.instantid.enabled is False
	assert config.controlnet.pose_enabled is False
	assert config.controlnet.depth_enabled is False
	assert config.restoration.enabled is False


def test_generation_config_defaults():
	config = GenerationConfig()

	assert config.motion.mode == "ken_burns_2d"
	assert config.render.strength == pytest.approx(0.35)
	assert config.lora.entries == []
	assert config.lora.max_active_loras == 3
	assert config.temporal.method == "none"
	assert config.output.fps == 8
	assert config.output.format == "mp4"
	assert config.seed is None


def test_garment_swap_config_defaults():
	config = GenerationConfig().garment

	assert config.sam.model_type == "vit_b"
	assert config.mask_dilation_px == 40
	assert config.mask_feather_px == 8
	assert config.include_arms_in_mask is True
	assert config.include_legs_in_mask is False
	assert config.inpaint_strength == pytest.approx(0.85)
	assert config.use_pose_controlnet is True


def test_settings_garment_env_override(monkeypatch):
	monkeypatch.setenv("HAROFRAME_GENERATION__GARMENT__MASK_DILATION_PX", "80")
	monkeypatch.setenv("HAROFRAME_GENERATION__GARMENT__SAM__MODEL_TYPE", "vit_h")

	settings = Settings()

	assert settings.generation.garment.mask_dilation_px == 80
	assert settings.generation.garment.sam.model_type == "vit_h"


def test_lora_entry_config_rejects_reserved_adapter_name():
	with pytest.raises(ValidationError):
		LoraEntryConfig(adapter_name="faceid", source="some/repo")


def test_lora_entry_config_accepts_other_names():
	entry = LoraEntryConfig(adapter_name="anime_style", source="some/repo", scale=0.7)

	assert entry.adapter_name == "anime_style"
	assert entry.enabled is True


def test_lora_config_rejects_too_many_enabled_entries():
	entries = [
		LoraEntryConfig(adapter_name=f"style_{i}", source="some/repo") for i in range(4)
	]

	with pytest.raises(ValidationError):
		LoraConfig(max_active_loras=3, entries=entries)


def test_lora_config_ignores_disabled_entries_when_counting():
	entries = [
		LoraEntryConfig(adapter_name="a", source="x", enabled=True),
		LoraEntryConfig(adapter_name="b", source="x", enabled=True),
		LoraEntryConfig(adapter_name="c", source="x", enabled=False),
		LoraEntryConfig(adapter_name="d", source="x", enabled=False),
	]

	config = LoraConfig(max_active_loras=2, entries=entries)

	assert len(config.entries) == 4


def test_settings_env_override(monkeypatch):
	monkeypatch.setenv("HAROFRAME_IDENTITY__DEVICE", "cpu")
	monkeypatch.setenv("HAROFRAME_IDENTITY__IPADAPTER__ENABLED", "true")
	monkeypatch.setenv("HAROFRAME_IDENTITY__FACE__MIN_DET_SCORE", "0.7")

	settings = Settings()

	assert settings.identity.device == "cpu"
	assert settings.identity.ipadapter.enabled is True
	assert settings.identity.face.min_det_score == pytest.approx(0.7)


def test_settings_generation_env_override(monkeypatch):
	monkeypatch.setenv("HAROFRAME_GENERATION__OUTPUT__FPS", "12")
	monkeypatch.setenv("HAROFRAME_GENERATION__MOTION__MODE", "static")

	settings = Settings()

	assert settings.generation.output.fps == 12
	assert settings.generation.motion.mode == "static"


def test_get_settings_is_cached():
	assert get_settings() is get_settings()
