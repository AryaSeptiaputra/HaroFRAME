from __future__ import annotations

import pytest

from app.core.config import IdentityConfig, Settings, get_settings


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


def test_settings_env_override(monkeypatch):
	monkeypatch.setenv("HAROFRAME_IDENTITY__DEVICE", "cpu")
	monkeypatch.setenv("HAROFRAME_IDENTITY__IPADAPTER__ENABLED", "true")
	monkeypatch.setenv("HAROFRAME_IDENTITY__FACE__MIN_DET_SCORE", "0.7")

	settings = Settings()

	assert settings.identity.device == "cpu"
	assert settings.identity.ipadapter.enabled is True
	assert settings.identity.face.min_det_score == pytest.approx(0.7)


def test_get_settings_is_cached():
	assert get_settings() is get_settings()
