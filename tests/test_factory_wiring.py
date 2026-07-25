from __future__ import annotations

import pytest

from app.core.config import ControlNetConfig, IdentityConfig
from app.identity.controlnet.factory import build_structure_conditioners
from app.identity.controlnet.provider.depth import DepthConditioner
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.exceptions import ConflictingAdapterConfigError
from app.identity.factory import build_identity_engine
from app.identity.instantid.provider import InstantIdProvider
from app.identity.ipadapter.factory import build_ipadapter_provider
from app.identity.ipadapter.provider.clip_ipadapter import ClipIpAdapterProvider
from app.identity.ipadapter.provider.faceid_sdxl import FaceIdSdxlProvider


def test_build_ipadapter_provider_selects_clip_variant():
	config = IdentityConfig(ipadapter={"enabled": True, "variant": "clip"}).ipadapter

	assert isinstance(build_ipadapter_provider(config), ClipIpAdapterProvider)


def test_build_ipadapter_provider_selects_faceid_sdxl_variant():
	config = IdentityConfig(ipadapter={"enabled": True, "variant": "faceid_sdxl"}).ipadapter

	assert isinstance(build_ipadapter_provider(config), FaceIdSdxlProvider)


def test_build_ipadapter_provider_rejects_unknown_variant():
	config = IdentityConfig().ipadapter
	config.variant = "bogus"

	with pytest.raises(ValueError):
		build_ipadapter_provider(config)


def test_build_structure_conditioners_respects_flags():
	assert build_structure_conditioners(ControlNetConfig()) == []

	conditioners = build_structure_conditioners(ControlNetConfig(pose_enabled=True, depth_enabled=True))

	assert [type(c) for c in conditioners] == [DwPoseConditioner, DepthConditioner]


def test_build_identity_engine_rejects_conflicting_adapters():
	config = IdentityConfig(ipadapter={"enabled": True}, instantid={"enabled": True})

	with pytest.raises(ConflictingAdapterConfigError):
		build_identity_engine(config)


def test_build_identity_engine_picks_instantid_when_enabled():
	config = IdentityConfig(instantid={"enabled": True})

	engine = build_identity_engine(config)

	assert isinstance(engine.face_adapter, InstantIdProvider)


def test_build_identity_engine_no_adapter_when_none_enabled():
	engine = build_identity_engine(IdentityConfig())

	assert engine.face_adapter is None
