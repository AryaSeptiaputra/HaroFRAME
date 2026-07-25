from __future__ import annotations

from app.core.config import IdentityConfig
from app.identity.controlnet.factory import build_structure_conditioners
from app.identity.engine import IdentityEngine
from app.identity.exceptions import ConflictingAdapterConfigError
from app.identity.face.analyzer_insightface import InsightFaceAnalyzer
from app.identity.instantid.factory import build_instantid_provider
from app.identity.interfaces import FaceConditioningProvider
from app.identity.ipadapter.factory import build_ipadapter_provider
from app.identity.restoration.factory import build_face_restorer


def build_identity_engine(config: IdentityConfig) -> IdentityEngine:
	"""Wire up an :class:`IdentityEngine` from top-level identity config.

	Exactly one of ``config.ipadapter`` / ``config.instantid`` may be enabled at a
	time -- both being enabled is rejected rather than silently picking one, since
	running two primary face adapters together isn't a supported configuration.
	"""
	if config.ipadapter.enabled and config.instantid.enabled:
		raise ConflictingAdapterConfigError(
			"both identity.ipadapter and identity.instantid are enabled; enable only one"
		)

	face_adapter: FaceConditioningProvider | None = None
	if config.ipadapter.enabled:
		face_adapter = build_ipadapter_provider(config.ipadapter)
	elif config.instantid.enabled:
		face_adapter = build_instantid_provider(config.instantid)

	return IdentityEngine(
		config,
		face_analyzer=InsightFaceAnalyzer(config.face),
		face_adapter=face_adapter,
		structure_conditioners=build_structure_conditioners(config.controlnet),
		face_restorer=build_face_restorer(config.restoration),
	)
