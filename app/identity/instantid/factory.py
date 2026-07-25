from __future__ import annotations

from app.core.config import InstantIdConfig
from app.identity.instantid.interfaces import InstantIdProvider as InstantIdProviderProtocol
from app.identity.instantid.provider import InstantIdProvider


def build_instantid_provider(config: InstantIdConfig) -> InstantIdProviderProtocol:
	return InstantIdProvider(config)
