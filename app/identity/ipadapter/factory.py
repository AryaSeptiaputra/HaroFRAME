from __future__ import annotations

from app.core.config import IpAdapterConfig
from app.identity.ipadapter.interfaces import IdentityAdapter
from app.identity.ipadapter.provider.clip_ipadapter import ClipIpAdapterProvider
from app.identity.ipadapter.provider.faceid_sdxl import FaceIdSdxlProvider

_PROVIDERS = {
	"clip": ClipIpAdapterProvider,
	"faceid_sdxl": FaceIdSdxlProvider,
}


def build_ipadapter_provider(config: IpAdapterConfig) -> IdentityAdapter:
	try:
		provider_cls = _PROVIDERS[config.variant]
	except KeyError as exc:
		raise ValueError(f"unknown ip-adapter variant: {config.variant!r}") from exc
	return provider_cls(config)
