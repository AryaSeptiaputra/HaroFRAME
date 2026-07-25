from __future__ import annotations

from app.core.config import RestorationConfig
from app.identity.restoration.gfpgan_restorer import GfpganRestorer
from app.identity.restoration.interfaces import FaceRestorer

_BACKENDS = {
	"gfpgan": GfpganRestorer,
}


def build_face_restorer(config: RestorationConfig) -> FaceRestorer | None:
	"""Build the configured FaceRestorer, or ``None`` when restoration is disabled."""
	if not config.enabled:
		return None
	try:
		backend_cls = _BACKENDS[config.backend]
	except KeyError as exc:
		raise ValueError(f"unknown restoration backend: {config.backend!r}") from exc
	return backend_cls(config)
