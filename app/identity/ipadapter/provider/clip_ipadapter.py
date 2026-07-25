from __future__ import annotations

from typing import Any

from app.core.config import IpAdapterConfig
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import IdentityConditioning, IdentityReference, StructureHint

_PROVIDER_NAME = "ip_adapter_clip"


class ClipIpAdapterProvider:
	"""Plain CLIP-image IP-Adapter: conditions on overall appearance/style/composition,
	not an identity embedding. Lower priority than FaceID/InstantID for identity fidelity,
	useful mainly for style transfer or as a lightweight baseline.
	"""

	def __init__(self, config: IpAdapterConfig) -> None:
		self._config = config
		self._loaded = False

	def load(self, pipeline: Any) -> None:
		if self._loaded:
			return
		if not hasattr(pipeline, "load_ip_adapter"):
			raise ModelLoadError(
				"pipeline does not support load_ip_adapter(); pass a diffusers DiffusionPipeline"
			)
		pipeline.load_ip_adapter(
			self._config.repo_id,
			subfolder=self._config.subfolder or "sdxl_models",
			weight_name=self._config.weight_name,
		)
		pipeline.set_ip_adapter_scale(self._config.scale)
		self._loaded = True

	def build_conditioning(
		self,
		reference: IdentityReference,
		*,
		structure: StructureHint | None = None,
		scale: float = 1.0,
	) -> IdentityConditioning:
		if not reference.images:
			raise ModelLoadError("no reference images provided for CLIP IP-Adapter conditioning")
		return IdentityConditioning(
			adapter_kwargs={"ip_adapter_image": reference.images[0]},
			applied_adapters=[_PROVIDER_NAME],
			used_structure_conditioning=False,
		)

	def unload(self, pipeline: Any) -> None:
		if hasattr(pipeline, "unload_ip_adapter"):
			pipeline.unload_ip_adapter()
		self._loaded = False
