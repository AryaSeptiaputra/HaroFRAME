from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.core.config import IpAdapterConfig
from app.identity.exceptions import ModelLoadError, NoFaceDetectedError
from app.identity.face.fusion import fuse_embeddings
from app.identity.interfaces import IdentityConditioning, IdentityReference, StructureHint

_PROVIDER_NAME = "ip_adapter_faceid_sdxl"


class FaceIdSdxlProvider:
	"""IP-Adapter-FaceID (SDXL): conditions on an ArcFace identity embedding rather than
	raw pixels, so the generated face keeps the reference person's identity across very
	different poses/styles.

	Requires ``reference.embeddings`` (and ideally ``reference.fused_embedding`` when more
	than one reference photo is given) to already be populated by a
	:class:`~app.identity.face.interfaces.FaceAnalyzer` — this provider does not run face
	detection itself.
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
			subfolder=self._config.subfolder,
			weight_name=self._config.weight_name,
		)
		if self._config.lora_weight_name:
			# adapter_name is pinned so app/generation/lora's PeftLoraManager can
			# name this adapter explicitly in its own set_adapters() calls --
			# "faceid" is reserved for this purpose (see LoraEntryConfig).
			pipeline.load_lora_weights(
				self._config.repo_id, weight_name=self._config.lora_weight_name, adapter_name="faceid"
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
		embedding = reference.fused_embedding
		if embedding is None:
			if not reference.embeddings:
				raise NoFaceDetectedError(
					"reference has no face embeddings; run a FaceAnalyzer before building conditioning"
				)
			embedding = (
				reference.embeddings[0]
				if len(reference.embeddings) == 1
				else fuse_embeddings(reference.embeddings)
			)
		# diffusers' FaceID cross-attention expects a list of (1, embed_dim) tensors,
		# one per adapter image; shape conventions can shift slightly across diffusers
		# releases, so verify against the installed version if conditioning looks wrong.
		embeds_tensor = torch.from_numpy(np.asarray(embedding.vector, dtype=np.float32)).unsqueeze(0)
		return IdentityConditioning(
			adapter_kwargs={"ip_adapter_image_embeds": [embeds_tensor]},
			applied_adapters=[_PROVIDER_NAME],
			used_structure_conditioning=False,
		)

	def unload(self, pipeline: Any) -> None:
		if hasattr(pipeline, "unload_ip_adapter"):
			pipeline.unload_ip_adapter()
		self._loaded = False
