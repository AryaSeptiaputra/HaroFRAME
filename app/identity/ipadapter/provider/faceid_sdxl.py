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
		self._dtype: Any = None

	def load(self, pipeline: Any) -> None:
		if self._loaded:
			return
		if not hasattr(pipeline, "load_ip_adapter"):
			raise ModelLoadError(
				"pipeline does not support load_ip_adapter(); pass a diffusers DiffusionPipeline"
			)
		pipeline.load_ip_adapter(
			self._config.repo_id,
			# "" not None: diffusers builds the CLIP path as
			# Path(subfolder, image_encoder_folder), which raises TypeError on None.
			subfolder=self._config.subfolder or "",
			weight_name=self._config.weight_name,
			# FaceID has no image encoder -- it conditions on the ArcFace vector that
			# build_conditioning() passes as ip_adapter_image_embeds, and
			# h94/IP-Adapter-FaceID ships no image_encoder/ folder to load one from.
			# None is diffusers' documented way to skip it (it warns that
			# ip_adapter_image is then unavailable, which is exactly right here).
			image_encoder_folder=None,
		)
		if self._config.lora_weight_name:
			# adapter_name is pinned so app/generation/lora's PeftLoraManager can
			# name this adapter explicitly in its own set_adapters() calls --
			# "faceid" is reserved for this purpose (see LoraEntryConfig).
			pipeline.load_lora_weights(
				self._config.repo_id, weight_name=self._config.lora_weight_name, adapter_name="faceid"
			)
		pipeline.set_ip_adapter_scale(self._config.scale)
		# diffusers' prepare_ip_adapter_image_embeds() only moves supplied embeds
		# to the pipeline's device -- it never casts them -- so build_conditioning()
		# has to hand back the pipeline's own dtype or fp32 embeds meet an fp16 UNet.
		self._dtype = getattr(pipeline, "dtype", None)
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
		# MultiIPAdapterImageProjection documents each tensor as
		# [batch_size, num_images, embed_dim] -- so (2, 1, 512) here, not the bare
		# (1, 512) an unsqueeze gives; check_inputs() rejects anything below 3D.
		#
		# The batch dim is 2 because prepare_ip_adapter_image_embeds() chunks it in
		# half into (negative, positive) when classifier-free guidance is on: the
		# zeroed negative half has to be supplied from here. That assumes CFG, which
		# every guidance_scale default in this project uses; at guidance_scale <= 1
		# diffusers would skip the chunk and read all of this as the positive half.
		identity = torch.from_numpy(np.asarray(embedding.vector, dtype=np.float32)).reshape(1, 1, -1)
		embeds_tensor = torch.cat([torch.zeros_like(identity), identity], dim=0)
		if self._dtype is not None:
			embeds_tensor = embeds_tensor.to(dtype=self._dtype)
		return IdentityConditioning(
			adapter_kwargs={"ip_adapter_image_embeds": [embeds_tensor]},
			applied_adapters=[_PROVIDER_NAME],
			used_structure_conditioning=False,
		)

	def unload(self, pipeline: Any) -> None:
		if hasattr(pipeline, "unload_ip_adapter"):
			pipeline.unload_ip_adapter()
		self._loaded = False
		self._dtype = None
