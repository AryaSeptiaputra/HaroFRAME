from __future__ import annotations

from typing import Any

import numpy as np

from app.core.config import InstantIdConfig
from app.identity.exceptions import ModelLoadError, NoFaceDetectedError
from app.identity.face.fusion import fuse_embeddings
from app.identity.instantid.vendor.pipeline_stable_diffusion_xl_instantid import draw_kps
from app.identity.interfaces import IdentityConditioning, IdentityReference, StructureHint

_PROVIDER_NAME = "instantid"


class InstantIdProvider:
	"""Hybrid identity provider: ArcFace embedding (via ip-adapter cross-attention) plus
	a facial-keypoint IdentityNet ControlNet, giving near-exact face fidelity across
	poses/styles without any per-identity fine-tuning.

	``pipeline`` passed to :meth:`load`/:meth:`unload` must be a
	``StableDiffusionXLInstantIDPipeline`` built by
	:func:`app.identity.instantid.pipeline.build_instantid_pipeline`, with the
	IdentityNet ControlNet already attached.
	"""

	def __init__(self, config: InstantIdConfig) -> None:
		self._config = config
		self._loaded = False

	def load(self, pipeline: Any) -> None:
		if self._loaded:
			return
		if not hasattr(pipeline, "load_ip_adapter_instantid"):
			raise ModelLoadError(
				"pipeline is not a StableDiffusionXLInstantIDPipeline; build one via "
				"app.identity.instantid.pipeline.build_instantid_pipeline() first"
			)
		weight_path = self._resolve_ip_adapter_weight()
		pipeline.load_ip_adapter_instantid(weight_path, scale=self._config.ip_adapter_scale)
		self._loaded = True

	def _resolve_ip_adapter_weight(self) -> str:
		try:
			from huggingface_hub import hf_hub_download
		except ImportError as exc:
			raise ModelLoadError("huggingface_hub is not installed; cannot fetch InstantID weights") from exc
		return hf_hub_download(
			repo_id=self._config.controlnet_repo_id,
			filename=self._config.ip_adapter_weight_name,
		)

	def build_conditioning(
		self,
		reference: IdentityReference,
		*,
		structure: StructureHint | None = None,
		scale: float = 0.8,
	) -> IdentityConditioning:
		if not reference.embeddings:
			raise NoFaceDetectedError(
				"reference has no face embeddings; run a FaceAnalyzer before building conditioning"
			)
		embedding = reference.fused_embedding
		if embedding is None:
			embedding = (
				reference.embeddings[0]
				if len(reference.embeddings) == 1
				else fuse_embeddings(reference.embeddings)
			)
		if embedding.landmarks_5pt is None:
			raise NoFaceDetectedError(
				"InstantID requires 5-point facial landmarks, which the FaceAnalyzer did not provide"
			)

		# The keypoint control map only needs *a* canvas of the right size; prefer an
		# explicit driving frame over the reference photo when one is given, since the
		# generated video frame's composition is what the ControlNet must line up with.
		canvas_image = None
		if structure is not None and structure.pose_image is not None:
			canvas_image = structure.pose_image
		elif reference.images:
			canvas_image = reference.images[0]
		if canvas_image is None:
			raise ModelLoadError("no image available to size the InstantID keypoint control map")

		kps_image = draw_kps(canvas_image, embedding.landmarks_5pt)
		image_embeds = np.asarray(embedding.vector, dtype=np.float32)

		return IdentityConditioning(
			adapter_kwargs={
				"image_embeds": image_embeds,
				"image": kps_image,
				"controlnet_conditioning_scale": scale,
			},
			applied_adapters=[_PROVIDER_NAME],
			used_structure_conditioning=True,
		)

	def unload(self, pipeline: Any) -> None:
		self._loaded = False
