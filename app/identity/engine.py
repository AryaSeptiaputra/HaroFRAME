from __future__ import annotations

from typing import Any

from app.core.config import IdentityConfig
from app.identity.controlnet.interfaces import StructureConditioner
from app.identity.exceptions import ConflictingAdapterConfigError
from app.identity.face.fusion import fuse_embeddings
from app.identity.face.interfaces import FaceAnalyzer
from app.identity.interfaces import (
	FaceConditioningProvider,
	IdentityConditioning,
	IdentityReference,
	StructureHint,
)
from app.identity.restoration.interfaces import FaceRestorer


class IdentityEngine:
	"""Orchestrates identity-preserving conditioning end to end: face analysis and
	embedding fusion, one primary face adapter (IP-Adapter-family or InstantID),
	optional structure conditioners (pose/depth ControlNets), and optional post-hoc
	face restoration.

	Holds a single ``face_adapter`` slot because IP-Adapter-family providers and
	InstantID both satisfy :class:`FaceConditioningProvider`, and running two
	primary face adapters at once isn't a supported configuration -- see
	:class:`~app.identity.exceptions.ConflictingAdapterConfigError`.
	"""

	def __init__(
		self,
		config: IdentityConfig,
		*,
		face_analyzer: FaceAnalyzer,
		face_adapter: FaceConditioningProvider | None,
		structure_conditioners: list[StructureConditioner],
		face_restorer: FaceRestorer | None,
	) -> None:
		self._config = config
		self.face_analyzer = face_analyzer
		self.face_adapter = face_adapter
		self.structure_conditioners = structure_conditioners
		self.face_restorer = face_restorer

	def prepare_reference(self, reference: IdentityReference) -> IdentityReference:
		"""Run face analysis and embedding fusion, populating ``reference`` in place."""
		self.face_analyzer.analyze_reference(reference)
		reference.fused_embedding = (
			reference.embeddings[0]
			if len(reference.embeddings) == 1
			else fuse_embeddings(reference.embeddings, strategy=self._config.face.fusion_strategy)
		)
		return reference

	def load(self, pipeline: Any) -> None:
		"""Attach the configured face adapter's weights onto ``pipeline``."""
		if self.face_adapter is not None:
			self.face_adapter.load(pipeline)

	def unload(self, pipeline: Any) -> None:
		if self.face_adapter is not None:
			self.face_adapter.unload(pipeline)

	def build_conditioning(
		self,
		reference: IdentityReference,
		*,
		structure: StructureHint | None = None,
		scale: float = 1.0,
	) -> IdentityConditioning:
		"""Build one combined kwargs bundle: the face adapter's conditioning, plus
		any structure conditioners' ControlNet kwargs merged in."""
		if self.face_adapter is None:
			raise ConflictingAdapterConfigError(
				"no face adapter configured; enable identity.ipadapter or identity.instantid"
			)
		conditioning = self.face_adapter.build_conditioning(reference, structure=structure, scale=scale)

		# Skip separate structure conditioners when the face adapter already carries
		# its own structure signal (e.g. InstantID's keypoint ControlNet) -- both
		# would otherwise emit colliding "controlnet"/"control_image" kwargs.
		if structure is not None and self.structure_conditioners and not conditioning.used_structure_conditioning:
			controls = [c.build_control(structure) for c in self.structure_conditioners]
			if len(controls) == 1:
				conditioning.adapter_kwargs.update(controls[0])
			else:
				# diffusers' MultiControlNetModel convention: parallel lists.
				conditioning.adapter_kwargs["controlnet"] = [c["controlnet"] for c in controls]
				conditioning.adapter_kwargs["control_image"] = [c["control_image"] for c in controls]
				conditioning.adapter_kwargs["controlnet_conditioning_scale"] = [
					c["controlnet_conditioning_scale"] for c in controls
				]
			conditioning.used_structure_conditioning = True

		if self.face_restorer is not None:
			conditioning.restorer_hook = self.face_restorer.restore

		return conditioning
