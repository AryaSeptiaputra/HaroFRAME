from __future__ import annotations

from typing import Any, Protocol

from app.identity.interfaces import IdentityConditioning, IdentityReference, StructureHint


class InstantIdProvider(Protocol):
	"""Hybrid face-identity provider: ArcFace embedding + facial-keypoint ControlNet (IdentityNet).

	Structurally compatible with :class:`app.identity.interfaces.FaceConditioningProvider`,
	but ``pipeline`` must be the vendored ``StableDiffusionXLInstantIDPipeline`` (built via
	:meth:`build_pipeline`, with its IdentityNet ControlNet already attached) rather than an
	arbitrary diffusers pipeline.
	"""

	def load(self, pipeline: Any) -> None:
		"""Load the InstantID ip-adapter checkpoint onto ``pipeline`` (idempotent)."""
		...

	def build_conditioning(
		self,
		reference: IdentityReference,
		*,
		structure: StructureHint | None = None,
		scale: float = 0.8,
	) -> IdentityConditioning:
		"""Build the kwargs needed to condition a pipeline call on ``reference``."""
		...

	def unload(self, pipeline: Any) -> None:
		...
