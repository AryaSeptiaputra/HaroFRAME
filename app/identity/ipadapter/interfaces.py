from __future__ import annotations

from typing import Any, Protocol

from app.identity.interfaces import IdentityConditioning, IdentityReference, StructureHint


class IdentityAdapter(Protocol):
	"""Applies an IP-Adapter-family provider (plain CLIP or FaceID) to a diffusers pipeline.

	Structurally compatible with :class:`app.identity.interfaces.FaceConditioningProvider`.
	"""

	def load(self, pipeline: Any) -> None:
		"""Register this adapter's weights onto ``pipeline`` (idempotent)."""
		...

	def build_conditioning(
		self,
		reference: IdentityReference,
		*,
		structure: StructureHint | None = None,
		scale: float = 1.0,
	) -> IdentityConditioning:
		"""Build the kwargs needed to condition a pipeline call on ``reference``."""
		...

	def unload(self, pipeline: Any) -> None:
		"""Remove this adapter's weights from ``pipeline``."""
		...
