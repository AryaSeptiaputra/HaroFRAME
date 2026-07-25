from __future__ import annotations

from typing import Protocol

from PIL import Image

from app.identity.interfaces import FaceEmbedding, IdentityReference


class FaceAnalyzer(Protocol):
	"""Detects faces in an image and extracts identity embeddings from them."""

	def analyze(self, image: Image.Image) -> list[FaceEmbedding]:
		"""Return one embedding per detected face above the configured score threshold."""
		...

	def analyze_reference(self, reference: IdentityReference) -> IdentityReference:
		"""Populate ``reference.embeddings`` (best face per image), in place, and return it."""
		...
