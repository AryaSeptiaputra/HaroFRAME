from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from app.identity.interfaces import StructureHint


class StructureConditioner(Protocol):
	"""Turns a raw driving image (pose or depth source) into ControlNet conditioning.

	Deliberately independent of any face/identity provider — this is the "structure
	signal" half of the identity/structure split, so callers can enable it, disable
	it, or feed it a different driving image than the identity reference photo.
	"""

	def preprocess(self, image: Image.Image) -> Image.Image:
		"""Run the underlying detector/estimator and return the control image."""
		...

	def build_control(self, hint: StructureHint) -> dict[str, Any]:
		"""Return a pipeline kwargs fragment: control image, ControlNet model, scale."""
		...
