from __future__ import annotations

from typing import Protocol

from PIL import Image


class FaceRestorer(Protocol):
	"""Post-hoc face restoration/enhancement, run on an already-generated frame.

	This fixes residual artifacts (blur, warped features) the diffusion pipeline
	leaves behind; it does not enforce identity itself — that's the job of the
	ipadapter/instantid providers upstream.
	"""

	def restore(self, image: Image.Image) -> Image.Image:
		"""Return a restored copy of ``image``; must not mutate the input."""
		...
