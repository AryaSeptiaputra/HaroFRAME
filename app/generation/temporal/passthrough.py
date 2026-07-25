from __future__ import annotations

from PIL import Image


class NullTemporalSmoother:
	"""No-op TemporalSmoother -- the default until EmaFrameSmoother is validated
	to actually reduce flicker rather than just adding blur (see EmaFrameSmoother's
	docstring)."""

	def smooth(self, frames: list[Image.Image]) -> list[Image.Image]:
		return list(frames)
