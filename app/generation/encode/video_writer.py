from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.generation.exceptions import VideoEncodeError


class ImageioVideoEncoder:
	"""VideoEncoder backed by imageio's ffmpeg plugin (imageio-ffmpeg, bundled binary --
	no system ffmpeg install required)."""

	def __init__(self, codec: str = "libx264") -> None:
		self._codec = codec

	def encode(self, frames: list[Image.Image], fps: int, output_path: Path) -> Path:
		if not frames:
			raise VideoEncodeError("cannot encode an empty frame sequence")
		try:
			import imageio
		except ImportError as exc:
			raise VideoEncodeError(
				"imageio is not installed; cannot encode frames into a video file"
			) from exc

		output_path.parent.mkdir(parents=True, exist_ok=True)
		frame_arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
		try:
			imageio.mimwrite(str(output_path), frame_arrays, fps=fps, codec=self._codec)
		except Exception as exc:
			raise VideoEncodeError(f"failed to encode video to {output_path}: {exc}") from exc
		return output_path
