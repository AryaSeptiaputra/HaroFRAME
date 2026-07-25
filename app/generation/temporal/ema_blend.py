from __future__ import annotations

import numpy as np
from PIL import Image

from app.generation.exceptions import GenerationModuleError


class EmaFrameSmoother:
	"""Motion-compensated exponential blend between consecutive frames, using
	dense optical flow (opencv's Farneback, already a core dependency) to warp
	the previous smoothed frame into the current frame's coordinate space before
	blending -- plain (non-motion-compensated) blending would just smear/ghost
	under the camera pan/zoom this module always applies.

	Not the default (see TemporalConfig.method="none") until validated on real
	output: too naive a blend can trade flicker for blur, which may be a worse
	trade depending on how strong the underlying per-frame render variance is.
	"""

	def __init__(self, smoothing_strength: float = 0.5) -> None:
		if not 0.0 <= smoothing_strength <= 1.0:
			raise GenerationModuleError(f"smoothing_strength must be in [0, 1], got {smoothing_strength}")
		self._alpha = smoothing_strength

	def smooth(self, frames: list[Image.Image]) -> list[Image.Image]:
		if len(frames) < 2:
			return list(frames)
		try:
			import cv2
		except ImportError as exc:
			raise GenerationModuleError("opencv-python is not installed; cannot run EmaFrameSmoother") from exc

		smoothed: list[Image.Image] = [frames[0]]
		prev_smoothed_bgr = _to_bgr(frames[0])
		prev_gray = cv2.cvtColor(prev_smoothed_bgr, cv2.COLOR_BGR2GRAY)

		for frame in frames[1:]:
			curr_bgr = _to_bgr(frame)
			curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			warped_prev_bgr = _warp_by_flow(cv2, prev_smoothed_bgr, flow)
			blended_bgr = cv2.addWeighted(curr_bgr, 1.0 - self._alpha, warped_prev_bgr, self._alpha, 0)

			smoothed.append(_from_bgr(blended_bgr))
			prev_smoothed_bgr = blended_bgr
			prev_gray = curr_gray

		return smoothed


def _to_bgr(image: Image.Image) -> np.ndarray:
	return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def _from_bgr(array: np.ndarray) -> Image.Image:
	return Image.fromarray(array[:, :, ::-1])


def _warp_by_flow(cv2, image_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
	height, width = flow.shape[:2]
	flow_map = -flow.copy()
	flow_map[..., 0] += np.arange(width)
	flow_map[..., 1] += np.arange(height)[:, None]
	return cv2.remap(image_bgr, flow_map.astype(np.float32), None, cv2.INTER_LINEAR)
