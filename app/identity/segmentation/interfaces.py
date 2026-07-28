from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(slots=True, frozen=True)
class BodyKeypoints:
	"""18-point OpenPose/COCO body keypoints for one detected person, normalized to [0,1].

	Index order: 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist, 5 l_shoulder,
	6 l_elbow, 7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee,
	13 l_ankle, 14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear -- matches controlnet_aux's
	shared OpenPose/DWPose body-keypoint convention.
	"""

	points: np.ndarray  # (18, 2) float, normalized [0,1]; (-1, -1) sentinel = undetected
	scores: np.ndarray  # (18,) float confidence; -1 sentinel matches undetected points
	image_size: tuple[int, int]  # (W, H) the points were normalized against


@dataclass(slots=True, frozen=True)
class SamPromptSet:
	"""Point/box prompts in pixel coordinates, ready for SamPredictor.predict()."""

	points: np.ndarray  # (N, 2) pixel coords
	labels: np.ndarray  # (N,) 1=foreground, 0=background
	box: tuple[float, float, float, float] | None  # x0,y0,x1,y1 pixel coords


@dataclass(slots=True, frozen=True)
class GarmentMask:
	mask: Image.Image  # mode "L", 255 = region to inpaint, 0 = keep
	prompt_set: SamPromptSet


class GarmentMaskGenerator(Protocol):
	def generate_mask(self, image: Image.Image) -> GarmentMask:
		...
